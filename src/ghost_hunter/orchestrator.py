"""Orchestrator — runs agents in dependency order with parallel waves."""

from __future__ import annotations

import asyncio
import logging

from src.ghost_hunter.agents import get_agent_registry
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.agents.planner import PlannerAgent
from src.ghost_hunter.agents.verifier import MIN_INDICATORS_FOR_VERIFICATION
from src.ghost_hunter.clients import AdaptiveHttpClient, LLMClient, trace_span
from src.ghost_hunter.models import AgentResult, ScanState
from src.ghost_hunter.models.insights import ScanInsight
from src.ghost_hunter.models.llm_responses import ReanalysisRequest
from src.ghost_hunter.output import print_agent_step
from src.ghost_hunter.prompts import PromptRegistry

logger = logging.getLogger(__name__)

AGENT_DEPS: dict[str, list[str]] = {
    "passive_recon": [],
    "web_crawler": ["passive_recon"],
    "api_discovery": ["web_crawler"],
    "js_analyzer": ["web_crawler"],
    "hypothesis": ["api_discovery", "js_analyzer"],
    "classifier": ["hypothesis"],
    "vuln_analyzer": ["classifier"],
    "verifier": ["vuln_analyzer"],
    "prioritizer": ["verifier"],
}


def _resolve_waves(deps: dict[str, list[str]]) -> list[list[str]]:
    """Topologically sort agents into parallel execution waves."""
    remaining = {name: set(d) for name, d in deps.items()}
    waves: list[list[str]] = []

    while remaining:
        wave = [name for name, d in remaining.items() if not d]
        if not wave:
            raise ValueError(f"Circular dependency detected: {remaining}")
        waves.append(sorted(wave))
        for name in wave:
            del remaining[name]
        for d in remaining.values():
            d -= set(wave)

    return waves


class Orchestrator:
    """Runs agents in dependency order with parallel waves."""

    def __init__(
        self,
        http_client: AdaptiveHttpClient,
        llm_client: LLMClient,
        state: ScanState,
    ):
        self.state = state
        self._llm_client = llm_client
        self._prompt_registry = PromptRegistry()
        self._planner = PlannerAgent(
            http_client=http_client,
            llm_client=llm_client,
            prompt_registry=self._prompt_registry,
        )
        self._agents: dict[str, BaseAgent] = {
            name: cls(
                http_client=http_client,
                llm_client=llm_client,
                prompt_registry=self._prompt_registry,
            )
            for name, cls in get_agent_registry().items()
            if name in AGENT_DEPS
        }

    @property
    def prompt_registry(self) -> PromptRegistry:
        return self._prompt_registry

    _PLANNER_WAVE_TRIGGERS = {"web_crawler", "api_discovery", "js_analyzer"}

    async def run(self) -> ScanState:
        """Execute the full scan via DAG-ordered parallel waves."""
        with trace_span("orchestrator", metadata={"target": self.state.target}):
            for wave in _resolve_waves(AGENT_DEPS):
                skip_set = {name for name in wave if self._should_skip(name)}
                runnable = [name for name in wave if name not in skip_set]

                for name in skip_set:
                    logger.info("Skipping agent: %s", name)
                    self.state.agents_completed.append(name)

                if not runnable:
                    continue

                logger.info("Starting wave: [%s]", ", ".join(runnable))
                results = await asyncio.gather(
                    *[self._run_agent(name) for name in runnable]
                )
                for name, result in zip(runnable, results):
                    self.state.merge_agent_result(result)
                    print_agent_step(agent_name=name, reason="", result=result)

                # Generate insight after each wave (skip wave 0 — passive_recon)
                if "passive_recon" not in runnable:
                    await self._generate_wave_insight(runnable)

                # Invoke planner at decision points
                if set(runnable) & self._PLANNER_WAVE_TRIGGERS:
                    await self._run_planner()

                if "verifier" in runnable:
                    await self._handle_reanalysis()

        logger.info(
            "Scan complete - %d endpoints, %d findings",
            len(self.state.endpoints),
            len(self.state.findings),
        )
        return self.state

    def _should_skip(self, name: str) -> bool:
        if name == "js_analyzer" and not self.state.js_urls:
            return True
        if name == "verifier":
            active_count = sum(
                1
                for inds in self.state.vuln_indicators.values()
                for ind in inds
                if not ind.suppressed
            )
            if active_count < MIN_INDICATORS_FOR_VERIFICATION:
                return True
        if (
            self.state.scan_strategy is not None
            and name in self.state.scan_strategy.skip_agents
        ):
            return True
        return False

    async def _run_planner(self) -> None:
        """Invoke the planner agent to set/refine scan strategy. Non-fatal on failure."""
        try:
            result = await self._planner.execute(self.state)
            self.state.merge_agent_result(result)
            print_agent_step(agent_name="planner", reason="strategy", result=result)
        except Exception as e:
            logger.warning("Planner invocation failed (continuing): %s", e)

    async def _generate_wave_insight(self, agents_in_wave: list[str]) -> None:
        """Generate a brief LLM insight after a wave completes. Non-fatal on failure."""
        phase = ", ".join(agents_in_wave)
        try:
            messages = [
                {
                    "role": "system",
                    "content": self._prompt_registry.get("insight").system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"Phase just completed: {phase}\n\n"
                        f"Scan state:\n{self.state.summary()}\n\n"
                        f"Prior insights:\n{self.state.insights_context()}"
                    ),
                },
            ]
            insight = await self._llm_client.chat_structured(
                messages, response_model=ScanInsight,
                name=f"insight_{phase.replace(', ', '_')}", max_tokens=512,
            )
            insight.phase = phase
            self.state.scan_insights.append(insight)
            logger.info("Insight generated for phase: %s", phase)
        except Exception as e:
            logger.warning("Insight generation failed for phase %s (continuing): %s", phase, e)

    async def _handle_reanalysis(self) -> None:
        """Run targeted re-analysis for endpoints flagged by the verifier."""
        pending = [
            r for r in self.state.reanalysis_requests
            if r.endpoint_key not in self.state.reanalyzed_keys
        ]
        if not pending:
            return

        by_agent: dict[str, list[ReanalysisRequest]] = {}
        for req in pending:
            by_agent.setdefault(req.target_agent, []).append(req)

        for agent_name, reqs in by_agent.items():
            agent = self._agents.get(agent_name)
            if agent is None:
                logger.warning("Reanalysis target agent not found: %s", agent_name)
                continue

            logger.info(
                "Re-analyzing %d endpoints via %s",
                len(reqs), agent_name,
            )
            try:
                result = await agent.execute(self.state)
                self.state.merge_agent_result(result)
                print_agent_step(
                    agent_name=agent_name,
                    reason="reanalysis",
                    result=result,
                )
            except Exception as e:
                logger.warning("Reanalysis via %s failed: %s", agent_name, e)

            for req in reqs:
                self.state.reanalyzed_keys.add(req.endpoint_key)

    async def _run_agent(self, agent_name: str) -> AgentResult:
        agent = self._agents[agent_name]
        logger.info("Running agent: %s", agent_name)
        return await agent.execute(self.state)
