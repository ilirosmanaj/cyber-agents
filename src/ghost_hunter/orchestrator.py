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
from src.ghost_hunter.output import print_agent_step

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
        self._planner = PlannerAgent(http_client=http_client, llm_client=llm_client)
        self._agents: dict[str, BaseAgent] = {
            name: cls(http_client=http_client, llm_client=llm_client)
            for name, cls in get_agent_registry().items()
            if name in AGENT_DEPS
        }

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
                    "content": (
                        "You are reviewing scan progress. Summarize what was learned in 2-3 sentences. "
                        "Identify key signals that should inform the next analysis phase.\n\n"
                        "Respond with JSON:\n"
                        '{"summary": "...", "key_signals": [...], "recommended_focus": [...]}'
                    ),
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
            data = await self._llm_client.chat_json(
                messages, name=f"insight_{phase.replace(', ', '_')}", max_tokens=512
            )
            insight = ScanInsight(
                phase=phase,
                summary=data.get("summary", ""),
                key_signals=data.get("key_signals", []),
                recommended_focus=data.get("recommended_focus", []),
            )
            self.state.scan_insights.append(insight)
            logger.info("Insight generated for phase: %s", phase)
        except Exception as e:
            logger.warning("Insight generation failed for phase %s (continuing): %s", phase, e)

    async def _run_agent(self, agent_name: str) -> AgentResult:
        agent = self._agents[agent_name]
        logger.info("Running agent: %s", agent_name)
        return await agent.execute(self.state)
