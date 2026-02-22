"""Orchestrator — runs agents in dependency order with parallel waves."""

from __future__ import annotations

import asyncio
import logging

from src.ghost_hunter.agents import get_agent_registry
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.clients import AdaptiveHttpClient, LLMClient, trace_span
from src.ghost_hunter.models import AgentResult, ScanState
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
    "prioritizer": ["vuln_analyzer"],
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
        self._agents: dict[str, BaseAgent] = {
            name: cls(http_client=http_client, llm_client=llm_client)
            for name, cls in get_agent_registry().items()
            if name in AGENT_DEPS
        }

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

        return self.state

    def _should_skip(self, name: str) -> bool:
        if name == "js_analyzer" and not self.state.js_urls:
            return True
        return False

    async def _run_agent(self, agent_name: str) -> AgentResult:
        agent = self._agents[agent_name]
        logger.info("Running agent: %s", agent_name)
        return await agent.execute(self.state)
