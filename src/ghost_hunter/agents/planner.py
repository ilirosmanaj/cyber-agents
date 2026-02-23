"""Planner agent — LLM decides scan focus, skippable agents, and extra paths to try.

NOT registered in AGENT_DEPS — the orchestrator calls it explicitly at decision points:
  1. After wave 1 (web_crawler) — initial strategy
  2. After wave 2 (api_discovery + js_analyzer) — refined strategy
"""

from __future__ import annotations

import logging

from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.models import (
    AgentResult,
    Finding,
    RiskLevel,
    ScanState,
)
from src.ghost_hunter.models.strategy import ScanStrategy

logger = logging.getLogger(__name__)

_MAX_ENDPOINT_CONTEXT = 60

PLANNER_SYSTEM_PROMPT = """\
You are a penetration test lead planning the next phase of a security scan.

Given the current scan state, decide:
1. What type of application is this? (SPA, API-only, CMS, banking app...)
2. Where is the highest-value attack surface?
3. Which agents should be skipped? (e.g., skip js_analyzer if no JS found)
4. What extra paths should we probe that standard lists miss?
5. What technology hypotheses can we form from the evidence?

Valid agents that can be skipped: js_analyzer, hypothesis, api_discovery

Respond with JSON:
{
  "focus_areas": ["area1", "area2"],
  "skip_agents": [],
  "extra_paths_to_try": ["/path1", "/path2"],
  "tech_hypotheses": ["hypothesis1"],
  "scan_depth": "normal",
  "priority_patterns": ["pattern1"]
}

Be conservative with skip_agents — only skip when there's clear evidence an agent would be wasteful.\
"""


class PlannerAgent(BaseAgent):
    """Plans scan strategy at decision points. Called explicitly by orchestrator."""

    name = "planner"
    description = "Analyzes scan state at decision points and outputs a strategy for downstream agents."

    async def run(self, state: ScanState) -> AgentResult:
        findings: list[Finding] = []
        errors: list[str] = []

        tech_context = self.build_tech_context(state)
        insights_context = self.build_insights_context(state)

        endpoint_list = sorted(state.endpoints.keys())[:_MAX_ENDPOINT_CONTEXT]
        endpoint_context = "\n".join(f"  {k}" for k in endpoint_list) or "None yet"

        prior_strategy_context = ""
        if state.scan_strategy is not None:
            prior_strategy_context = (
                f"\nPrior Strategy (refine, don't start over):\n"
                f"  Focus: {', '.join(state.scan_strategy.focus_areas)}\n"
                f"  Skip: {', '.join(state.scan_strategy.skip_agents) or 'none'}\n"
                f"  Extra paths: {', '.join(state.scan_strategy.extra_paths_to_try[:10]) or 'none'}\n"
                f"  Tech hypotheses: {', '.join(state.scan_strategy.tech_hypotheses) or 'none'}\n"
                f"  Depth: {state.scan_strategy.scan_depth}\n"
            )

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Target: {state.target}\n\n"
                    f"Scan State:\n{state.summary()}\n\n"
                    f"Prior Insights:\n{insights_context}\n\n"
                    f"Tech Stack:\n{tech_context}\n\n"
                    f"Endpoints ({len(endpoint_list)}):\n{endpoint_context}"
                    f"{prior_strategy_context}"
                ),
            },
        ]

        try:
            strategy = await self.llm.chat_structured(
                messages, response_model=ScanStrategy,
                name="planner_strategy", max_tokens=2048,
            )
            state.scan_strategy = strategy

            detail_parts = []
            if strategy.focus_areas:
                detail_parts.append(f"Focus: {', '.join(strategy.focus_areas)}")
            if strategy.skip_agents:
                detail_parts.append(f"Skip: {', '.join(strategy.skip_agents)}")
            if strategy.extra_paths_to_try:
                detail_parts.append(f"Extra paths: {len(strategy.extra_paths_to_try)}")
            if strategy.tech_hypotheses:
                detail_parts.append(f"Tech: {', '.join(strategy.tech_hypotheses)}")
            detail_parts.append(f"Depth: {strategy.scan_depth}")

            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="scan_strategy",
                    title="Scan strategy determined",
                    detail="; ".join(detail_parts),
                    severity=RiskLevel.INFO,
                )
            )

        except Exception as e:
            errors.append(f"Planner failed: {e}")
            logger.warning("Planner agent failed (continuing without strategy): %s", e)

        return AgentResult(
            agent_name=self.name,
            success=len(errors) == 0,
            findings=findings,
            errors=errors,
        )
