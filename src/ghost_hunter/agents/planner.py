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
            {"role": "system", "content": self.prompt_registry.get("planner").system_prompt},
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

            logger.info(
                "Strategy: focus=[%s], skip=[%s], depth=%s",
                ", ".join(strategy.focus_areas),
                ", ".join(strategy.skip_agents) or "none",
                strategy.scan_depth,
            )

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
