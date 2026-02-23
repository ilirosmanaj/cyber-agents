"""Verifier agent — checks findings for cross-cutting contradictions."""

from __future__ import annotations

import logging

from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.models import (
    AgentResult,
    Finding,
    RiskLevel,
    ScanState,
    VulnIndicator,
)
from src.ghost_hunter.models.llm_responses import VerifierBatchResponse

logger = logging.getLogger(__name__)

MIN_INDICATORS_FOR_VERIFICATION = 3
_MAX_INDICATORS_PER_BATCH = 40
_MAX_VERIFIER_CALLS = 2
_MAX_ENDPOINT_CONTEXT = 60



@register_agent
class VerifierAgent(BaseAgent):
    """Checks all vuln indicators for contradictions and false positives."""

    name = "verifier"
    description = "Reviews vuln indicators holistically to catch contradictions the per-endpoint pass missed."

    async def run(self, state: ScanState) -> AgentResult:
        findings: list[Finding] = []
        errors: list[str] = []

        all_indicators = [
            (key, ind)
            for key, inds in state.vuln_indicators.items()
            for ind in inds
            if not ind.suppressed
        ]

        if len(all_indicators) < MIN_INDICATORS_FOR_VERIFICATION:
            return AgentResult(
                agent_name=self.name,
                success=True,
                findings=[
                    Finding(
                        agent_name=self.name,
                        finding_type="verification_skipped",
                        title="Verification skipped — too few indicators",
                        detail=f"Only {len(all_indicators)} active indicators; minimum {MIN_INDICATORS_FOR_VERIFICATION} required.",
                        severity=RiskLevel.INFO,
                    )
                ],
            )

        tech_context = self.build_tech_context(state)
        insights_context = self.build_insights_context(state)
        indicator_context = self._format_indicators(state)
        classifier_context = self._format_classifications(state)

        suppressed_count = 0
        adjusted_count = 0
        annotated_count = 0
        cross_cutting_notes: list[str] = []

        for batch_idx in range(_MAX_VERIFIER_CALLS):
            start = batch_idx * _MAX_INDICATORS_PER_BATCH
            batch_indicators = all_indicators[start : start + _MAX_INDICATORS_PER_BATCH]
            if not batch_indicators:
                break

            batch_context = self._format_indicator_batch(batch_indicators)

            messages = [
                {"role": "system", "content": self.prompt_registry.get("verifier").system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Target: {state.target}\n\n"
                        f"Prior Insights:\n{insights_context}\n\n"
                        f"Tech Stack:\n{tech_context}\n\n"
                        f"Endpoint Classifications:\n{classifier_context}\n\n"
                        f"Vulnerability Indicators to review:\n{batch_context}\n\n"
                        f"Full indicator context:\n{indicator_context}"
                    ),
                },
            ]

            try:
                response = await self.llm.chat_structured(
                    messages, response_model=VerifierBatchResponse,
                    name=f"verifier_batch_{batch_idx}", max_tokens=2048,
                )

                actions_as_dicts = [a.model_dump() for a in response.actions]
                s, a, n = self._apply_actions(state, actions_as_dicts)
                suppressed_count += s
                adjusted_count += a
                annotated_count += n
                cross_cutting_notes.extend(response.cross_cutting_notes)
                state.reanalysis_requests.extend(response.reanalysis_requests)

            except Exception as e:
                errors.append(f"Verification batch {batch_idx} failed: {e}")
                logger.warning("Verifier LLM call failed: %s", e)

        detail_parts = []
        if suppressed_count:
            detail_parts.append(f"{suppressed_count} suppressed")
        if adjusted_count:
            detail_parts.append(f"{adjusted_count} confidence adjusted")
        if annotated_count:
            detail_parts.append(f"{annotated_count} annotated")
        if cross_cutting_notes:
            detail_parts.append(
                f"Cross-cutting: {'; '.join(cross_cutting_notes[:3])}"
            )

        findings.append(
            Finding(
                agent_name=self.name,
                finding_type="verification_complete",
                title=f"Verified {len(all_indicators)} indicators",
                detail="; ".join(detail_parts) if detail_parts else "No changes needed",
                severity=RiskLevel.INFO,
                verification_status="consistent" if not suppressed_count else "conflicting",
            )
        )

        return AgentResult(
            agent_name=self.name,
            success=len(errors) == 0,
            findings=findings,
            errors=errors,
        )

    @staticmethod
    def _format_indicators(state: ScanState) -> str:
        """Format all vuln indicators grouped by endpoint."""
        lines: list[str] = []
        for key, indicators in state.vuln_indicators.items():
            active = [ind for ind in indicators if not ind.suppressed]
            if not active:
                continue
            lines.append(f"\n{key}:")
            for ind in active:
                lines.append(
                    f"  - {ind.pattern.value} ({ind.confidence.value}): {ind.evidence}"
                )
        return "\n".join(lines) if lines else "No indicators"

    @staticmethod
    def _format_indicator_batch(
        batch: list[tuple[str, VulnIndicator]]
    ) -> str:
        """Format a batch of indicators for the LLM prompt."""
        lines: list[str] = []
        for key, ind in batch:
            lines.append(
                f"  {key} | {ind.pattern.value} ({ind.confidence.value}) | "
                f"{ind.evidence}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_classifications(state: ScanState) -> str:
        """Format endpoint classifications for cross-reference."""
        lines: list[str] = []
        for key, ep in list(state.endpoints.items())[:_MAX_ENDPOINT_CONTEXT]:
            cat = ep.category.value if ep.category else "unclassified"
            auth = (
                "requires_auth"
                if ep.requires_auth
                else "no_auth" if ep.requires_auth is False
                else "auth_unknown"
            )
            lines.append(f"  {key} | {cat} | {auth}")
        return "\n".join(lines) if lines else "No classifications"

    @staticmethod
    def _apply_actions(
        state: ScanState, actions: list[dict]
    ) -> tuple[int, int, int]:
        """Apply verifier actions to state. Returns (suppressed, adjusted, annotated)."""
        suppressed = 0
        adjusted = 0
        annotated = 0

        for action_data in actions:
            ep_key = action_data.get("endpoint_key", "")
            pattern = action_data.get("pattern", "")
            action = action_data.get("action", "")
            reason = action_data.get("reason", "")

            indicators = state.vuln_indicators.get(ep_key, [])
            if not indicators:
                continue

            for ind in indicators:
                if ind.pattern.value != pattern or ind.suppressed:
                    continue

                if action == "suppress":
                    ind.suppressed = True
                    ind.description += f" [SUPPRESSED by verifier: {reason}]"
                    suppressed += 1
                    break

                elif action == "adjust_confidence":
                    new_conf = action_data.get("new_confidence", "")
                    try:
                        ind.confidence = RiskLevel(new_conf)
                        ind.llm_enhanced = True
                        ind.description += f" [Verifier adjusted: {reason}]"
                        adjusted += 1
                    except ValueError:
                        pass
                    break

                elif action == "annotate":
                    ind.description += f" [Verifier note: {reason}]"
                    annotated += 1
                    break

        return suppressed, adjusted, annotated
