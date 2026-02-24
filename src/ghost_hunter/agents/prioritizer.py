"""Prioritizer agent — LLM ranks endpoints by attack priority with rationale."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.config import settings
from src.ghost_hunter.models import (
    AgentResult,
    AttackSurfaceEntry,
    EndpointCategory,
    Endpoint,
    Finding,
    RiskLevel,
    ScanState,
    VulnIndicator,
)
from src.ghost_hunter.models.llm_responses import PrioritizationBatchResponse
from src.ghost_hunter.report import _METADATA_PATH_PREFIXES

logger = logging.getLogger(__name__)

PRIORITIZER_BATCH_SIZE = 25

# severity ordering for deterministic fallback — lower index = higher severity
_SEVERITY_ORDER = {level: idx for idx, level in enumerate(RiskLevel)}

# categories that get INFO risk when no vuln indicators exist
_LOW_RISK_CATEGORIES = {
    EndpointCategory.STATIC_ASSET,
    EndpointCategory.HEALTH_CHECK,
    EndpointCategory.DOCUMENTATION,
}

# max response body snippet chars included in prioritizer context
_MAX_SNIPPET_IN_CONTEXT = 300


def _is_metadata_path(url: str) -> bool:
    """True if the URL is a cloud metadata probe artifact (not a real endpoint)."""
    path = urlparse(url).path
    return any(path.startswith(prefix) for prefix in _METADATA_PATH_PREFIXES)


def _highest_severity(indicators: list[VulnIndicator]) -> RiskLevel:
    """Return the highest severity from a list of indicators, defaulting to INFO."""
    active = [ind for ind in indicators if not ind.suppressed]
    if not active:
        return RiskLevel.INFO
    return min(active, key=lambda i: _SEVERITY_ORDER[i.confidence]).confidence


def _deterministic_risk(ep: Endpoint, indicators: list[VulnIndicator]) -> RiskLevel:
    """Assign risk level based on vuln indicators and endpoint characteristics."""
    if indicators:
        return _highest_severity(indicators)

    cat = ep.category or EndpointCategory.UNKNOWN
    if cat in _LOW_RISK_CATEGORIES:
        return RiskLevel.INFO

    if ep.requires_auth is False:
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


def _format_ep_line(
    index: int, ep: Endpoint, indicators: list[VulnIndicator]
) -> str:
    """Format a single endpoint for the LLM batch context."""
    cat = ep.category.value if ep.category else "unknown"
    auth = (
        "requires_auth"
        if ep.requires_auth
        else "no_auth" if ep.requires_auth is False
        else "auth_unknown"
    )
    path = urlparse(ep.url).path

    parts = [
        f"  {index}. {ep.method} {ep.url} [{ep.status_code}] "
        f"category={cat} {auth}"
    ]

    if ep.parameters:
        parts.append(f"params=[{', '.join(ep.parameters)}]")
    if ep.response_fields:
        parts.append(f"response_fields=[{', '.join(ep.response_fields[:10])}]")
    if ep.discovered_by:
        parts.append(f"src={ep.discovered_by.value}")

    line = " | ".join(parts)

    active_indicators = [ind for ind in indicators if not ind.suppressed]
    if active_indicators:
        vuln_tags = ", ".join(
            f"{ind.pattern.value}({ind.confidence.value})"
            for ind in active_indicators
        )
        line += f" VULN=[{vuln_tags}]"

        # include chain info if present
        chain_ids = {ind.chain_id for ind in active_indicators if ind.chain_id}
        if chain_ids:
            line += f" CHAINS={len(chain_ids)}"

    if ep.response_body_snippet:
        snippet = ep.response_body_snippet[:_MAX_SNIPPET_IN_CONTEXT]
        line += f"\n      body_preview: {snippet}"

    return line


@register_agent
class PrioritizerAgent(BaseAgent):
    name = "prioritizer"
    description = (
        "Uses LLM to rank all endpoints by testing priority, producing an attack surface map "
        "with risk levels, rationale, and suggested security tests."
    )

    async def run(self, state: ScanState) -> AgentResult:
        findings: list[Finding] = []
        errors: list[str] = []

        all_endpoints = [
            (k, ep) for k, ep in state.endpoints.items()
            if not _is_metadata_path(ep.url)
        ]
        if not all_endpoints:
            return AgentResult(agent_name=self.name, success=True)

        tech_context = self.build_tech_context(state)
        findings_context = self.build_findings_context(state, limit=20)
        insights_context = self.build_insights_context(state)

        prioritized_keys: set[str] = set()

        for i in range(0, len(all_endpoints), PRIORITIZER_BATCH_SIZE):
            batch = all_endpoints[i : i + PRIORITIZER_BATCH_SIZE]
            batch_entries = await self._prioritize_batch(
                state=state,
                batch=batch,
                tech_context=tech_context,
                findings_context=findings_context,
                insights_context=insights_context,
            )

            if batch_entries is not None:
                for entry in batch_entries:
                    key = state.endpoint_key(entry.endpoint.method, entry.endpoint.url)
                    if key not in prioritized_keys:
                        state.attack_surface.append(entry)
                        prioritized_keys.add(key)
            else:
                errors.append(f"Prioritization batch {i // PRIORITIZER_BATCH_SIZE} failed")
                # deterministic fallback for this batch
                for key, ep in batch:
                    if key in prioritized_keys:
                        continue
                    entry = self._fallback_entry(state=state, key=key, ep=ep)
                    state.attack_surface.append(entry)
                    prioritized_keys.add(key)

        # catch any endpoints the LLM missed across all batches
        for key, ep in all_endpoints:
            if key in prioritized_keys:
                continue
            entry = self._fallback_entry(state=state, key=key, ep=ep)
            state.attack_surface.append(entry)

        # re-assign sequential ranks by risk severity
        state.attack_surface.sort(
            key=lambda e: _SEVERITY_ORDER.get(e.risk_level, 99)
        )
        for rank, entry in enumerate(state.attack_surface, start=1):
            entry.priority_rank = rank

        risk_counts: dict[str, int] = {}
        for entry in state.attack_surface:
            risk_counts[entry.risk_level.value] = (
                risk_counts.get(entry.risk_level.value, 0) + 1
            )

        findings.append(
            Finding(
                agent_name=self.name,
                finding_type="attack_surface_mapped",
                title=f"Attack surface: {len(state.attack_surface)} endpoints prioritized",
                detail=(
                    "Risk distribution: "
                    + ", ".join(f"{v} {k}" for k, v in risk_counts.items())
                ),
                severity=RiskLevel.INFO,
            )
        )

        return AgentResult(
            agent_name=self.name,
            success=len(errors) == 0,
            endpoints_found=[],
            findings=findings,
            errors=errors,
        )

    async def _prioritize_batch(
        self,
        state: ScanState,
        batch: list[tuple[str, Endpoint]],
        tech_context: str,
        findings_context: str,
        insights_context: str = "",
    ) -> list[AttackSurfaceEntry] | None:
        """Run LLM prioritization on a batch. Returns None on failure."""
        ep_lines = []
        for idx, (key, ep) in enumerate(batch, start=1):
            indicators = state.vuln_indicators.get(key, [])
            ep_lines.append(_format_ep_line(
                index=idx, ep=ep, indicators=indicators,
            ))

        messages = [
            {
                "role": "system",
                "content": self.prompt_registry.get("prioritizer").system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Target: {state.target}\n\n"
                    f"Prior Insights:\n{insights_context}\n\n"
                    f"Tech stack:\n{tech_context}\n\n"
                    f"Endpoints to prioritize ({len(batch)}):\n"
                    + "\n".join(ep_lines)
                    + f"\n\nFindings:\n{findings_context}"
                ),
            },
        ]

        try:
            response = await self.llm.chat_structured(
                messages, response_model=PrioritizationBatchResponse,
                name=f"prioritize_batch_{hash(batch[0][0]) % 1000}",
                confidence_threshold=settings.active_confidence_threshold,
            )
            return self._parse_llm_response(state=state, data=response, batch=batch)
        except Exception as e:
            logger.warning("Prioritization LLM call failed: %s", e)
            return None

    @staticmethod
    def _parse_llm_response(
        state: ScanState,
        data: PrioritizationBatchResponse,
        batch: list[tuple[str, Endpoint]],
    ) -> list[AttackSurfaceEntry]:
        """Parse LLM response into AttackSurfaceEntry objects using index-based matching."""
        entries: list[AttackSurfaceEntry] = []

        for entry_data in data.attack_surface:
            if entry_data.index is None:
                # fallback to URL-based matching
                ep = state.find_endpoint(entry_data.method, entry_data.url)
                if ep is None:
                    continue
                ep_key = state.endpoint_key(entry_data.method, entry_data.url)
            else:
                list_idx = entry_data.index - 1
                if list_idx < 0 or list_idx >= len(batch):
                    continue
                ep_key, ep = batch[list_idx]

            try:
                risk = RiskLevel(entry_data.risk_level)
            except ValueError:
                risk = RiskLevel.INFO

            try:
                cat = EndpointCategory(entry_data.category)
            except ValueError:
                cat = ep.category or EndpointCategory.UNKNOWN

            ep_vulns = state.vuln_indicators.get(ep_key, [])
            active_vulns = [ind for ind in ep_vulns if not ind.suppressed]

            entries.append(AttackSurfaceEntry(
                endpoint=ep,
                category=cat,
                risk_level=risk,
                priority_rank=0,
                rationale=entry_data.rationale,
                suggested_tests=entry_data.suggested_tests,
                vuln_indicators=active_vulns,
            ))

        return entries

    @staticmethod
    def _fallback_entry(
        state: ScanState, key: str, ep: Endpoint
    ) -> AttackSurfaceEntry:
        """Create a deterministic fallback entry when LLM is unavailable."""
        indicators = state.vuln_indicators.get(key, [])
        active = [ind for ind in indicators if not ind.suppressed]
        risk = _deterministic_risk(ep=ep, indicators=active)

        if active:
            vuln_summary = ", ".join(
                f"{ind.pattern.value} ({ind.confidence.value})"
                for ind in active
            )
            rationale = f"Rule-based risk assessment: {vuln_summary}"
        else:
            rationale = "No vulnerability indicators detected"

        return AttackSurfaceEntry(
            endpoint=ep,
            category=ep.category or EndpointCategory.UNKNOWN,
            risk_level=risk,
            priority_rank=0,
            rationale=rationale,
            vuln_indicators=active,
        )
