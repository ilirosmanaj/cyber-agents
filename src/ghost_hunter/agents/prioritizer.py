"""Prioritizer agent — LLM ranks endpoints by attack priority with rationale."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
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

        all_endpoints = list(state.endpoints.items())
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
                "content": (
                    "CONTEXT:\n"
                    "You are producing the final attack surface map — the primary deliverable of this "
                    "security scan. All prior analysis (crawling, OpenAPI parsing, JS analysis, endpoint "
                    "classification, and vulnerability pattern detection) feeds into your prioritization. "
                    "Endpoints tagged with VULN= have been flagged by both deterministic pattern matching "
                    "and LLM-enhanced analysis. Chained vulnerabilities are especially critical.\n\n"
                    "ROLE:\n"
                    "You are a principal application security engineer and penetration test lead. You "
                    "understand real-world exploitation, attack chaining, and business impact assessment.\n\n"
                    "ACTION:\n"
                    "Follow these steps for each endpoint:\n"
                    "1. Cross-reference VULN tags with the tech stack — a race_condition on a Flask app "
                    "with SQLite is more exploitable than on a Go service with PostgreSQL\n"
                    "2. Evaluate chained vulnerabilities FIRST — auth_boundary_gap + bola_idor is far "
                    "more critical than either alone\n"
                    "3. Consider business context — financial endpoints (/transfer, /payment) warrant "
                    "higher priority than informational ones\n"
                    "4. Write executable suggested_tests with exact curl commands, payloads, and expected "
                    "responses\n\n"
                    "RISK LEVEL DEFINITIONS:\n"
                    "- critical (CVSS 9.0-10.0): Unauthenticated RCE, unauthenticated IDOR on financial "
                    "data, auth bypass on admin, chained vulns enabling account takeover. "
                    "Example: POST /transfer/{account_number} with no auth + IDOR\n"
                    "- high (CVSS 7.0-8.9): Authenticated IDOR, SSRF to internal services, mass assignment "
                    "on privilege fields, unrestricted file upload. "
                    "Example: PUT /profile with is_admin field accepted\n"
                    "- medium (CVSS 4.0-6.9): Reflected XSS, information disclosure of non-critical data, "
                    "race conditions on non-financial endpoints, JWT with weak config. "
                    "Example: GET /debug returns stack traces\n"
                    "- low (CVSS 0.1-3.9): Verbose error messages, missing non-critical security headers, "
                    "version disclosure. Example: Server header reveals exact version\n"
                    "- info (CVSS 0.0): Informational findings, best practice recommendations, no direct "
                    "security impact. Example: API documentation publicly accessible\n\n"
                    "VULN PATTERNS:\n"
                    "bola_idor=IDOR/BOLA, mass_assignment=mass assignment, ssrf=SSRF, "
                    "file_upload=unrestricted upload, jwt_weakness=auth weakness, "
                    "race_condition=concurrency issue, prompt_injection=AI manipulation, "
                    "info_disclosure=sensitive exposure, auth_boundary_gap=missing auth on data endpoint, "
                    "excessive_data_exposure=sensitive response fields, "
                    "api_version_confusion=inconsistent auth across versions, "
                    "broken_function_level_auth=admin without auth, "
                    "chained_vulnerability=compound attack chain\n\n"
                    "FORMAT:\n"
                    "Respond with JSON:\n"
                    "{\n"
                    '  "reasoning": "3-5 sentence overall attack surface analysis",\n'
                    '  "attack_surface": [\n'
                    "    {\n"
                    '      "index": 1,\n'
                    '      "category": "rest_api|auth_endpoint|admin_endpoint|...",\n'
                    '      "risk_level": "critical|high|medium|low|info",\n'
                    '      "rationale": "Why this is high priority...",\n'
                    '      "suggested_tests": [\n'
                    '        "curl -X POST https://target/transfer/12345 '
                    "-H 'Content-Type: application/json' "
                    "-d '{\\\"amount\\\": 1000}' — test unauthenticated fund transfer\"\n"
                    "      ]\n"
                    "    }\n"
                    "  ]\n"
                    "}\n\n"
                    "IMPORTANT: Use the index number to identify each endpoint. "
                    "Include ALL endpoints from the batch."
                ),
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
            data = await self.llm.chat_json(
                messages, name=f"prioritize_batch_{hash(batch[0][0]) % 1000}"
            )
            return self._parse_llm_response(state=state, data=data, batch=batch)
        except Exception as e:
            logger.warning("Prioritization LLM call failed: %s", e)
            return None

    @staticmethod
    def _parse_llm_response(
        state: ScanState,
        data: dict,
        batch: list[tuple[str, Endpoint]],
    ) -> list[AttackSurfaceEntry]:
        """Parse LLM response into AttackSurfaceEntry objects using index-based matching."""
        entries: list[AttackSurfaceEntry] = []

        for entry_data in data.get("attack_surface", []):
            batch_idx = entry_data.get("index")
            if batch_idx is None:
                # fallback to URL-based matching
                url = entry_data.get("url", "")
                method = entry_data.get("method", "GET")
                ep = state.find_endpoint(method, url)
                if ep is None:
                    continue
                ep_key = state.endpoint_key(method, url)
            else:
                list_idx = batch_idx - 1
                if list_idx < 0 or list_idx >= len(batch):
                    continue
                ep_key, ep = batch[list_idx]

            try:
                risk = RiskLevel(entry_data.get("risk_level", "info"))
            except ValueError:
                risk = RiskLevel.INFO

            try:
                cat = EndpointCategory(entry_data.get("category", "unknown"))
            except ValueError:
                cat = ep.category or EndpointCategory.UNKNOWN

            ep_vulns = state.vuln_indicators.get(ep_key, [])
            active_vulns = [ind for ind in ep_vulns if not ind.suppressed]

            entries.append(AttackSurfaceEntry(
                endpoint=ep,
                category=cat,
                risk_level=risk,
                priority_rank=0,
                rationale=entry_data.get("rationale", ""),
                suggested_tests=entry_data.get("suggested_tests", []),
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
