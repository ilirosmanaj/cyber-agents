"""Prioritizer agent — LLM ranks endpoints by attack priority with rationale."""

from __future__ import annotations

import logging

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.models import (
    AgentResult,
    AttackSurfaceEntry,
    EndpointCategory,
    Finding,
    RiskLevel,
    ScanState,
)

logger = logging.getLogger(__name__)


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

        all_endpoints = list(state.endpoints.values())
        if not all_endpoints:
            return AgentResult(agent_name=self.name, success=True)

        ep_lines = []
        for ep in all_endpoints:
            cat = ep.category.value if ep.category else "unknown"
            auth = (
                "requires_auth"
                if ep.requires_auth
                else "no_auth" if ep.requires_auth is False
                else "auth_unknown"
            )
            params = f" params=[{', '.join(ep.parameters)}]" if ep.parameters else ""
            line = (
                f"  {ep.method} {ep.url} [{ep.status_code}] "
                f"category={cat} {auth}{params}"
            )
            # append vuln indicators if present
            key = state.endpoint_key(ep.method, ep.url)
            indicators = state.vuln_indicators.get(key, [])
            if indicators:
                vuln_tags = ", ".join(
                    f"{ind.pattern.value}({ind.confidence.value})"
                    for ind in indicators
                )
                line += f" VULN=[{vuln_tags}]"
            ep_lines.append(line)

        tech_context = self.build_tech_context(state)
        findings_context = self.build_findings_context(state, limit=20)

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
                    '      "url": "...",\n'
                    '      "method": "...",\n'
                    '      "category": "rest_api|auth_endpoint|admin_endpoint|...",\n'
                    '      "risk_level": "critical|high|medium|low|info",\n'
                    '      "priority_rank": 1,\n'
                    '      "rationale": "Why this is high priority...",\n'
                    '      "suggested_tests": [\n'
                    '        "curl -X POST https://target/transfer/12345 -H \'Content-Type: application/json\' '
                    "-d '{\\\"amount\\\": 1000, \\\"to_account\\\": \\\"attacker\\\"}' — test unauthenticated fund transfer\",\n"
                    '        "curl -X PUT https://target/profile -H \'Authorization: Bearer TOKEN\' '
                    "-d '{\\\"is_admin\\\": true}' — test mass assignment privilege escalation\"\n"
                    "      ]\n"
                    "    }\n"
                    "  ]\n"
                    "}\n\n"
                    "Include ALL endpoints, ranked from highest to lowest priority."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target: {state.target}\n\n"
                    f"Tech stack:\n{tech_context}\n\n"
                    f"Endpoints ({len(all_endpoints)}):\n"
                    + "\n".join(ep_lines)
                    + f"\n\nFindings:\n{findings_context}"
                ),
            },
        ]

        try:
            data = await self.llm.chat_json(messages, name="prioritize_attack_surface")
            surface = data.get("attack_surface", [])

            for i, entry in enumerate(surface):
                url = entry.get("url", "")
                method = entry.get("method", "GET")

                ep = state.find_endpoint(method, url)
                if ep is None:
                    continue

                try:
                    risk = RiskLevel(entry.get("risk_level", "info"))
                except ValueError:
                    risk = RiskLevel.INFO

                try:
                    cat = EndpointCategory(entry.get("category", "unknown"))
                except ValueError:
                    cat = ep.category or EndpointCategory.UNKNOWN

                ep_key = state.endpoint_key(method, url)
                ep_vulns = state.vuln_indicators.get(ep_key, [])

                state.attack_surface.append(
                    AttackSurfaceEntry(
                        endpoint=ep,
                        category=cat,
                        risk_level=risk,
                        priority_rank=entry.get("priority_rank", i + 1),
                        rationale=entry.get("rationale", ""),
                        suggested_tests=entry.get("suggested_tests", []),
                        vuln_indicators=ep_vulns,
                    )
                )

            state.attack_surface.sort(key=lambda x: x.priority_rank)

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

        except Exception as e:
            errors.append(f"Prioritization failed: {e}")

        return AgentResult(
            agent_name=self.name,
            success=len(errors) == 0,
            endpoints_found=[],
            findings=findings,
            errors=errors,
        )
