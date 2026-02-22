"""Probe validator — sends targeted probes to confirm or refute suspected vulnerabilities."""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.models import (
    AgentResult,
    AttackSurfaceEntry,
    Endpoint,
    Finding,
    RiskLevel,
    ScanState,
    VulnIndicator,
    VulnPattern,
)

logger = logging.getLogger(__name__)

MAX_PROBES = 20
_MAX_BODY_SNIPPET = 2000

_SECURITY_RELEVANT_HEADERS = frozenset({
    "content-type", "server", "x-powered-by", "www-authenticate",
    "x-frame-options", "x-content-type-options", "access-control-allow-origin",
    "set-cookie", "location",
})

VALIDATION_SYSTEM_PROMPT = """\
CONTEXT:
You are analyzing an HTTP response to validate a suspected vulnerability in a \
web application security assessment. The probe was sent to confirm or refute a \
specific vulnerability pattern detected during earlier analysis.

ROLE:
Offensive security researcher who distinguishes true positives from false positives. \
You look for concrete evidence in response bodies, headers, and status codes — not \
theoretical possibilities.

ACTION:
Analyze the response body, headers, and status code. Determine if the vulnerability \
is confirmed, refuted, or inconclusive. Provide specific evidence from the response.

Key signals:
- Auth bypass: 200 response with user data when no auth was sent = CONFIRMED
- BOLA/IDOR: 200 with data for different IDs without ownership validation = CONFIRMED
- Info disclosure: response contains stack traces, internal IPs, credentials, debug info = CONFIRMED
- Mass assignment: extra fields reflected in response = CONFIRMED
- Excessive data: sensitive fields (passwords, tokens, SSNs) in response JSON = CONFIRMED
- Generic error pages, 401/403 responses, or empty bodies = REFUTED

FORMAT:
Respond with JSON:
{
  "validated": true | false | null,
  "confidence": "high" | "medium" | "low",
  "evidence": "specific evidence from the response that confirms or refutes the vulnerability",
  "severity_adjustment": "critical" | "high" | "medium" | "low" | null
}

TONE:
Be precise and conservative. Only mark validated=true when there is clear evidence. \
Mark validated=null (inconclusive) when the response is ambiguous.
"""

_MASS_ASSIGN_PROBE_FIELDS = {"role": "admin", "is_admin": True, "balance": 99999}

_NUMERIC_ID_IN_PATH = re.compile(r"/\d+(?=/|$)")
_TEMPLATE_PARAM_IN_PATH = re.compile(r"\{[^}]+\}")


class ProbeAnalysis(BaseModel):
    """Parsed LLM response for a single probe."""

    validated: bool | None = None
    confidence: str = "low"
    evidence: str = ""
    severity_adjustment: str | None = None


def _interpret_analysis(
    analysis: ProbeAnalysis, default_severity: RiskLevel = RiskLevel.HIGH
) -> tuple[bool | None, RiskLevel]:
    """Extract validated flag and severity from LLM analysis."""
    severity = default_severity if analysis.validated else RiskLevel.INFO
    if analysis.severity_adjustment:
        # LLM may return a severity string not in our enum — ignore gracefully
        try:
            severity = RiskLevel(analysis.severity_adjustment)
        except ValueError:
            pass

    return analysis.validated, severity


@register_agent
class ProbeValidatorAgent(BaseAgent):
    name = "probe_validator"
    description = (
        "Actively validates top-priority suspected vulnerabilities by sending "
        "safe probes and analyzing real responses with LLM assistance."
    )

    async def run(self, state: ScanState) -> AgentResult:
        candidates = self._select_candidates(state)
        if not candidates:
            return AgentResult(
                agent_name=self.name,
                success=True,
                findings=[
                    Finding(
                        agent_name=self.name,
                        finding_type="probe_validation_skip",
                        title="No candidates for active validation",
                        detail="No prioritized endpoints with vulnerability indicators to validate.",
                    )
                ],
            )

        findings: list[Finding] = []
        errors: list[str] = []

        for entry in candidates:
            entry_findings, entry_errors = await self._validate_entry(entry, state)
            findings.extend(entry_findings)
            errors.extend(entry_errors)

        validated_count = sum(1 for f in findings if f.validated is True)
        refuted_count = sum(1 for f in findings if f.validated is False)
        inconclusive_count = sum(1 for f in findings if f.validated is None)

        findings.insert(0, Finding(
            agent_name=self.name,
            finding_type="probe_validation_summary",
            title=(
                f"Active validation: {validated_count} confirmed, "
                f"{refuted_count} refuted, {inconclusive_count} inconclusive"
            ),
            detail=(
                f"Sent probes to {len(candidates)} high-priority endpoints. "
                f"Validated {validated_count} vulnerabilities with real evidence."
            ),
            severity=RiskLevel.INFO,
        ))

        return AgentResult(
            agent_name=self.name,
            success=True,
            findings=findings,
            errors=errors,
            metadata={
                "candidates": len(candidates),
                "validated": validated_count,
                "refuted": refuted_count,
                "inconclusive": inconclusive_count,
            },
        )

    def _select_candidates(self, state: ScanState) -> list[AttackSurfaceEntry]:
        """Select top-N attack surface entries that have vulnerability indicators."""
        candidates = [
            entry for entry in state.attack_surface
            if entry.vuln_indicators
            and entry.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM)
        ]
        return candidates[:MAX_PROBES]

    async def _validate_entry(
        self, entry: AttackSurfaceEntry, state: ScanState
    ) -> tuple[list[Finding], list[str]]:
        """Run all applicable probe strategies on one attack surface entry."""
        ep = entry.endpoint
        ep_key = state.endpoint_key(ep.method, ep.url)
        indicators = state.vuln_indicators.get(ep_key, [])
        strategies = self._plan_strategies(ep, indicators)

        findings: list[Finding] = []
        errors: list[str] = []

        for strategy in strategies:
            try:
                result = await self._run_strategy(strategy, ep)
                if result is not None:
                    findings.append(result)
            except Exception as e:
                errors.append(f"Probe failed for {ep.url}: {e}")
                logger.warning("Probe execution error on %s: %s", ep.url, e)

        return findings, errors

    @staticmethod
    def _plan_strategies(ep: Endpoint, indicators: list[VulnIndicator]) -> list[str]:
        """Return list of probe strategy names applicable to this endpoint."""
        strategies: list[str] = []
        patterns = {ind.pattern for ind in indicators if not ind.suppressed}

        if ep.requires_auth and VulnPattern.AUTH_BOUNDARY_GAP not in patterns:
            strategies.append("auth_bypass")
        if VulnPattern.AUTH_BOUNDARY_GAP in patterns:
            strategies.append("auth_bypass")
        if VulnPattern.BOLA_IDOR in patterns:
            strategies.append("bola_idor")
        if VulnPattern.INFO_DISCLOSURE in patterns:
            strategies.append("info_disclosure")
        if VulnPattern.MASS_ASSIGNMENT in patterns and ep.method in ("POST", "PUT", "PATCH"):
            strategies.append("mass_assignment")
        if VulnPattern.EXCESSIVE_DATA_EXPOSURE in patterns:
            strategies.append("excessive_data")

        if not strategies and patterns:
            strategies.append("generic_analysis")

        return strategies

    async def _run_strategy(self, strategy: str, ep: Endpoint) -> Finding | None:
        """Execute a single probe strategy and return a Finding or None."""
        path = urlparse(ep.url).path

        if strategy == "auth_bypass":
            return await self._probe_auth_bypass(ep, path)
        if strategy == "bola_idor":
            return await self._probe_bola_idor(ep, path)
        if strategy == "mass_assignment":
            return await self._probe_mass_assignment(ep, path)
        # info_disclosure, excessive_data, generic_analysis all use GET + LLM
        return await self._probe_get_and_analyze(ep, path, strategy)

    # ------------------------------------------------------------------
    # Strategy-specific probes (only for non-GET-based probes)
    # ------------------------------------------------------------------

    async def _probe_auth_bypass(self, ep: Endpoint, path: str) -> Finding | None:
        """Send request without auth to check for auth bypass."""
        resp = await self.http.get(ep.url)
        if resp is None:
            return None

        if resp.status_code in (401, 403):
            return Finding(
                agent_name=self.name,
                finding_type="probe_auth_bypass",
                title=f"Auth Bypass REFUTED — {ep.method} {path}",
                detail=f"Endpoint correctly returned {resp.status_code} without credentials",
                severity=RiskLevel.INFO,
                validated=False,
                validation_evidence=f"HTTP {resp.status_code} response without auth",
            )

        if resp.status_code != 200:
            return None

        analysis = await self._llm_analyze(
            strategy="auth_bypass",
            endpoint=f"{ep.method} {path}",
            resp=resp,
        )
        validated, severity = _interpret_analysis(analysis, RiskLevel.CRITICAL)

        return Finding(
            agent_name=self.name,
            finding_type="probe_auth_bypass",
            title=f"Auth Bypass {'CONFIRMED' if validated else 'tested'} — {ep.method} {path}",
            detail=analysis.evidence or "No auth required to access endpoint",
            severity=severity,
            evidence=f"Status {resp.status_code} without credentials. {analysis.evidence}",
            validated=validated,
            validation_evidence=analysis.evidence,
        )

    async def _probe_bola_idor(self, ep: Endpoint, path: str) -> Finding | None:
        """Send requests with different IDs to test BOLA/IDOR."""
        id_sub_path = _NUMERIC_ID_IN_PATH.sub("/9999", path)
        if id_sub_path == path:
            id_sub_path = _TEMPLATE_PARAM_IN_PATH.sub("9999", path)
        if id_sub_path == path:
            return None

        url_with_alt_id = ep.url.replace(path, id_sub_path)
        resp = await self.http.get(url_with_alt_id)
        if resp is None:
            return None

        analysis = await self._llm_analyze(
            strategy="bola_idor",
            endpoint=f"{ep.method} {path}",
            resp=resp,
            extra_context=f"Original path: {path}, probed with: {id_sub_path}",
        )
        validated, severity = _interpret_analysis(analysis, RiskLevel.CRITICAL)

        return Finding(
            agent_name=self.name,
            finding_type="probe_bola_idor",
            title=f"BOLA/IDOR {'CONFIRMED' if validated else 'tested'} — {ep.method} {path}",
            detail=analysis.evidence or f"Probed {id_sub_path}",
            severity=severity,
            evidence=f"Status {resp.status_code} for {id_sub_path}. {analysis.evidence}",
            validated=validated,
            validation_evidence=analysis.evidence,
        )

    async def _probe_mass_assignment(self, ep: Endpoint, path: str) -> Finding | None:
        """Send request with extra privilege fields to test mass assignment."""
        resp = await self.http.request(
            ep.method,
            ep.url,
            json=_MASS_ASSIGN_PROBE_FIELDS,
            headers={"Content-Type": "application/json"},
        )
        if resp is None:
            return None

        analysis = await self._llm_analyze(
            strategy="mass_assignment",
            endpoint=f"{ep.method} {path}",
            resp=resp,
            extra_context=f"Sent extra fields: {json.dumps(_MASS_ASSIGN_PROBE_FIELDS)}",
        )
        validated, severity = _interpret_analysis(analysis, RiskLevel.CRITICAL)

        return Finding(
            agent_name=self.name,
            finding_type="probe_mass_assignment",
            title=f"Mass Assignment {'CONFIRMED' if validated else 'tested'} — {ep.method} {path}",
            detail=analysis.evidence or "Tested with extra privilege fields",
            severity=severity,
            evidence=f"Status {resp.status_code}. {analysis.evidence}",
            validated=validated,
            validation_evidence=analysis.evidence,
        )

    # ------------------------------------------------------------------
    # Generic GET + LLM analysis (shared by info_disclosure, excessive_data, generic)
    # ------------------------------------------------------------------

    async def _probe_get_and_analyze(
        self, ep: Endpoint, path: str, strategy: str
    ) -> Finding | None:
        """GET endpoint and have LLM analyze the response."""
        resp = await self.http.get(ep.url)
        if resp is None or resp.status_code >= 400:
            return None

        analysis = await self._llm_analyze(
            strategy=strategy,
            endpoint=f"{ep.method} {path}",
            resp=resp,
        )
        validated, severity = _interpret_analysis(analysis, RiskLevel.HIGH)
        title_label = strategy.replace("_", " ").title()

        return Finding(
            agent_name=self.name,
            finding_type=f"probe_{strategy}",
            title=f"{title_label} {'CONFIRMED' if validated else 'tested'} — {ep.method} {path}",
            detail=analysis.evidence or f"Analyzed response for {strategy.replace('_', ' ')} indicators",
            severity=severity,
            evidence=analysis.evidence,
            validated=validated,
            validation_evidence=analysis.evidence,
        )

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    async def _llm_analyze(
        self,
        strategy: str,
        endpoint: str,
        resp: httpx.Response,
        extra_context: str = "",
    ) -> ProbeAnalysis:
        """Have LLM analyze a probe response to determine if vulnerability is confirmed."""
        relevant_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() in _SECURITY_RELEVANT_HEADERS
        }
        body_snippet = resp.text[:_MAX_BODY_SNIPPET] if resp.text else ""

        user_content = (
            f"PROBE STRATEGY: {strategy}\n"
            f"ENDPOINT: {endpoint}\n"
            f"STATUS CODE: {resp.status_code}\n"
            f"RESPONSE HEADERS: {json.dumps(relevant_headers)}\n"
            f"RESPONSE BODY (truncated):\n{body_snippet}\n"
        )
        if extra_context:
            user_content += f"\nADDITIONAL CONTEXT: {extra_context}"

        messages = [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = await self.llm.chat_json(messages, name=f"probe_{strategy}")
            return ProbeAnalysis.model_validate(raw)
        except Exception as e:
            logger.warning("LLM analysis failed for %s probe on %s: %s", strategy, endpoint, e)
            return ProbeAnalysis(evidence=f"LLM analysis failed: {e}")
