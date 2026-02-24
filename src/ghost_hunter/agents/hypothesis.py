"""Hypothesis agent — LLM generates and validates endpoint guesses using domain reasoning."""

from __future__ import annotations

import logging

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.models import (
    AgentResult,
    DiscoverySource,
    Endpoint,
    Finding,
    RiskLevel,
    ScanState,
)
from src.ghost_hunter.models.llm_responses import HypothesisResponse

logger = logging.getLogger(__name__)

MAX_ENDPOINT_CONTEXT = 60
MAX_HYPOTHESES = 25

_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


@register_agent
class HypothesisAgent(BaseAgent):
    name = "hypothesis"
    description = (
        "Uses LLM to hypothesize undiscovered endpoints based on the full scan context, "
        "domain signals, and REST conventions, then validates each guess."
    )

    async def run(self, state: ScanState) -> AgentResult:
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []
        errors: list[str] = []

        known_endpoints = sorted(state.endpoints.keys())[:MAX_ENDPOINT_CONTEXT]
        if not known_endpoints:
            return AgentResult(
                agent_name=self.name,
                success=True,
                findings=[
                    Finding(
                        agent_name=self.name,
                        finding_type="no_endpoints",
                        title="No endpoints to reason about",
                        detail="Hypothesis agent requires previously discovered endpoints.",
                        severity=RiskLevel.INFO,
                    )
                ],
            )

        tech_info = self.build_tech_context(state)
        findings_context = self.build_findings_context(state)
        insights_context = self.build_insights_context(state)
        strategy_context = self._build_strategy_context(state)
        endpoint_context = self._build_endpoint_context(state, known_endpoints)
        blocked_paths_context = self._build_blocked_paths_context(state)

        messages = [
            {
                "role": "system",
                "content": self.prompt_registry.get("hypothesis").system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Target: {state.target}\n\n"
                    f"Prior Insights:\n{insights_context}\n\n"
                    f"{strategy_context}"
                    f"Tech Stack:\n{tech_info}\n\n"
                    f"Discovered Endpoints ({len(known_endpoints)}):\n{endpoint_context}\n\n"
                    f"{blocked_paths_context}"
                    f"Findings so far:\n{findings_context}"
                ),
            },
        ]

        try:
            response = await self.llm.chat_structured(
                messages, response_model=HypothesisResponse,
                name="hypothesis_generation",
            )

            # sort by confidence so high-confidence get validated first
            sorted_hypotheses = sorted(
                response.hypotheses,
                key=lambda h: _CONFIDENCE_ORDER.get(h.confidence, 2),
            )

            validated = 0
            auth_protected = 0
            total = 0
            seen: set[str] = set()

            for hyp in sorted_hypotheses[:MAX_HYPOTHESES]:
                path = hyp.path
                method = hyp.method.upper()
                if not path:
                    continue

                # ensure leading slash for relative paths
                if not path.startswith(("/", "http://", "https://")):
                    path = "/" + path

                resolved = self.http.resolve_url(path)

                # skip off-origin and already-known endpoints
                if not self.http.is_same_origin(resolved):
                    continue

                key = state.endpoint_key(method, resolved)
                if key in state.endpoints or key in seen:
                    continue
                seen.add(key)

                total += 1

                # use actual method for POST/PUT/PATCH with minimal body
                if method in ("POST", "PUT", "PATCH"):
                    resp = await self.http.request(
                        method, path,
                        headers={"Content-Type": "application/json"},
                        content="{}",
                    )
                else:
                    resp = await self.http.request("GET" if method == "GET" else "HEAD", path)

                if resp is None or resp.status_code == 404:
                    continue

                if resp.status_code in (401, 403):
                    auth_protected += 1
                    endpoints.append(
                        Endpoint(
                            url=resolved,
                            method=method,
                            status_code=resp.status_code,
                            discovered_by=DiscoverySource.LLM_HYPOTHESIS,
                            requires_auth=True,
                            notes=(
                                f"Hypothesis ({hyp.confidence}): "
                                f"{hyp.reasoning}"
                            ),
                        )
                    )
                    validated += 1
                    continue

                endpoints.append(
                    Endpoint(
                        url=resolved,
                        method=method,
                        status_code=resp.status_code,
                        content_type=resp.headers.get("content-type", ""),
                        discovered_by=DiscoverySource.LLM_HYPOTHESIS,
                        response_body_snippet=self.extract_body_snippet(resp),
                        notes=(
                            f"Hypothesis ({hyp.confidence}): "
                            f"{hyp.reasoning}"
                        ),
                    )
                )
                validated += 1

            logger.info(
                "Hypotheses: %d generated, %d probed, %d validated",
                len(response.hypotheses), total, validated,
            )

            detail = (
                f"LLM generated {len(response.hypotheses)} hypotheses. "
                f"After filtering known endpoints, {total} were probed. "
                f"{validated} returned non-404 responses."
            )
            if auth_protected:
                detail += f" {auth_protected} require authentication."

            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="hypothesis_results",
                    title=f"Validated {validated}/{total} hypothesized endpoints",
                    detail=detail,
                    severity=RiskLevel.INFO if validated == 0 else RiskLevel.MEDIUM,
                )
            )

        except Exception as e:
            errors.append(f"Hypothesis generation failed: {e}")

        return AgentResult(
            agent_name=self.name,
            success=len(errors) == 0,
            endpoints_found=endpoints,
            findings=findings,
            errors=errors,
        )

    @staticmethod
    def _build_endpoint_context(
        state: ScanState, keys: list[str]
    ) -> str:
        """Format endpoint keys with metadata for richer LLM context."""
        lines: list[str] = []
        for key in keys:
            ep = state.endpoints.get(key)
            if ep is None:
                lines.append(f"  {key}")
                continue

            parts = [f"  {key}"]
            if ep.parameters:
                parts.append(f"params={','.join(ep.parameters[:5])}")
            if ep.content_type:
                parts.append(f"type={ep.content_type.split(';')[0]}")
            if ep.status_code:
                parts.append(f"status={ep.status_code}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _build_strategy_context(state: ScanState) -> str:
        """Format planner strategy for LLM context."""
        if state.scan_strategy is None:
            return ""
        s = state.scan_strategy
        parts = []
        if s.focus_areas:
            parts.append(f"Focus areas: {', '.join(s.focus_areas)}")
        if s.tech_hypotheses:
            parts.append(f"Tech hypotheses: {', '.join(s.tech_hypotheses)}")
        if s.priority_patterns:
            parts.append(f"Priority patterns: {', '.join(s.priority_patterns)}")
        if not parts:
            return ""
        return "Scan Strategy:\n" + "\n".join(f"  {p}" for p in parts) + "\n\n"

    @staticmethod
    def _build_blocked_paths_context(state: ScanState) -> str:
        """Format blocked paths for structured LLM context."""
        if not state.blocked_paths:
            return ""
        paths = ", ".join(state.blocked_paths[:15])
        return f"Blocked paths (403): {paths}\n\n"
