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

logger = logging.getLogger(__name__)


@register_agent
class HypothesisAgent(BaseAgent):
    name = "hypothesis"
    description = (
        "Uses LLM to hypothesize undiscovered endpoints based on the full scan context, "
        "domain signals, and REST conventions, then validates each guess with a HEAD request."
    )

    async def run(self, state: ScanState) -> AgentResult:
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []
        errors: list[str] = []

        known_endpoints = sorted(state.endpoints.keys())[:60]
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

        messages = [
            {
                "role": "system",
                "content": (
                    "CONTEXT:\n"
                    "You are part of a multi-agent security scanner. Prior agents have already discovered "
                    "endpoints via crawling, OpenAPI spec parsing, common path probing, and JavaScript "
                    "analysis. Your job is to hypothesize endpoints those methods missed. Each hypothesis "
                    "costs one HTTP request to validate, so quality matters more than quantity.\n\n"
                    "ROLE:\n"
                    "You are a senior penetration tester with deep expertise in web application security "
                    "and API reverse-engineering. You excel at inferring hidden endpoints from naming "
                    "patterns, domain context, and tech stack conventions.\n\n"
                    "ACTION:\n"
                    "Follow this 5-step reasoning process:\n"
                    "1. NAMING CONVENTIONS: Analyze the discovered endpoint naming scheme (snake_case, "
                    "camelCase, kebab-case, plural/singular) and generate hypotheses that match\n"
                    "2. DOMAIN INFERENCE: Infer the application domain from endpoint names and content. "
                    "A banking app likely has /transfer, /statement, /beneficiary endpoints\n"
                    "3. BLOCKED PATH EXPLOITATION: 403 responses on blocked paths hint at auth-walled "
                    "content. Try sub-paths, alternative HTTP methods, or version variants\n"
                    "4. SECURITY-SENSITIVE PATHS: Target admin panels, debug endpoints, backup files, "
                    "internal APIs, password reset flows, and OAuth/SAML endpoints\n"
                    "5. TECH-STACK SPECIFICS: Use known tech stack to guess framework-specific paths "
                    "(e.g., Django: /admin/, Flask: /static/, Spring: /actuator/)\n\n"
                    "Endpoint keys use the format 'METHOD URL' (e.g., 'GET https://example.com/api/users').\n\n"
                    "FORMAT:\n"
                    "Respond with JSON:\n"
                    "{\n"
                    '  "reasoning": "2-3 sentences about domain/pattern observations",\n'
                    '  "hypotheses": [\n'
                    '    {"path": "/...", "method": "GET", "confidence": "high|medium|low", '
                    '"reasoning": "why this endpoint likely exists"}\n'
                    "  ]\n"
                    "}\n\n"
                    "FEW-SHOT EXAMPLES:\n"
                    "Given endpoints: GET /api/v1/accounts, GET /api/v1/accounts/{id}, "
                    "POST /api/v1/transfer\n"
                    "Good hypotheses:\n"
                    '  {"path": "/api/v1/accounts/{id}/transactions", "method": "GET", '
                    '"confidence": "high", "reasoning": "Banking app with accounts and transfers — '
                    'transaction history per account is standard"}\n'
                    '  {"path": "/api/v1/accounts/{id}/beneficiaries", "method": "GET", '
                    '"confidence": "medium", "reasoning": "Transfer endpoint implies beneficiary '
                    'management for recurring transfers"}\n'
                    '  {"path": "/api/v2/accounts", "method": "GET", "confidence": "medium", '
                    '"reasoning": "v1 exists, v2 may have different auth requirements"}\n\n'
                    "Generate 15-25 hypotheses, prioritizing high-confidence ones."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target: {state.target}\n\n"
                    f"Tech Stack:\n{tech_info}\n\n"
                    f"Discovered Endpoints ({len(known_endpoints)}):\n"
                    + "\n".join(f"  {ep}" for ep in known_endpoints)
                    + f"\n\nFindings so far:\n{findings_context}"
                ),
            },
        ]

        try:
            data = await self.llm.chat_json(messages, name="hypothesis_generation")
            hypotheses = data.get("hypotheses", [])

            validated = 0
            total = 0

            for hyp in hypotheses[:25]:
                path = hyp.get("path", "")
                method = hyp.get("method", "GET").upper()
                if not path:
                    continue

                key = state.endpoint_key(method, self.http.resolve_url(path))
                if key in state.endpoints:
                    continue

                total += 1

                probe_method = "HEAD" if method != "GET" else "GET"
                resp = await self.http.request(probe_method, path)

                if resp and resp.status_code < 404:
                    endpoints.append(
                        Endpoint(
                            url=self.http.resolve_url(path),
                            method=method,
                            status_code=resp.status_code,
                            content_type=resp.headers.get("content-type", ""),
                            discovered_by=DiscoverySource.LLM_HYPOTHESIS,
                            notes=(
                                f"Hypothesis ({hyp.get('confidence', '?')}): "
                                f"{hyp.get('reasoning', '')}"
                            ),
                        )
                    )
                    validated += 1

            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="hypothesis_results",
                    title=f"Validated {validated}/{total} hypothesized endpoints",
                    detail=(
                        f"LLM generated {len(hypotheses)} hypotheses. "
                        f"After filtering known endpoints, {total} were probed. "
                        f"{validated} returned non-404 responses."
                    ),
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

