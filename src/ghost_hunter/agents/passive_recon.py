"""Passive reconnaissance — robots.txt, sitemap, headers, security.txt."""

from __future__ import annotations

import logging
import re
from xml.etree import ElementTree

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.models import (
    AgentResult,
    DiscoverySource,
    Endpoint,
    Finding,
    RiskLevel,
    ScanState,
    TechFingerprint,
)
from src.ghost_hunter.models.llm_responses import ReconAnalysisResponse

logger = logging.getLogger(__name__)

SECURITY_HEADERS = [
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "X-Frame-Options",
    "X-XSS-Protection",
    "Referrer-Policy",
    "Permissions-Policy",
]

WELL_KNOWN_PATHS = [
    "/.well-known/security.txt",
    "/.well-known/openid-configuration",
    "/favicon.ico",
    "/humans.txt",
]

# max blocked paths to include in LLM recon context
_MAX_BLOCKED_PATHS_IN_RECON = 20


@register_agent
class PassiveReconAgent(BaseAgent):
    name = "passive_recon"
    description = "Fetches robots.txt, sitemap.xml, headers, security.txt, and fingerprints the target."

    async def run(self, state: ScanState) -> AgentResult:
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []
        errors: list[str] = []

        # --- robots.txt ---
        eps, fds, errs = await self._parse_robots(state)
        endpoints.extend(eps)
        findings.extend(fds)
        errors.extend(errs)

        # --- sitemap.xml ---
        eps, errs = await self._parse_sitemap(state)
        endpoints.extend(eps)
        errors.extend(errs)

        # --- Header fingerprint ---
        fp, fds, errs = await self._fingerprint(state)
        state.tech_fingerprint = fp
        findings.extend(fds)
        errors.extend(errs)

        # --- Well-known paths ---
        eps, errs = await self._probe_well_known(state)
        endpoints.extend(eps)
        errors.extend(errs)

        # --- LLM analysis of collected recon data ---
        llm_findings = await self._llm_analyze_recon(state)
        findings.extend(llm_findings)

        return AgentResult(
            agent_name=self.name,
            success=True,
            endpoints_found=endpoints,
            findings=findings,
            errors=errors,
        )

    async def _parse_robots(
        self, state: ScanState
    ) -> tuple[list[Endpoint], list[Finding], list[str]]:
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []
        errors: list[str] = []

        resp = await self.http.get("/robots.txt")
        if resp is None or resp.status_code != 200:
            return endpoints, findings, errors

        endpoints.append(
            Endpoint(
                url=self.http.resolve_url("/robots.txt"),
                method="GET",
                status_code=200,
                discovered_by=DiscoverySource.ROBOTS_TXT,
            )
        )

        text = resp.text
        for line in text.splitlines():
            ep = self._parse_robots_line(line.strip(), state)
            if ep:
                endpoints.append(ep)

        if state.blocked_paths:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="robots_disallow",
                    title="Disallowed paths in robots.txt",
                    detail=f"Found {len(state.blocked_paths)} disallowed paths: {', '.join(state.blocked_paths[:10])}",
                    severity=RiskLevel.INFO,
                    evidence=text[:500],
                )
            )

        return endpoints, findings, errors

    def _parse_robots_line(self, line: str, state: ScanState) -> Endpoint | None:
        match = re.match(r"^(Disallow|Allow):\s*(.+)", line, re.IGNORECASE)
        if not match:
            return None
        directive, path = match.group(1), match.group(2).strip()
        if not path or path == "/" or path.startswith("#"):
            return None
        clean = re.sub(r"[\*\$\?].*", "", path)
        if not clean:
            return None
        if directive.lower() == "disallow":
            state.blocked_paths.append(clean)
        return Endpoint(
            url=self.http.resolve_url(clean),
            method="GET",
            discovered_by=DiscoverySource.ROBOTS_TXT,
            notes=f"robots.txt {directive}",
        )

    async def _parse_sitemap(
        self, state: ScanState
    ) -> tuple[list[Endpoint], list[str]]:
        endpoints: list[Endpoint] = []
        errors: list[str] = []

        resp = await self.http.get("/sitemap.xml")
        if resp is None or resp.status_code != 200:
            return endpoints, errors

        endpoints.append(
            Endpoint(
                url=self.http.resolve_url("/sitemap.xml"),
                method="GET",
                status_code=200,
                discovered_by=DiscoverySource.SITEMAP,
            )
        )

        try:
            root = ElementTree.fromstring(resp.text)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            for loc in root.findall(".//sm:loc", ns):
                if loc.text and self.http.is_same_origin(loc.text):
                    endpoints.append(
                        Endpoint(
                            url=loc.text.strip(),
                            method="GET",
                            discovered_by=DiscoverySource.SITEMAP,
                        )
                    )
            # also try without namespace
            for loc in root.iter("loc"):
                if loc.text and self.http.is_same_origin(loc.text):
                    url = loc.text.strip()
                    if not any(e.url == url for e in endpoints):
                        endpoints.append(
                            Endpoint(
                                url=url,
                                method="GET",
                                discovered_by=DiscoverySource.SITEMAP,
                            )
                        )
        except ElementTree.ParseError as e:
            errors.append(f"Failed to parse sitemap.xml: {e}")

        return endpoints, errors

    async def _fingerprint(
        self, state: ScanState
    ) -> tuple[TechFingerprint, list[Finding], list[str]]:
        fp = TechFingerprint()
        findings: list[Finding] = []
        errors: list[str] = []

        resp = await self.http.head("/")
        if resp is None:
            errors.append("Could not reach target for fingerprinting")
            return fp, findings, errors

        headers = resp.headers

        fp.server = headers.get("Server")
        if fp.server:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="server_header",
                    title=f"Server header exposed: {fp.server}",
                    detail="Server version disclosure can aid attackers in targeting known vulnerabilities.",
                    severity=RiskLevel.LOW,
                    evidence=f"Server: {fp.server}",
                )
            )

        x_powered = headers.get("X-Powered-By")
        if x_powered:
            fp.frameworks.append(x_powered)
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="framework_disclosure",
                    title=f"Framework disclosed: {x_powered}",
                    detail="X-Powered-By header reveals technology stack.",
                    severity=RiskLevel.LOW,
                    evidence=f"X-Powered-By: {x_powered}",
                )
            )

        for header_name in SECURITY_HEADERS:
            val = headers.get(header_name)
            if val:
                fp.security_headers[header_name] = val
            else:
                fp.missing_security_headers.append(header_name)

        if fp.missing_security_headers:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="missing_security_headers",
                    title=f"Missing {len(fp.missing_security_headers)} security headers",
                    detail=f"Missing: {', '.join(fp.missing_security_headers)}",
                    severity=RiskLevel.MEDIUM,
                )
            )

        for cookie_header in headers.get_list("set-cookie"):
            fp.cookies.append(cookie_header.split(";")[0])
            if "httponly" not in cookie_header.lower():
                findings.append(
                    Finding(
                        agent_name=self.name,
                        finding_type="cookie_no_httponly",
                        title="Cookie without HttpOnly flag",
                        detail="Cookies without HttpOnly are accessible via JavaScript.",
                        severity=RiskLevel.MEDIUM,
                        evidence=cookie_header[:200],
                    )
                )
            if "secure" not in cookie_header.lower():
                findings.append(
                    Finding(
                        agent_name=self.name,
                        finding_type="cookie_no_secure",
                        title="Cookie without Secure flag",
                        detail="Cookies without Secure can be sent over HTTP.",
                        severity=RiskLevel.LOW,
                        evidence=cookie_header[:200],
                    )
                )

        return fp, findings, errors

    async def _probe_well_known(
        self, state: ScanState
    ) -> tuple[list[Endpoint], list[str]]:
        endpoints: list[Endpoint] = []
        errors: list[str] = []

        for path in WELL_KNOWN_PATHS:
            resp = await self.http.get(path)
            if resp and resp.status_code == 200:
                endpoints.append(
                    Endpoint(
                        url=self.http.resolve_url(path),
                        method="GET",
                        status_code=200,
                        content_type=resp.headers.get("content-type", ""),
                        discovered_by=DiscoverySource.HEADER_PROBE,
                        notes=f"well-known path: {path}",
                    )
                )

        return endpoints, errors

    async def _llm_analyze_recon(self, state: ScanState) -> list[Finding]:
        """Run LLM analysis on headers, cookies, and robots.txt data."""
        findings: list[Finding] = []
        fp = state.tech_fingerprint

        context_parts: list[str] = []

        if fp.server:
            context_parts.append(f"Server: {fp.server}")
        if fp.frameworks:
            context_parts.append(f"Frameworks: {', '.join(fp.frameworks)}")
        if fp.security_headers:
            headers_str = ", ".join(f"{k}: {v}" for k, v in fp.security_headers.items())
            context_parts.append(f"Security headers present: {headers_str}")
        if fp.missing_security_headers:
            context_parts.append(f"Missing security headers: {', '.join(fp.missing_security_headers)}")
        if fp.cookies:
            context_parts.append(f"Cookies: {', '.join(fp.cookies)}")
        if state.blocked_paths:
            context_parts.append(f"Disallowed paths (robots.txt): {', '.join(state.blocked_paths[:_MAX_BLOCKED_PATHS_IN_RECON])}")

        well_known_notes = [
            f"{ep.url} [{ep.status_code}]"
            for ep in state.endpoints.values()
            if ep.discovered_by == DiscoverySource.HEADER_PROBE
        ]
        if well_known_notes:
            context_parts.append(f"Well-known path results: {', '.join(well_known_notes)}")

        if not context_parts:
            return findings

        messages = [
            {
                "role": "system",
                "content": self.prompt_registry.get("passive_recon").system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Target: {state.target}\n\n"
                    f"Reconnaissance data:\n" + "\n".join(context_parts)
                ),
            },
        ]

        try:
            response = await self.llm.chat_structured(
                messages, response_model=ReconAnalysisResponse,
                name="recon_analysis", max_tokens=1024,
            )

            # write tech hypotheses to state
            for hypothesis in response.tech_hypotheses:
                if hypothesis not in fp.technologies:
                    fp.technologies.append(hypothesis)

            if response.header_assessment:
                findings.append(Finding(
                    agent_name=self.name,
                    finding_type="recon_header_assessment",
                    title="LLM header security assessment",
                    detail=response.header_assessment,
                    severity=RiskLevel.INFO,
                ))

            if response.interesting_patterns:
                findings.append(Finding(
                    agent_name=self.name,
                    finding_type="recon_interesting_patterns",
                    title=f"Identified {len(response.interesting_patterns)} interesting pattern(s)",
                    detail="; ".join(response.interesting_patterns),
                    severity=RiskLevel.INFO,
                ))

            if response.initial_attack_vectors:
                findings.append(Finding(
                    agent_name=self.name,
                    finding_type="recon_attack_vectors",
                    title=f"Suggested {len(response.initial_attack_vectors)} initial attack vector(s)",
                    detail="; ".join(response.initial_attack_vectors),
                    severity=RiskLevel.INFO,
                ))

        except Exception as e:
            logger.warning(
                "LLM recon analysis failed (deterministic results intact): %s", e,
            )

        return findings
