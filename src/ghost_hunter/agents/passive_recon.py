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
