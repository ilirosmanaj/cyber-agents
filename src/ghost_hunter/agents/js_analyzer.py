"""JavaScript bundle analyzer — extracts API routes from JS files via regex."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

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

# Patterns that match API endpoint references in JavaScript
JS_API_PATTERNS = [
    # fetch / axios / XMLHttpRequest style calls
    re.compile(r"""(?:fetch|axios\.(?:get|post|put|delete|patch)|\.open)\s*\(\s*['"`]([^'"`\s]+)['"`]""", re.IGNORECASE),
    # url assignment patterns
    re.compile(r"""(?:url|endpoint|api_?url|base_?url|href|path|route)\s*[:=]\s*['"`]([^'"`\s]{4,})['"`]""", re.IGNORECASE),
    # string literals that look like API paths
    re.compile(r"""['"`](/api/[^'"`\s]+)['"`]"""),
    re.compile(r"""['"`](/v[0-9]+/[^'"`\s]+)['"`]"""),
    # template literals
    re.compile(r"""`(/[^`\s]*\$\{[^}]+\}[^`\s]*)`"""),
]

MAX_JS_FILES = 50


@register_agent
class JSAnalyzerAgent(BaseAgent):
    name = "js_analyzer"
    description = "Downloads JS files and extracts API route references via regex."

    _STATIC_EXTENSIONS = {".js", ".css", ".png", ".jpg", ".svg", ".gif", ".woff"}

    async def run(self, state: ScanState) -> AgentResult:
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []
        errors: list[str] = []

        js_urls = list(set(state.js_urls))
        if not js_urls:
            return AgentResult(
                agent_name=self.name,
                success=True,
                findings=[
                    Finding(
                        agent_name=self.name,
                        finding_type="no_js_files",
                        title="No JavaScript files to analyze",
                        detail="Web crawler did not discover any JS files.",
                        severity=RiskLevel.INFO,
                    )
                ],
            )

        extracted_paths: set[str] = set()

        for js_url in js_urls[:MAX_JS_FILES]:
            resp = await self.http.get(js_url)
            if resp is None or resp.status_code != 200:
                continue
            new_eps = self._extract_paths_from_js(resp.text, js_url, extracted_paths)
            endpoints.extend(new_eps)

        if extracted_paths:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="js_api_routes",
                    title=f"Found {len(extracted_paths)} API paths in JavaScript",
                    detail=f"Analyzed {len(js_urls)} JS files. Unique paths: {', '.join(list(extracted_paths)[:10])}",
                    severity=RiskLevel.INFO,
                )
            )

        return AgentResult(
            agent_name=self.name,
            success=True,
            endpoints_found=endpoints,
            findings=findings,
            errors=errors,
            metadata={"js_files_analyzed": len(js_urls), "paths_extracted": len(extracted_paths)},
        )

    def _extract_paths_from_js(
        self, text: str, js_url: str, seen: set[str]
    ) -> list[Endpoint]:
        endpoints: list[Endpoint] = []
        for pattern in JS_API_PATTERNS:
            for match in pattern.finditer(text):
                full_url = self._resolve_js_path(match.group(1))
                if full_url is None or full_url in seen:
                    continue
                seen.add(full_url)
                endpoints.append(
                    Endpoint(
                        url=full_url,
                        method="GET",
                        discovered_by=DiscoverySource.JS_ANALYSIS,
                        notes=f"extracted from {js_url}",
                    )
                )
        return endpoints

    MIN_PATH_LENGTH = 4
    MAX_PATH_LENGTH = 200

    def _resolve_js_path(self, path: str) -> str | None:
        if len(path) < self.MIN_PATH_LENGTH or len(path) > self.MAX_PATH_LENGTH:
            return None
        if any(ext in path for ext in self._STATIC_EXTENSIONS):
            return None
        if path.startswith("/"):
            return self.http.resolve_url(path)
        if path.startswith(("http://", "https://")):
            parsed = urlparse(path)
            if not parsed.netloc or len(parsed.netloc) < 4:
                return None
            if self.http.is_same_origin(path):
                return path
        return None
