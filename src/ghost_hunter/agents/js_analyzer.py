"""JavaScript bundle analyzer — extracts API routes from JS files via regex + LLM."""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlparse

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.config import settings
from src.ghost_hunter.models import (
    AgentResult,
    DiscoverySource,
    Endpoint,
    Finding,
    RiskLevel,
    ScanState,
)
from src.ghost_hunter.models.llm_responses import JSAnalysisBatchResponse

logger = logging.getLogger(__name__)

# fetch / axios / XMLHttpRequest with method extraction
_FETCH_PATTERN = re.compile(
    r"""fetch\s*\(\s*['"`]([^'"`\s]+)['"`]""", re.IGNORECASE
)
_AXIOS_METHOD_PATTERN = re.compile(
    r"""axios\.(get|post|put|delete|patch)\s*\(\s*['"`]([^'"`\s]+)['"`]""",
    re.IGNORECASE,
)
_XHR_OPEN_PATTERN = re.compile(
    r"""\.open\s*\(\s*['"`](GET|POST|PUT|DELETE|PATCH)['"`]\s*,\s*['"`]([^'"`\s]+)['"`]""",
    re.IGNORECASE,
)
_JQUERY_AJAX_URL_PATTERN = re.compile(
    r"""\$\.(?:ajax|get|post|getJSON)\s*\(\s*['"`]([^'"`\s]+)['"`]""",
    re.IGNORECASE,
)
_JQUERY_AJAX_METHOD_PATTERN = re.compile(
    r"""\$\.ajax\s*\(\s*\{[^}]*?['"]?url['"]?\s*:\s*['"`]([^'"`\s]+)['"`]""",
    re.IGNORECASE,
)

# url / endpoint assignment patterns
_URL_ASSIGN_PATTERN = re.compile(
    r"""(?:url|endpoint|api_?url|base_?url|href|path|route)\s*[:=]\s*['"`]([^'"`\s]{4,})['"`]""",
    re.IGNORECASE,
)

# string literals that look like API paths
_API_PATH_PATTERN = re.compile(r"""['"`](/api/[^'"`\s]+)['"`]""")
_VERSIONED_PATH_PATTERN = re.compile(r"""['"`](/v[0-9]+/[^'"`\s]+)['"`]""")

# template literals — matches paths with or without interpolation
_TEMPLATE_LITERAL_PATTERN = re.compile(r"""`(/[^`\s]{3,})`""")

# WebSocket connections
_WEBSOCKET_PATTERN = re.compile(
    r"""(?:new\s+WebSocket|io)\s*\(\s*['"`](wss?://[^'"`\s]+|/[^'"`\s]+)['"`]""",
    re.IGNORECASE,
)

# GraphQL endpoints
_GRAPHQL_PATTERN = re.compile(
    r"""(?:graphql|gql)\s*\(\s*['"`]([^'"`\s]+)['"`]""", re.IGNORECASE
)

# simple patterns — (pattern, method_or_None)
# for patterns that don't capture method, method defaults to "GET"
JS_API_PATTERNS: list[re.Pattern] = [
    _FETCH_PATTERN,
    _JQUERY_AJAX_URL_PATTERN,
    _JQUERY_AJAX_METHOD_PATTERN,
    _URL_ASSIGN_PATTERN,
    _API_PATH_PATTERN,
    _VERSIONED_PATH_PATTERN,
    _TEMPLATE_LITERAL_PATTERN,
    _WEBSOCKET_PATTERN,
    _GRAPHQL_PATTERN,
]

_STATIC_EXTENSIONS = {
    ".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".gif",
    ".woff", ".woff2", ".ttf", ".eot", ".ico", ".webp", ".avif",
    ".map", ".mp4", ".mp3",
}

_JS_CONTENT_TYPES = {"application/javascript", "text/javascript", "application/x-javascript"}

# don't try to parse JS files larger than this
_MAX_JS_FILE_SIZE = 5 * 1024 * 1024

# parallel fetch concurrency
_JS_FETCH_CONCURRENCY = 5

_METHOD_MAP = {
    "get": "GET", "post": "POST", "put": "PUT",
    "delete": "DELETE", "patch": "PATCH",
}

# snippet extraction patterns for LLM analysis
_SNIPPET_TRIGGERS = re.compile(
    r"(?:fetch\s*\(|axios\.|XMLHttpRequest|\.open\s*\(|\$\.ajax|"
    r"localStorage\.getItem|sessionStorage\.getItem|"
    r"Authorization|Bearer|"
    r"React(?:Router|\.lazy)|createBrowserRouter|Route\s*path|"
    r"Vue\.use\(Router|routes\s*:\s*\[|"
    r"app\.(?:get|post|put|delete|patch|use)\s*\()",
    re.IGNORECASE,
)
_SNIPPET_CONTEXT_CHARS = 400  # chars around each trigger match
_MAX_SNIPPETS_PER_FILE = 10
_MAX_TOTAL_SNIPPETS = 30
_LLM_SNIPPET_BATCH_SIZE = 8
_MAX_LLM_BATCHES = 3

_JS_LLM_SYSTEM_PROMPT = """\
You are a JavaScript security analyst extracting API information from code snippets.

For each snippet, identify:
1. API endpoints the code calls (that regex-based extraction might miss)
2. Authentication patterns (how tokens are stored/sent)
3. Custom API client patterns (wrappers around fetch/axios)

Respond with JSON:
{
  "endpoints": [
    {"path": "/api/...", "method": "GET|POST|PUT|DELETE|PATCH", "evidence": "brief reason"}
  ],
  "auth_patterns": ["description of auth pattern found"],
  "notes": "any other security-relevant observations"
}\
"""


@register_agent
class JSAnalyzerAgent(BaseAgent):
    name = "js_analyzer"
    description = "Downloads JS files and extracts API route references via regex and LLM."

    MIN_PATH_LENGTH = 4
    MAX_PATH_LENGTH = 200

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
        all_snippets: list[str] = []
        fetch_failures = 0
        files_analyzed = 0
        files_to_fetch = js_urls[:settings.max_js_files]

        semaphore = asyncio.Semaphore(_JS_FETCH_CONCURRENCY)

        async def fetch_and_analyze(js_url: str) -> list[Endpoint]:
            nonlocal fetch_failures, files_analyzed
            async with semaphore:
                resp = await self.http.get(js_url)
                if resp is None or resp.status_code != 200:
                    fetch_failures += 1
                    return []

                content_type = resp.headers.get("content-type", "").split(";")[0].strip()
                if content_type and content_type not in _JS_CONTENT_TYPES:
                    return []

                text = resp.text
                if len(text) > _MAX_JS_FILE_SIZE:
                    text = text[:_MAX_JS_FILE_SIZE]

                files_analyzed += 1
                eps = self._extract_paths_from_js(text, js_url, extracted_paths)

                # Collect interesting snippets for LLM analysis
                if len(all_snippets) < _MAX_TOTAL_SNIPPETS:
                    snippets = self._extract_interesting_snippets(text)
                    remaining = _MAX_TOTAL_SNIPPETS - len(all_snippets)
                    all_snippets.extend(snippets[:remaining])

                return eps

        tasks = [fetch_and_analyze(url) for url in files_to_fetch]
        results = await asyncio.gather(*tasks)
        for eps in results:
            endpoints.extend(eps)

        # LLM pass: analyze collected snippets for endpoints regex missed
        llm_endpoints, llm_findings = await self._llm_analyze_snippets(
            all_snippets, extracted_paths
        )
        endpoints.extend(llm_endpoints)
        findings.extend(llm_findings)

        if extracted_paths:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="js_api_routes",
                    title=f"Found {len(extracted_paths)} API paths in JavaScript",
                    detail=f"Analyzed {files_analyzed} JS files. Unique paths: {', '.join(list(extracted_paths)[:10])}",
                    severity=RiskLevel.INFO,
                )
            )

        if fetch_failures:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="js_fetch_failures",
                    title=f"Failed to fetch {fetch_failures} of {len(files_to_fetch)} JS files",
                    detail="Some JavaScript files could not be retrieved for analysis",
                    severity=RiskLevel.INFO,
                )
            )

        return AgentResult(
            agent_name=self.name,
            success=True,
            endpoints_found=endpoints,
            findings=findings,
            errors=errors,
            metadata={"js_files_analyzed": files_analyzed, "paths_extracted": len(extracted_paths)},
        )

    @staticmethod
    def _extract_interesting_snippets(
        text: str, max_snippets: int = _MAX_SNIPPETS_PER_FILE
    ) -> list[str]:
        """Extract code snippets around API-related patterns for LLM analysis."""
        snippets: list[str] = []
        seen_positions: set[int] = set()

        for match in _SNIPPET_TRIGGERS.finditer(text):
            pos = match.start()
            # Avoid overlapping snippets
            bucket = pos // _SNIPPET_CONTEXT_CHARS
            if bucket in seen_positions:
                continue
            seen_positions.add(bucket)

            start = max(0, pos - _SNIPPET_CONTEXT_CHARS // 4)
            end = min(len(text), pos + _SNIPPET_CONTEXT_CHARS)
            snippet = text[start:end].strip()
            if snippet:
                snippets.append(snippet)

            if len(snippets) >= max_snippets:
                break

        return snippets

    async def _llm_analyze_snippets(
        self,
        snippets: list[str],
        regex_found: set[str],
    ) -> tuple[list[Endpoint], list[Finding]]:
        """Send collected JS snippets to LLM for semantic analysis."""
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []

        if not snippets:
            return endpoints, findings

        llm_found_count = 0
        auth_patterns: list[str] = []

        for batch_idx in range(_MAX_LLM_BATCHES):
            start = batch_idx * _LLM_SNIPPET_BATCH_SIZE
            batch = snippets[start : start + _LLM_SNIPPET_BATCH_SIZE]
            if not batch:
                break

            snippet_text = "\n\n---\n\n".join(
                f"Snippet {i + 1}:\n{s}" for i, s in enumerate(batch)
            )

            messages = [
                {"role": "system", "content": _JS_LLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze these JavaScript code snippets:\n\n{snippet_text}",
                },
            ]

            try:
                response = await self.llm.chat_structured(
                    messages, response_model=JSAnalysisBatchResponse,
                    name=f"js_llm_batch_{batch_idx}", max_tokens=1024,
                )

                for ep_data in response.endpoints:
                    path = ep_data.path
                    method = ep_data.method.upper()
                    if not path:
                        continue

                    full_url = self._resolve_js_path(path)
                    if full_url is None or full_url in regex_found:
                        continue

                    regex_found.add(full_url)
                    llm_found_count += 1
                    endpoints.append(
                        Endpoint(
                            url=full_url,
                            method=method,
                            discovered_by=DiscoverySource.JS_LLM_ANALYSIS,
                            notes=f"LLM JS analysis: {ep_data.evidence}",
                        )
                    )

                auth_patterns.extend(response.auth_patterns)

            except Exception as e:
                logger.warning(
                    "JS LLM analysis batch %d failed (regex results intact): %s",
                    batch_idx, e,
                )
                continue

        if llm_found_count:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="js_llm_endpoints",
                    title=f"LLM found {llm_found_count} additional endpoints in JavaScript",
                    detail=f"LLM analyzed {len(snippets)} code snippets, found {llm_found_count} new endpoints",
                    severity=RiskLevel.INFO,
                )
            )

        if auth_patterns:
            unique_patterns = list(dict.fromkeys(auth_patterns))[:5]
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="js_auth_patterns",
                    title=f"Detected {len(unique_patterns)} auth pattern(s) in JavaScript",
                    detail="; ".join(unique_patterns),
                    severity=RiskLevel.MEDIUM,
                )
            )

        return endpoints, findings

    def _extract_paths_from_js(
        self, text: str, js_url: str, seen: set[str]
    ) -> list[Endpoint]:
        endpoints: list[Endpoint] = []

        # patterns that capture method + path
        for match in _AXIOS_METHOD_PATTERN.finditer(text):
            method = _METHOD_MAP.get(match.group(1).lower(), "GET")
            self._try_add_endpoint(
                match.group(2), method, js_url, seen, endpoints
            )

        for match in _XHR_OPEN_PATTERN.finditer(text):
            method = match.group(1).upper()
            self._try_add_endpoint(
                match.group(2), method, js_url, seen, endpoints
            )

        # patterns that only capture path (method defaults to GET)
        for pattern in JS_API_PATTERNS:
            for match in pattern.finditer(text):
                self._try_add_endpoint(
                    match.group(1), "GET", js_url, seen, endpoints
                )

        return endpoints

    def _try_add_endpoint(
        self,
        raw_path: str,
        method: str,
        js_url: str,
        seen: set[str],
        endpoints: list[Endpoint],
    ) -> None:
        """Resolve a path and add it as an endpoint if valid and not seen."""
        full_url = self._resolve_js_path(raw_path)
        if full_url is None or full_url in seen:
            return
        seen.add(full_url)
        endpoints.append(
            Endpoint(
                url=full_url,
                method=method,
                discovered_by=DiscoverySource.JS_ANALYSIS,
                notes=f"extracted from {js_url}",
            )
        )

    def _resolve_js_path(self, path: str) -> str | None:
        if len(path) < self.MIN_PATH_LENGTH or len(path) > self.MAX_PATH_LENGTH:
            return None
        if any(path.endswith(ext) or ext + "?" in path for ext in _STATIC_EXTENSIONS):
            return None

        # absolute URL
        if path.startswith(("http://", "https://", "ws://", "wss://")):
            parsed = urlparse(path)
            if not parsed.netloc or len(parsed.netloc) < 4:
                return None
            if self.http.is_same_origin(path):
                return path
            return None

        # root-relative path
        if path.startswith("/"):
            return self.http.resolve_url(path)

        # relative paths that look like API routes (e.g. "api/users")
        if path.startswith(("api/", "v1/", "v2/", "v3/")):
            return self.http.resolve_url("/" + path)

        return None
