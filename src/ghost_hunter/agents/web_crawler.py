"""Async BFS web crawler with link, form, and script extraction."""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup, Comment

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

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication: strip fragment, trailing slash."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, ""))


_INTERESTING_HEADERS = {
    "x-debug", "x-request-id", "x-powered-by", "x-ratelimit-limit",
    "x-ratelimit-remaining", "x-ratelimit-reset", "www-authenticate",
    "x-frame-options", "x-content-type-options", "server",
}


@register_agent
class WebCrawlerAgent(BaseAgent):
    name = "web_crawler"
    description = "BFS crawls the target, extracting links, forms, and script URLs."

    _COMMENT_KEYWORDS = re.compile(
        r"(?:todo|fixme|hack|password|secret|api[_-]?key|token|debug|admin|internal)",
        re.IGNORECASE,
    )
    _LEAKED_SECRET_PATTERN = re.compile(
        r"""(?:api[_-]?key|secret|token|password|passwd|credential)\s*[=:]\s*['"]?[^\s'"]{8,}""",
        re.IGNORECASE,
    )
    _DEBUG_PATTERN = re.compile(
        r"(?:django\.debug|flask\.debug|DEBUG\s*=\s*True|traceback|stacktrace|Traceback \(most recent)",
        re.IGNORECASE,
    )

    async def run(self, state: ScanState) -> AgentResult:
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []
        errors: list[str] = []

        visited: set[str] = set()
        queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        queue.put_nowait((state.base_url, 0))

        pages_crawled = 0
        forms_found = 0

        while not queue.empty() and pages_crawled < settings.max_pages:
            url, depth = queue.get_nowait()
            normalized = _normalize_url(url)

            if normalized in visited:
                continue
            if depth > settings.max_crawl_depth:
                continue
            if not self.http.is_same_origin(url):
                continue

            visited.add(normalized)
            resp = await self.http.get(url)

            if resp is None:
                continue

            pages_crawled += 1

            content_type_header = resp.headers.get("content-type", "")

            page_ep = Endpoint(
                url=normalized,
                method="GET",
                status_code=resp.status_code,
                content_type=content_type_header,
                discovered_by=DiscoverySource.CRAWL,
                response_body_snippet=self.extract_body_snippet(resp),
            )
            endpoints.append(page_ep)

            if resp.status_code == 403:
                path = urlparse(normalized).path
                if path not in state.blocked_paths:
                    state.blocked_paths.append(path)
                continue

            if resp.status_code != 200:
                continue

            if "text/html" not in content_type_header:
                continue

            try:
                soup = BeautifulSoup(resp.text, "lxml")
            except Exception as e:
                errors.append(f"Parse error on {url}: {e}")
                continue

            self._extract_links(soup, normalized, depth, queue)
            form_eps = self._extract_forms(soup, normalized)
            endpoints.extend(form_eps)
            forms_found += len(form_eps)
            script_eps = self._extract_scripts(soup, normalized, state)
            endpoints.extend(script_eps)

            findings.extend(self._extract_html_intelligence(soup, normalized))

            # capture interesting response headers on the page endpoint
            for name, value in resp.headers.items():
                if name.lower() in _INTERESTING_HEADERS:
                    page_ep.response_headers[name.lower()] = value

        if pages_crawled > 0:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="crawl_summary",
                    title=f"Crawled {pages_crawled} pages",
                    detail=f"Discovered {len(endpoints)} endpoints, {forms_found} forms, {len(state.js_urls)} JS files",
                    severity=RiskLevel.INFO,
                )
            )

        if state.blocked_paths:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="blocked_paths",
                    title=f"Found {len(state.blocked_paths)} blocked (403) paths",
                    detail=f"Paths: {', '.join(state.blocked_paths[:10])}",
                    severity=RiskLevel.LOW,
                )
            )

        return AgentResult(
            agent_name=self.name,
            success=True,
            endpoints_found=endpoints,
            findings=findings,
            errors=errors,
            metadata={"pages_crawled": pages_crawled, "forms_found": forms_found},
        )

    def _extract_links(
        self,
        soup: BeautifulSoup,
        page_url: str,
        depth: int,
        queue: asyncio.Queue[tuple[str, int]],
    ) -> None:
        _SKIP_PREFIXES = ("#", "mailto:", "tel:", "javascript:")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith(_SKIP_PREFIXES):
                continue
            abs_url = urljoin(page_url, href)
            if self.http.is_same_origin(abs_url):
                queue.put_nowait((abs_url, depth + 1))

    def _extract_forms(self, soup: BeautifulSoup, page_url: str) -> list[Endpoint]:
        endpoints: list[Endpoint] = []
        for form in soup.find_all("form"):
            action = form.get("action", "")
            method = form.get("method", "GET").upper()
            abs_action = urljoin(page_url, action) if action else page_url
            if not self.http.is_same_origin(abs_action):
                continue
            params = [
                inp.get("name")
                for inp in form.find_all(["input", "select", "textarea"])
                if inp.get("name")
            ]
            endpoints.append(
                Endpoint(
                    url=_normalize_url(abs_action),
                    method=method,
                    discovered_by=DiscoverySource.CRAWL,
                    parameters=params,
                    notes=f"form on {page_url}",
                )
            )
        return endpoints

    def _extract_scripts(
        self, soup: BeautifulSoup, page_url: str, state: ScanState
    ) -> list[Endpoint]:
        endpoints: list[Endpoint] = []
        for script in soup.find_all("script", src=True):
            abs_src = urljoin(page_url, script["src"])
            if not self.http.is_same_origin(abs_src):
                continue
            state.js_urls.append(abs_src)
            endpoints.append(
                Endpoint(
                    url=abs_src,
                    method="GET",
                    discovered_by=DiscoverySource.CRAWL,
                    notes="script src",
                )
            )
        return endpoints

    def _extract_html_intelligence(
        self, soup: BeautifulSoup, page_url: str
    ) -> list[Finding]:
        """Analyze already-fetched HTML for leaked secrets, comments, and debug indicators."""
        findings: list[Finding] = []
        path = urlparse(page_url).path
        raw_html = str(soup)

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            if not self._COMMENT_KEYWORDS.search(str(comment)):
                continue
            findings.append(Finding(
                agent_name=self.name,
                finding_type="html_comment_leak",
                title=f"Sensitive HTML comment on {path}",
                detail="Comment contains sensitive keyword",
                severity=RiskLevel.LOW,
                evidence=str(comment).strip()[:120],
            ))

        for match in self._LEAKED_SECRET_PATTERN.finditer(raw_html):
            findings.append(Finding(
                agent_name=self.name,
                finding_type="leaked_secret",
                title=f"Potential secret leak on {path}",
                detail="Credential or API key pattern detected in page source",
                severity=RiskLevel.HIGH,
                evidence=match.group(0)[:100],
            ))

        for match in self._DEBUG_PATTERN.finditer(raw_html):
            findings.append(Finding(
                agent_name=self.name,
                finding_type="debug_indicator",
                title=f"Debug indicator on {path}",
                detail="Framework debug mode or stack trace pattern detected",
                severity=RiskLevel.MEDIUM,
                evidence=match.group(0)[:100],
            ))

        return findings
