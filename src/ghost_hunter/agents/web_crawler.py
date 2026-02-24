"""Async BFS web crawler with link, form, script extraction, and LLM HTML analysis."""

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
from src.ghost_hunter.models.llm_responses import HTMLIntelAnalysisResponse

logger = logging.getLogger(__name__)

_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}

# scan_depth multipliers for max_pages and max_crawl_depth
_SHALLOW_PAGES_RATIO = 0.25
_SHALLOW_DEPTH_RATIO = 0.5
_DEEP_PAGES_RATIO = 2.0
_DEEP_DEPTH_RATIO = 1.5

# caps for HTML content sent to LLM
_MAX_INTEL_CONTENT_CHARS = 4000
_MAX_COMMENTS_PER_PAGE = 20
_MAX_SCRIPTS_PER_PAGE = 10
_MAX_SCRIPT_CHARS = 2000

# patterns for detecting API hints inside inline scripts and data-* attributes
_API_HINT_PATTERN = re.compile(
    r"""(?:/api/|/graphql|/swagger|/openapi|/rest/)""", re.IGNORECASE
)


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

_LINK_SKIP_PREFIXES = ("#", "mailto:", "tel:", "javascript:")


def _is_html_content_type(content_type_header: str) -> bool:
    """Check if the Content-Type indicates HTML (ignoring charset)."""
    mime = content_type_header.split(";")[0].strip().lower()
    return mime in _HTML_CONTENT_TYPES


@register_agent
class WebCrawlerAgent(BaseAgent):
    name = "web_crawler"
    description = "BFS crawls the target, extracting links, forms, and script URLs."

    _HTML_INTEL_BATCH_SIZE = 4

    async def run(self, state: ScanState) -> AgentResult:
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []
        errors: list[str] = []

        visited: set[str] = set()
        queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()

        # seed from base_url and any endpoints already discovered (sitemap, robots)
        self._seed_queue(state, queue)

        pages_crawled = 0
        forms_found = 0
        seen_forms: set[str] = set()
        html_intel_candidates: dict[str, str] = {}  # page_url -> extracted content

        # compute effective limits from scan_depth strategy
        max_pages = settings.max_pages
        max_depth = settings.max_crawl_depth
        if state.scan_strategy and state.scan_strategy.scan_depth:
            depth_setting = state.scan_strategy.scan_depth
            if depth_setting == "shallow":
                max_pages = max(1, int(settings.max_pages * _SHALLOW_PAGES_RATIO))
                max_depth = max(1, int(settings.max_crawl_depth * _SHALLOW_DEPTH_RATIO))
            elif depth_setting == "deep":
                max_pages = int(settings.max_pages * _DEEP_PAGES_RATIO)
                max_depth = int(settings.max_crawl_depth * _DEEP_DEPTH_RATIO)

        while not queue.empty() and pages_crawled < max_pages:
            url, depth = queue.get_nowait()
            normalized = _normalize_url(url)

            if normalized in visited:
                continue
            if depth > max_depth:
                continue
            if not self.http.is_same_origin(url):
                continue

            path = urlparse(normalized).path
            if path in state.blocked_paths:
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
                if path not in state.blocked_paths:
                    state.blocked_paths.append(path)
                continue

            if resp.status_code != 200:
                continue

            if not _is_html_content_type(content_type_header):
                continue

            try:
                soup = BeautifulSoup(resp.text, "lxml")
            except Exception as e:
                errors.append(f"Parse error on {url}: {e}")
                continue

            base_href = self._extract_base_href(soup, normalized)

            self._extract_links(soup, base_href, depth, queue)
            form_eps = self._extract_forms(soup, base_href, seen_forms)
            endpoints.extend(form_eps)
            forms_found += len(form_eps)
            script_eps = self._extract_scripts(soup, base_href, state)
            endpoints.extend(script_eps)

            self._extract_additional_urls(soup, base_href, depth, queue, state)

            intel_content = self._collect_html_intel_content(soup)
            if intel_content:
                html_intel_candidates[normalized] = intel_content
            findings.extend(self._extract_api_hints(soup, normalized))

            for name, value in resp.headers.items():
                if name.lower() in _INTERESTING_HEADERS:
                    page_ep.response_headers[name.lower()] = value

        if html_intel_candidates:
            logger.info("LLM HTML analysis on %d pages", len(html_intel_candidates))
            intel_findings = await self._llm_html_intel_pass(html_intel_candidates)
            findings.extend(intel_findings)
            logger.info("LLM HTML analysis found %d findings", len(intel_findings))

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

    @staticmethod
    def _seed_queue(
        state: ScanState, queue: asyncio.Queue[tuple[str, int]]
    ) -> None:
        """Seed the BFS queue from base_url and any already-discovered endpoints."""
        queue.put_nowait((state.base_url, 0))
        for ep in state.endpoints.values():
            if ep.discovered_by in (DiscoverySource.SITEMAP, DiscoverySource.ROBOTS_TXT):
                queue.put_nowait((ep.url, 1))

    @staticmethod
    def _extract_base_href(soup: BeautifulSoup, page_url: str) -> str:
        """Extract <base href> from the document, falling back to page_url."""
        base_tag = soup.find("base", href=True)
        if base_tag:
            return base_tag["href"]
        return page_url

    def _extract_links(
        self,
        soup: BeautifulSoup,
        base_href: str,
        depth: int,
        queue: asyncio.Queue[tuple[str, int]],
    ) -> None:
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith(_LINK_SKIP_PREFIXES):
                continue
            abs_url = urljoin(base_href, href)
            if self.http.is_same_origin(abs_url):
                queue.put_nowait((abs_url, depth + 1))

    def _extract_forms(
        self,
        soup: BeautifulSoup,
        base_href: str,
        seen_forms: set[str],
    ) -> list[Endpoint]:
        endpoints: list[Endpoint] = []
        for form in soup.find_all("form"):
            action = form.get("action", "")
            method = form.get("method", "GET").upper()
            abs_action = urljoin(base_href, action) if action else base_href
            if not self.http.is_same_origin(abs_action):
                continue

            normalized_action = _normalize_url(abs_action)
            form_key = f"{method} {normalized_action}"
            if form_key in seen_forms:
                continue
            seen_forms.add(form_key)

            enctype = form.get("enctype", "")
            params: list[str] = []
            has_file_input = False

            for inp in form.find_all(["input", "select", "textarea"]):
                name = inp.get("name")
                if not name:
                    continue
                params.append(name)
                if inp.get("type", "").lower() == "file":
                    has_file_input = True

            content_type = enctype if enctype else None
            if has_file_input and not enctype:
                content_type = "multipart/form-data"

            endpoints.append(
                Endpoint(
                    url=normalized_action,
                    method=method,
                    discovered_by=DiscoverySource.CRAWL,
                    parameters=params,
                    request_body_content_type=content_type,
                    notes=f"form on {base_href}",
                )
            )
        return endpoints

    def _extract_scripts(
        self, soup: BeautifulSoup, base_href: str, state: ScanState
    ) -> list[Endpoint]:
        endpoints: list[Endpoint] = []
        for script in soup.find_all("script", src=True):
            abs_src = urljoin(base_href, script["src"])
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

    def _extract_additional_urls(
        self,
        soup: BeautifulSoup,
        base_href: str,
        depth: int,
        queue: asyncio.Queue[tuple[str, int]],
        state: ScanState,
    ) -> None:
        """Extract URLs from iframes, stylesheets, srcset, and data-* attributes."""
        for iframe in soup.find_all("iframe", src=True):
            abs_url = urljoin(base_href, iframe["src"])
            if self.http.is_same_origin(abs_url):
                queue.put_nowait((abs_url, depth + 1))

        for link in soup.find_all("link", rel=True, href=True):
            abs_url = urljoin(base_href, link["href"])
            if not self.http.is_same_origin(abs_url):
                continue
            rels = link.get("rel", [])
            if "stylesheet" in rels or "preload" in rels:
                state.js_urls.append(abs_url)

        for tag in soup.find_all(attrs={"srcset": True}):
            for entry in tag["srcset"].split(","):
                src = entry.strip().split()[0] if entry.strip() else ""
                if not src:
                    continue
                abs_url = urljoin(base_href, src)
                if self.http.is_same_origin(abs_url):
                    queue.put_nowait((abs_url, depth + 1))

        for tag in soup.find_all(attrs=True):
            for attr_name, attr_val in tag.attrs.items():
                if not attr_name.startswith("data-"):
                    continue
                if not isinstance(attr_val, str):
                    continue
                if attr_val.startswith("/") or attr_val.startswith("http"):
                    abs_url = urljoin(base_href, attr_val)
                    if self.http.is_same_origin(abs_url):
                        queue.put_nowait((abs_url, depth + 1))

    def _extract_api_hints(
        self, soup: BeautifulSoup, page_url: str
    ) -> list[Finding]:
        """Detect API paths in inline scripts and data-* attributes."""
        findings: list[Finding] = []
        hints: set[str] = set()
        path = urlparse(page_url).path

        for script in soup.find_all("script", src=False):
            text = script.string or ""
            for match in _API_HINT_PATTERN.finditer(text):
                hints.add(match.group(0))

        for tag in soup.find_all(attrs=True):
            for attr_name, attr_val in tag.attrs.items():
                if not attr_name.startswith("data-"):
                    continue
                if isinstance(attr_val, str) and _API_HINT_PATTERN.search(attr_val):
                    hints.add(attr_val[:80])

        if hints:
            findings.append(Finding(
                agent_name=self.name,
                finding_type="api_hints_in_html",
                title=f"API hints in HTML on {path}",
                detail=f"Found {len(hints)} API-related references in inline scripts or data attributes",
                severity=RiskLevel.INFO,
                evidence=", ".join(sorted(hints)[:5]),
            ))

        return findings

    # ------------------------------------------------------------------
    # LLM HTML intelligence analysis
    # ------------------------------------------------------------------

    _FINDING_SEVERITY_MAP: dict[str, RiskLevel] = {
        "leaked_secret": RiskLevel.HIGH,
        "sensitive_comment": RiskLevel.LOW,
        "debug_indicator": RiskLevel.MEDIUM,
    }

    @staticmethod
    def _collect_html_intel_content(soup: BeautifulSoup) -> str:
        """Extract HTML comments and inline scripts for LLM analysis."""
        parts: list[str] = []

        comments = [
            str(c).strip()
            for c in soup.find_all(string=lambda text: isinstance(text, Comment))
            if len(str(c).strip()) > 3
        ]
        if comments:
            parts.append("HTML COMMENTS:\n" + "\n".join(comments[:_MAX_COMMENTS_PER_PAGE]))

        scripts: list[str] = []
        for script in soup.find_all("script", src=False):
            text = (script.string or "").strip()
            if text and len(text) > 10:
                scripts.append(text[:_MAX_SCRIPT_CHARS])
        if scripts:
            parts.append("INLINE SCRIPTS:\n" + "\n---\n".join(scripts[:_MAX_SCRIPTS_PER_PAGE]))

        content = "\n\n".join(parts)
        return content[:_MAX_INTEL_CONTENT_CHARS] if content else ""

    async def _llm_html_intel_pass(
        self, candidates: dict[str, str]
    ) -> list[Finding]:
        """Batch-analyze collected HTML content with the LLM."""
        findings: list[Finding] = []
        items = list(candidates.items())

        for i in range(0, len(items), self._HTML_INTEL_BATCH_SIZE):
            batch = items[i : i + self._HTML_INTEL_BATCH_SIZE]
            try:
                batch_context = self._format_html_intel_batch(batch)
                messages = [
                    {
                        "role": "system",
                        "content": self.prompt_registry.get(
                            "crawler_html_analyzer"
                        ).system_prompt,
                    },
                    {"role": "user", "content": batch_context},
                ]
                response = await self.llm.chat_structured(
                    messages,
                    response_model=HTMLIntelAnalysisResponse,
                    name=f"html_intel_batch_{i // self._HTML_INTEL_BATCH_SIZE}",
                )
                findings.extend(
                    self._process_html_intel_results(response, batch)
                )
            except Exception as e:
                logger.warning(
                    "HTML intel batch %d failed: %s",
                    i // self._HTML_INTEL_BATCH_SIZE,
                    e,
                )
                continue

        return findings

    @staticmethod
    def _format_html_intel_batch(
        batch: list[tuple[str, str]],
    ) -> str:
        """Format HTML content for the LLM."""
        lines: list[str] = []
        for page_url, content in batch:
            path = urlparse(page_url).path
            lines.append(f"--- {path} ({page_url}) ---\n{content}\n")
        return "\n".join(lines)

    def _process_html_intel_results(
        self,
        response: HTMLIntelAnalysisResponse,
        batch: list[tuple[str, str]],
    ) -> list[Finding]:
        """Convert LLM findings into Finding objects."""
        results: list[Finding] = []
        batch_urls = [url for url, _ in batch]

        for finding in response.findings:
            if finding.is_placeholder:
                continue

            try:
                severity = RiskLevel(finding.confidence)
            except ValueError:
                severity = self._FINDING_SEVERITY_MAP.get(
                    finding.finding_type, RiskLevel.MEDIUM
                )

            page_path = self._resolve_finding_path(
                source_url=finding.source_url, batch_urls=batch_urls,
            )
            results.append(Finding(
                agent_name=self.name,
                finding_type=finding.finding_type,
                title=f"{finding.finding_type.replace('_', ' ').title()} on {page_path}",
                detail=finding.context,
                severity=severity,
                evidence=finding.evidence[:120],
            ))

        return results

    @staticmethod
    def _resolve_finding_path(
        source_url: str, batch_urls: list[str],
    ) -> str:
        """Resolve which page a finding belongs to and return its path."""
        fallback = urlparse(batch_urls[0]).path if batch_urls else "/"
        if not source_url:
            return fallback if len(batch_urls) == 1 else "/"
        # try exact or substring match against batch URLs
        for url in batch_urls:
            if source_url in url or url.endswith(source_url):
                return urlparse(url).path
        # source_url might already be a path
        if source_url.startswith("/"):
            return source_url
        return fallback
