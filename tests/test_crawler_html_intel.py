"""Tests for WebCrawlerAgent HTML intelligence — LLM-based analysis."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from bs4 import BeautifulSoup

from src.ghost_hunter.agents.web_crawler import WebCrawlerAgent
from src.ghost_hunter.models import Finding, RiskLevel
from src.ghost_hunter.models.llm_responses import (
    HTMLIntelAnalysisResponse,
    HTMLIntelFinding,
)


def _make_crawler() -> WebCrawlerAgent:
    """Create a WebCrawlerAgent with mocked dependencies."""
    return WebCrawlerAgent(
        http_client=MagicMock(),
        llm_client=MagicMock(),
        prompt_registry=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Content collection
# ---------------------------------------------------------------------------


class TestCollectHTMLIntelContent:
    """Tests for _collect_html_intel_content."""

    def test_collects_html_comments(self):
        """Comments like <!-- TODO: remove password --> should appear in output."""
        html = "<html><!-- TODO: remove hardcoded password --><body>Hello</body></html>"
        soup = BeautifulSoup(html, "html.parser")
        content = WebCrawlerAgent._collect_html_intel_content(soup)
        assert "HTML COMMENTS" in content
        assert "TODO: remove hardcoded password" in content

    def test_collects_inline_scripts(self):
        """Inline <script> blocks end up in the output under INLINE SCRIPTS."""
        html = '<html><body><script>var apiKey = "sk-secret-123";</script></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        content = WebCrawlerAgent._collect_html_intel_content(soup)
        assert "INLINE SCRIPTS" in content
        assert 'apiKey = "sk-secret-123"' in content

    def test_skips_external_scripts(self):
        """Scripts with src attribute are skipped (they're fetched separately)."""
        html = '<html><body><script src="/static/app.js"></script></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        content = WebCrawlerAgent._collect_html_intel_content(soup)
        assert content == ""

    def test_empty_page_returns_empty(self):
        """Plain HTML with no comments or scripts → empty string."""
        html = "<html><body><p>Hello world</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        content = WebCrawlerAgent._collect_html_intel_content(soup)
        assert content == ""

    def test_tiny_comments_skipped(self):
        """Comments <= 3 chars are noise, not worth sending to the LLM."""
        html = "<html><!-- x --><body>Hello</body></html>"
        soup = BeautifulSoup(html, "html.parser")
        content = WebCrawlerAgent._collect_html_intel_content(soup)
        assert content == ""

    def test_tiny_scripts_skipped(self):
        """Scripts <= 10 chars (like `x=1;`) are too short to contain secrets."""
        html = "<html><body><script>x=1;</script></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        content = WebCrawlerAgent._collect_html_intel_content(soup)
        assert content == ""


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------


class TestHTMLIntelResultProcessing:
    """Tests for _process_html_intel_results."""

    def test_leaked_secret_gets_high_severity(self):
        """leaked_secret → HIGH severity by default."""
        crawler = _make_crawler()
        response = HTMLIntelAnalysisResponse(
            reasoning="Found API key in inline script",
            findings=[
                HTMLIntelFinding(
                    finding_type="leaked_secret",
                    evidence='apiKey = "sk-proj-abc..."',
                    context="API key assigned in inline JavaScript",
                    confidence="high",
                    is_placeholder=False,
                ),
            ],
        )
        batch = [("https://vulnbank.org/dashboard", "some content")]
        results = crawler._process_html_intel_results(response, batch)

        assert len(results) == 1
        assert results[0].finding_type == "leaked_secret"
        assert results[0].severity == RiskLevel.HIGH

    def test_debug_indicator_gets_medium_severity(self):
        """Django DEBUG=True should come back as MEDIUM."""
        crawler = _make_crawler()
        response = HTMLIntelAnalysisResponse(
            reasoning="Found Django debug mode",
            findings=[
                HTMLIntelFinding(
                    finding_type="debug_indicator",
                    evidence="DEBUG = True in settings dump",
                    context="Django debug mode is enabled in production",
                    confidence="medium",
                    is_placeholder=False,
                ),
            ],
        )
        batch = [("https://vulnbank.org/debug", "some content")]
        results = crawler._process_html_intel_results(response, batch)

        assert len(results) == 1
        assert results[0].finding_type == "debug_indicator"
        assert results[0].severity == RiskLevel.MEDIUM

    def test_sensitive_comment_produces_finding(self):
        """Auth bypass comment in HTML → finding with sensitive_comment type."""
        crawler = _make_crawler()
        response = HTMLIntelAnalysisResponse(
            reasoning="Found comment about auth bypass",
            findings=[
                HTMLIntelFinding(
                    finding_type="sensitive_comment",
                    evidence="<!-- HACK: bypassing auth for dev -->",
                    context="Developer comment about auth bypass",
                    confidence="medium",
                    is_placeholder=False,
                ),
            ],
        )
        batch = [("https://vulnbank.org/login", "some content")]
        results = crawler._process_html_intel_results(response, batch)

        assert len(results) == 1
        assert results[0].finding_type == "sensitive_comment"

    def test_placeholder_values_ignored(self):
        """'your_key_here' is obviously not a real secret — skip it."""
        crawler = _make_crawler()
        response = HTMLIntelAnalysisResponse(
            reasoning="Found example value",
            findings=[
                HTMLIntelFinding(
                    finding_type="leaked_secret",
                    evidence='apiKey = "your_key_here"',
                    context="Placeholder in documentation",
                    confidence="high",
                    is_placeholder=True,
                ),
            ],
        )
        batch = [("https://vulnbank.org/docs", "some content")]
        results = crawler._process_html_intel_results(response, batch)

        assert len(results) == 0

    def test_critical_confidence_overrides_default_severity(self):
        """LLM says 'critical' → severity should be CRITICAL, not the default HIGH for leaked_secret."""
        crawler = _make_crawler()
        response = HTMLIntelAnalysisResponse(
            reasoning="Found private key",
            findings=[
                HTMLIntelFinding(
                    finding_type="leaked_secret",
                    evidence="-----BEGIN RSA PRIVATE KEY-----",
                    context="Private key embedded in page source",
                    confidence="critical",
                    is_placeholder=False,
                ),
            ],
        )
        batch = [("https://vulnbank.org/config", "some content")]
        results = crawler._process_html_intel_results(response, batch)

        assert len(results) == 1
        assert results[0].severity == RiskLevel.CRITICAL

    def test_source_url_routes_to_correct_page(self):
        """Finding on /console shouldn't end up attributed to /register."""
        crawler = _make_crawler()
        response = HTMLIntelAnalysisResponse(
            reasoning="Found secret on console, not register",
            findings=[
                HTMLIntelFinding(
                    finding_type="leaked_secret",
                    evidence="SECRET_KEY = ...",
                    context="Secret in debug console",
                    source_url="https://target.com/console",
                    confidence="critical",
                    is_placeholder=False,
                ),
            ],
        )
        batch = [
            ("https://target.com/register", "register html"),
            ("https://target.com/console", "console html"),
        ]
        results = crawler._process_html_intel_results(response, batch)

        assert len(results) == 1
        assert "/console" in results[0].title
        assert "/register" not in results[0].title

    def test_single_page_batch_doesnt_need_source_url(self):
        """With only one page in the batch, source_url is unnecessary."""
        crawler = _make_crawler()
        response = HTMLIntelAnalysisResponse(
            reasoning="Found something",
            findings=[
                HTMLIntelFinding(
                    finding_type="debug_indicator",
                    evidence="DEBUG = True",
                    context="Debug mode",
                    confidence="high",
                    is_placeholder=False,
                ),
            ],
        )
        batch = [("https://target.com/admin", "admin html")]
        results = crawler._process_html_intel_results(response, batch)

        assert len(results) == 1
        assert "/admin" in results[0].title

    def test_ambiguous_batch_falls_back_to_root(self):
        """No source_url + multiple pages = can't guess, so path becomes /."""
        crawler = _make_crawler()
        response = HTMLIntelAnalysisResponse(
            reasoning="Found something",
            findings=[
                HTMLIntelFinding(
                    finding_type="debug_indicator",
                    evidence="DEBUG = True",
                    context="Debug mode",
                    confidence="high",
                    is_placeholder=False,
                ),
            ],
        )
        batch = [
            ("https://target.com/page1", "html1"),
            ("https://target.com/page2", "html2"),
        ]
        results = crawler._process_html_intel_results(response, batch)

        assert len(results) == 1
        # can't determine which page, so falls back to /
        assert results[0].title == "Debug Indicator on /"


# ---------------------------------------------------------------------------
# LLM pass integration
# ---------------------------------------------------------------------------


class TestHTMLIntelLLMPass:
    """Tests for _llm_html_intel_pass end-to-end."""

    def test_llm_failure_graceful(self):
        """LLM timeout shouldn't crash the crawler — just return nothing."""
        crawler = _make_crawler()
        crawler.llm = AsyncMock()
        crawler.llm.chat_structured.side_effect = RuntimeError("LLM timeout")
        crawler.prompt_registry = MagicMock()

        candidates = {
            "https://vulnbank.org/admin": "<!-- admin password: secret123 -->",
        }

        results = asyncio.get_event_loop().run_until_complete(
            crawler._llm_html_intel_pass(candidates)
        )

        assert results == []

    def test_multiple_findings_across_batch(self):
        """Both a leaked_secret and a debug_indicator from one batch come through."""
        crawler = _make_crawler()
        crawler.llm = AsyncMock()
        crawler.llm.chat_structured.return_value = HTMLIntelAnalysisResponse(
            reasoning="Found secrets",
            findings=[
                HTMLIntelFinding(
                    finding_type="leaked_secret",
                    evidence='token = "abc123..."',
                    context="Auth token in script",
                    confidence="high",
                    is_placeholder=False,
                ),
                HTMLIntelFinding(
                    finding_type="debug_indicator",
                    evidence="DEBUG = True",
                    context="Debug mode enabled",
                    confidence="medium",
                    is_placeholder=False,
                ),
            ],
        )
        crawler.prompt_registry = MagicMock()

        candidates = {
            "https://vulnbank.org/app": "some html content with scripts",
        }

        results = asyncio.get_event_loop().run_until_complete(
            crawler._llm_html_intel_pass(candidates)
        )

        assert len(results) == 2
        types = {r.finding_type for r in results}
        assert "leaked_secret" in types
        assert "debug_indicator" in types
