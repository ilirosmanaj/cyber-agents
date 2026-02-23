"""Tests for PrioritizerAgent deterministic helpers and fallback logic."""

from __future__ import annotations

from src.ghost_hunter.agents.prioritizer import (
    _deterministic_risk,
    _format_ep_line,
    _highest_severity,
)
from src.ghost_hunter.models import (
    EndpointCategory,
    RiskLevel,
    ScanState,
    VulnIndicator,
    VulnPattern,
)
from src.ghost_hunter.agents.prioritizer import PrioritizerAgent
from tests.conftest import make_endpoint


# ---------------------------------------------------------------------------
# _highest_severity
# ---------------------------------------------------------------------------


class TestHighestSeverity:
    def test_returns_critical_from_mixed(self):
        """When indicators span LOW to CRITICAL, return CRITICAL."""
        indicators = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.LOW,
                evidence="test",
                description="test",
            ),
            VulnIndicator(
                pattern=VulnPattern.AUTH_BOUNDARY_GAP,
                confidence=RiskLevel.CRITICAL,
                evidence="test",
                description="test",
            ),
        ]
        assert _highest_severity(indicators) == RiskLevel.CRITICAL

    def test_suppressed_indicators_excluded(self):
        """Suppressed indicators should not affect severity."""
        indicators = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.CRITICAL,
                evidence="test",
                description="test",
                suppressed=True,
            ),
            VulnIndicator(
                pattern=VulnPattern.INFO_DISCLOSURE,
                confidence=RiskLevel.LOW,
                evidence="test",
                description="test",
            ),
        ]
        assert _highest_severity(indicators) == RiskLevel.LOW

    def test_all_suppressed_returns_info(self):
        """If every indicator is suppressed, default to INFO."""
        indicators = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="test",
                description="test",
                suppressed=True,
            ),
        ]
        assert _highest_severity(indicators) == RiskLevel.INFO

    def test_empty_list_returns_info(self):
        assert _highest_severity([]) == RiskLevel.INFO


# ---------------------------------------------------------------------------
# _deterministic_risk
# ---------------------------------------------------------------------------


class TestDeterministicRisk:
    def test_with_indicators_uses_highest(self):
        """Risk should match the highest active indicator severity."""
        ep = make_endpoint()
        indicators = [
            VulnIndicator(
                pattern=VulnPattern.SSRF,
                confidence=RiskLevel.HIGH,
                evidence="test",
                description="test",
            ),
        ]
        assert _deterministic_risk(ep=ep, indicators=indicators) == RiskLevel.HIGH

    def test_static_asset_gets_info(self):
        """Static assets with no vulns should be INFO."""
        ep = make_endpoint(category=EndpointCategory.STATIC_ASSET)
        assert _deterministic_risk(ep=ep, indicators=[]) == RiskLevel.INFO

    def test_health_check_gets_info(self):
        ep = make_endpoint(category=EndpointCategory.HEALTH_CHECK)
        assert _deterministic_risk(ep=ep, indicators=[]) == RiskLevel.INFO

    def test_no_auth_data_endpoint_gets_medium(self):
        """An unclassified endpoint with no auth should be MEDIUM."""
        ep = make_endpoint(requires_auth=False, category=EndpointCategory.REST_API)
        assert _deterministic_risk(ep=ep, indicators=[]) == RiskLevel.MEDIUM

    def test_auth_unknown_gets_low(self):
        """No indicators and auth unknown should default to LOW."""
        ep = make_endpoint(requires_auth=None, category=EndpointCategory.REST_API)
        assert _deterministic_risk(ep=ep, indicators=[]) == RiskLevel.LOW


# ---------------------------------------------------------------------------
# _format_ep_line
# ---------------------------------------------------------------------------


class TestFormatEpLine:
    def test_includes_method_and_url(self):
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users",
            method="POST",
        )
        line = _format_ep_line(index=1, ep=ep, indicators=[])
        assert "1." in line
        assert "POST" in line
        assert "/api/v1/users" in line

    def test_includes_vuln_tags(self):
        """Active indicators should appear as VULN=[...] in the line."""
        ep = make_endpoint()
        indicators = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="test",
                description="test",
            ),
        ]
        line = _format_ep_line(index=1, ep=ep, indicators=indicators)
        assert "VULN=[bola_idor(high)]" in line

    def test_suppressed_indicators_excluded_from_tags(self):
        """Suppressed indicators shouldn't show up in VULN tags."""
        ep = make_endpoint()
        indicators = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="test",
                description="test",
                suppressed=True,
            ),
        ]
        line = _format_ep_line(index=1, ep=ep, indicators=indicators)
        assert "VULN=" not in line

    def test_chain_count_shown(self):
        """When indicators have chain_ids, CHAINS= should appear."""
        ep = make_endpoint()
        indicators = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="test",
                description="test",
                chain_id="abc123",
            ),
            VulnIndicator(
                pattern=VulnPattern.AUTH_BOUNDARY_GAP,
                confidence=RiskLevel.HIGH,
                evidence="test",
                description="test",
                chain_id="abc123",
            ),
        ]
        line = _format_ep_line(index=1, ep=ep, indicators=indicators)
        assert "CHAINS=1" in line

    def test_body_preview_truncated(self):
        """Long response snippets should be truncated in context."""
        ep = make_endpoint(response_body_snippet="x" * 500)
        line = _format_ep_line(index=1, ep=ep, indicators=[])
        assert "body_preview:" in line
        # should be truncated to _MAX_SNIPPET_IN_CONTEXT (300)
        assert len(line) < 600


# ---------------------------------------------------------------------------
# Fallback entry creation
# ---------------------------------------------------------------------------


class TestFallbackEntry:
    def test_creates_entry_with_vuln_indicators(self):
        """Fallback should use vuln indicators for risk and include them."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users/{id}",
            category=EndpointCategory.REST_API,
        )
        state = ScanState(target="vulnbank.org", base_url="https://vulnbank.org")
        key = state.endpoint_key("GET", ep.url)
        indicator = VulnIndicator(
            pattern=VulnPattern.BOLA_IDOR,
            confidence=RiskLevel.HIGH,
            evidence="Path param",
            description="IDOR risk",
        )
        state.vuln_indicators[key] = [indicator]

        entry = PrioritizerAgent._fallback_entry(state=state, key=key, ep=ep)
        assert entry.risk_level == RiskLevel.HIGH
        assert len(entry.vuln_indicators) == 1
        assert "bola_idor" in entry.rationale

    def test_creates_info_entry_for_health_check(self):
        """Health checks with no vulns should get INFO risk."""
        ep = make_endpoint(
            url="https://vulnbank.org/health",
            category=EndpointCategory.HEALTH_CHECK,
        )
        state = ScanState(target="vulnbank.org", base_url="https://vulnbank.org")
        key = state.endpoint_key("GET", ep.url)

        entry = PrioritizerAgent._fallback_entry(state=state, key=key, ep=ep)
        assert entry.risk_level == RiskLevel.INFO
        assert "No vulnerability indicators" in entry.rationale

    def test_suppressed_indicators_excluded(self):
        """Suppressed vulns should not inflate the risk or appear in the entry."""
        ep = make_endpoint(category=EndpointCategory.REST_API)
        state = ScanState(target="vulnbank.org", base_url="https://vulnbank.org")
        key = state.endpoint_key("GET", ep.url)
        state.vuln_indicators[key] = [
            VulnIndicator(
                pattern=VulnPattern.INFO_DISCLOSURE,
                confidence=RiskLevel.CRITICAL,
                evidence="test",
                description="test",
                suppressed=True,
            ),
        ]

        entry = PrioritizerAgent._fallback_entry(state=state, key=key, ep=ep)
        assert entry.risk_level != RiskLevel.CRITICAL
        assert len(entry.vuln_indicators) == 0
