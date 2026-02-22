"""Tests for Pydantic models — serialization, merging, and edge cases."""

from __future__ import annotations

from src.ghost_hunter.models import (
    AgentResult,
    DiscoverySource,
    Endpoint,
    Finding,
    RiskLevel,
    ScanState,
    VulnIndicator,
    VulnPattern,
)
from tests.conftest import make_endpoint


# ---------------------------------------------------------------------------
# ScanState.endpoint_key
# ---------------------------------------------------------------------------


class TestEndpointKey:
    """Tests for ScanState.endpoint_key()."""

    def test_format_consistency(self, scan_state: ScanState):
        """Given method and URL, endpoint_key returns 'METHOD URL' format."""
        key = scan_state.endpoint_key("GET", "https://vulnbank.org/api/v1/users")
        assert key == "GET https://vulnbank.org/api/v1/users"

    def test_uppercase_method(self, scan_state: ScanState):
        """Lowercase 'get' should be normalized to 'GET' in the key."""
        key = scan_state.endpoint_key("get", "https://vulnbank.org/api/v1/users")
        assert key == "GET https://vulnbank.org/api/v1/users"

    def test_strips_trailing_slash(self, scan_state: ScanState):
        """Trailing slash on URL should be stripped for consistent keys."""
        key = scan_state.endpoint_key("GET", "https://vulnbank.org/api/v1/users/")
        assert key == "GET https://vulnbank.org/api/v1/users"


# ---------------------------------------------------------------------------
# ScanState.add_endpoint — new vs merge
# ---------------------------------------------------------------------------


class TestAddEndpoint:
    """Tests for ScanState.add_endpoint() and merge behavior."""

    def test_add_new_endpoint(self, scan_state: ScanState):
        """Given a new endpoint, add_endpoint returns True and stores it."""
        ep = make_endpoint()
        assert scan_state.add_endpoint(ep) is True
        assert len(scan_state.endpoints) == 1

    def test_merge_duplicate_endpoint(self, scan_state: ScanState):
        """Given a duplicate endpoint, add_endpoint returns False and merges data."""
        ep1 = make_endpoint(status_code=200)
        ep2 = make_endpoint(status_code=None, requires_auth=True)

        scan_state.add_endpoint(ep1)
        result = scan_state.add_endpoint(ep2)

        assert result is False
        assert len(scan_state.endpoints) == 1

        # merged: keeps first status_code, gets requires_auth from second
        key = scan_state.endpoint_key("GET", ep1.url)
        merged = scan_state.endpoints[key]
        assert merged.status_code == 200
        assert merged.requires_auth is True

    def test_merge_parameters(self, scan_state: ScanState):
        """Given overlapping parameters, merge deduplicates them."""
        ep1 = make_endpoint(parameters=["id", "name"])
        ep2 = make_endpoint(parameters=["name", "email"])

        scan_state.add_endpoint(ep1)
        scan_state.add_endpoint(ep2)

        key = scan_state.endpoint_key("GET", ep1.url)
        merged = scan_state.endpoints[key]
        assert merged.parameters == ["id", "name", "email"]

    def test_merge_response_body_snippet(self, scan_state: ScanState):
        """Given second endpoint with response_body_snippet, merge preserves first if set."""
        ep1 = make_endpoint(response_body_snippet='{"data": "first"}')
        ep2 = make_endpoint(response_body_snippet='{"data": "second"}')

        scan_state.add_endpoint(ep1)
        scan_state.add_endpoint(ep2)

        key = scan_state.endpoint_key("GET", ep1.url)
        merged = scan_state.endpoints[key]
        assert merged.response_body_snippet == '{"data": "first"}'

    def test_merge_response_body_snippet_fills_empty(self, scan_state: ScanState):
        """Given first endpoint without snippet, merge fills it from second."""
        ep1 = make_endpoint(response_body_snippet="")
        ep2 = make_endpoint(response_body_snippet='{"data": "filled"}')

        scan_state.add_endpoint(ep1)
        scan_state.add_endpoint(ep2)

        key = scan_state.endpoint_key("GET", ep1.url)
        merged = scan_state.endpoints[key]
        assert merged.response_body_snippet == '{"data": "filled"}'


# ---------------------------------------------------------------------------
# Endpoint serialization round-trip
# ---------------------------------------------------------------------------


class TestEndpointSerialization:
    """Tests for Endpoint model serialization and deserialization."""

    def test_model_dump_and_load(self):
        """Given an endpoint, model_dump and model_validate produce equal objects."""
        ep = make_endpoint(
            parameters=["id", "name"],
            requires_auth=True,
            response_body_snippet='{"test": true}',
        )
        data = ep.model_dump()
        restored = Endpoint.model_validate(data)
        assert restored == ep

    def test_response_body_snippet_default(self):
        """Snippet should default to empty string when not provided."""
        ep = Endpoint(
            url="https://vulnbank.org/test",
            discovered_by=DiscoverySource.CRAWL,
        )
        assert ep.response_body_snippet == ""


# ---------------------------------------------------------------------------
# Finding model — validated field
# ---------------------------------------------------------------------------


class TestFinding:
    """Tests for Finding model with validation fields."""

    def test_finding_default_not_validated(self):
        """Fresh findings start as untested (validated=None)."""
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="Test finding",
            detail="Test detail",
        )
        assert f.validated is None
        assert f.validation_evidence == ""

    def test_finding_validated_true(self):
        """Given a validated finding, validated=True is preserved."""
        f = Finding(
            agent_name="probe_validator",
            finding_type="probe_auth_bypass",
            title="Auth Bypass CONFIRMED",
            detail="Endpoint returned 200 without credentials",
            validated=True,
            validation_evidence="HTTP 200 with user data",
        )
        assert f.validated is True
        assert f.validation_evidence == "HTTP 200 with user data"

    def test_finding_serialization_round_trip(self):
        """Given a finding with validation fields, serialization round-trip works."""
        f = Finding(
            agent_name="probe_validator",
            finding_type="probe_bola_idor",
            title="BOLA CONFIRMED",
            detail="Different IDs return same data",
            severity=RiskLevel.CRITICAL,
            validated=True,
            validation_evidence="Both /users/1 and /users/9999 returned 200",
        )
        data = f.model_dump()
        restored = Finding.model_validate(data)
        assert restored == f
        assert restored.validated is True


# ---------------------------------------------------------------------------
# VulnIndicator suppression
# ---------------------------------------------------------------------------


class TestVulnIndicatorSuppression:
    """Tests for VulnIndicator suppression flag behavior."""

    def test_default_not_suppressed(self):
        """Indicators start as not-suppressed by default."""
        ind = VulnIndicator(
            pattern=VulnPattern.BOLA_IDOR,
            confidence=RiskLevel.HIGH,
            evidence="test",
            description="test",
        )
        assert ind.suppressed is False

    def test_suppressed_indicator(self):
        """Given suppressed=True, indicator is marked as suppressed."""
        ind = VulnIndicator(
            pattern=VulnPattern.INFO_DISCLOSURE,
            confidence=RiskLevel.MEDIUM,
            evidence="test",
            description="test",
            suppressed=True,
        )
        assert ind.suppressed is True

    def test_chain_id_default_none(self):
        """Given no chain_id, default is None."""
        ind = VulnIndicator(
            pattern=VulnPattern.BOLA_IDOR,
            confidence=RiskLevel.HIGH,
            evidence="test",
            description="test",
        )
        assert ind.chain_id is None

    def test_llm_enhanced_flag(self):
        """Given llm_enhanced=True, flag is preserved."""
        ind = VulnIndicator(
            pattern=VulnPattern.BOLA_IDOR,
            confidence=RiskLevel.HIGH,
            evidence="test",
            description="test",
            llm_enhanced=True,
        )
        assert ind.llm_enhanced is True


# ---------------------------------------------------------------------------
# ScanState.merge_agent_result
# ---------------------------------------------------------------------------


class TestMergeAgentResult:
    """Tests for ScanState.merge_agent_result()."""

    def test_merge_records_agent_completed(self, scan_state: ScanState):
        """Given an agent result, merge records the agent as completed."""
        result = AgentResult(
            agent_name="test_agent",
            success=True,
            endpoints_found=[make_endpoint()],
            findings=[
                Finding(
                    agent_name="test_agent",
                    finding_type="test",
                    title="Test",
                    detail="Test detail",
                )
            ],
        )
        scan_state.merge_agent_result(result)
        assert "test_agent" in scan_state.agents_completed
        assert len(scan_state.endpoints) == 1
        assert len(scan_state.findings) == 1

    def test_merge_does_not_duplicate_agent(self, scan_state: ScanState):
        """Merging the same agent twice shouldn't create a duplicate entry."""
        result = AgentResult(agent_name="test_agent", success=True)
        scan_state.merge_agent_result(result)
        scan_state.merge_agent_result(result)
        assert scan_state.agents_completed.count("test_agent") == 1
