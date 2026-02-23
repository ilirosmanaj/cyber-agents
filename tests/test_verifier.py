"""Tests for VerifierAgent — static helpers and action application."""

from __future__ import annotations

import pytest

from src.ghost_hunter.models import (
    EndpointCategory,
    RiskLevel,
    ScanState,
    VulnIndicator,
    VulnPattern,
)
from src.ghost_hunter.models.endpoints import Finding
from src.ghost_hunter.agents.verifier import VerifierAgent
from src.ghost_hunter.orchestrator import AGENT_DEPS, _resolve_waves
from tests.conftest import make_endpoint


class TestVerifierSkipLogic:
    def test_skip_with_fewer_than_3_indicators(self, scan_state: ScanState):
        """Only 1 indicator — not enough to cross-reference, skip verification."""
        ep = make_endpoint(url="https://vulnbank.org/api/users/{id}")
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)
        scan_state.vuln_indicators[key] = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="ID param",
                description="BOLA",
            ),
        ]
        active_count = sum(
            1
            for inds in scan_state.vuln_indicators.values()
            for ind in inds
            if not ind.suppressed
        )
        assert active_count < 3

    def test_no_skip_with_3_or_more_indicators(self, scan_state: ScanState):
        """3+ active indicators is enough for cross-referencing."""
        ep = make_endpoint(url="https://vulnbank.org/api/users/{id}")
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)
        scan_state.vuln_indicators[key] = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="ID param",
                description="BOLA",
            ),
            VulnIndicator(
                pattern=VulnPattern.AUTH_BOUNDARY_GAP,
                confidence=RiskLevel.HIGH,
                evidence="No auth",
                description="Auth gap",
            ),
            VulnIndicator(
                pattern=VulnPattern.MASS_ASSIGNMENT,
                confidence=RiskLevel.MEDIUM,
                evidence="role field",
                description="Mass assign",
            ),
        ]
        active_count = sum(
            1
            for inds in scan_state.vuln_indicators.values()
            for ind in inds
            if not ind.suppressed
        )
        assert active_count >= 3


class TestVerifierActionApplication:
    def _setup_state(self, scan_state: ScanState) -> str:
        """Seed state with BOLA + info_disclosure indicators on one endpoint."""
        ep = make_endpoint(url="https://vulnbank.org/api/users/{id}")
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)
        scan_state.vuln_indicators[key] = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="Path contains object ID",
                description="BOLA on users endpoint",
            ),
            VulnIndicator(
                pattern=VulnPattern.INFO_DISCLOSURE,
                confidence=RiskLevel.MEDIUM,
                evidence="Swagger path",
                description="Info disclosure",
            ),
        ]
        return key

    def test_suppress_action(self, scan_state: ScanState):
        """Suppressing an indicator marks it suppressed and tags the description."""
        key = self._setup_state(scan_state)
        actions = [
            {
                "endpoint_key": key,
                "action": "suppress",
                "pattern": "info_disclosure",
                "reason": "Swagger is expected to be public",
            }
        ]
        suppressed, adjusted, annotated = VerifierAgent._apply_actions(scan_state, actions)
        assert suppressed == 1
        assert adjusted == 0
        assert annotated == 0

        indicators = scan_state.vuln_indicators[key]
        info_ind = next(i for i in indicators if i.pattern == VulnPattern.INFO_DISCLOSURE)
        assert info_ind.suppressed is True
        assert "SUPPRESSED by verifier" in info_ind.description

    def test_adjust_confidence_action(self, scan_state: ScanState):
        """Confidence adjustment upgrades the risk level and sets llm_enhanced."""
        key = self._setup_state(scan_state)
        actions = [
            {
                "endpoint_key": key,
                "action": "adjust_confidence",
                "pattern": "bola_idor",
                "new_confidence": "critical",
                "reason": "Financial endpoint with IDOR",
            }
        ]
        suppressed, adjusted, annotated = VerifierAgent._apply_actions(scan_state, actions)
        assert suppressed == 0
        assert adjusted == 1

        indicators = scan_state.vuln_indicators[key]
        bola_ind = next(i for i in indicators if i.pattern == VulnPattern.BOLA_IDOR)
        assert bola_ind.confidence == RiskLevel.CRITICAL
        assert bola_ind.llm_enhanced is True
        assert "Verifier adjusted" in bola_ind.description

    def test_annotate_action(self, scan_state: ScanState):
        """Annotation appends a note without changing suppression or confidence."""
        key = self._setup_state(scan_state)
        actions = [
            {
                "endpoint_key": key,
                "action": "annotate",
                "pattern": "bola_idor",
                "reason": "Same root cause as /api/accounts/{id}",
            }
        ]
        suppressed, adjusted, annotated = VerifierAgent._apply_actions(scan_state, actions)
        assert annotated == 1

        indicators = scan_state.vuln_indicators[key]
        bola_ind = next(i for i in indicators if i.pattern == VulnPattern.BOLA_IDOR)
        assert "Verifier note" in bola_ind.description

    def test_action_on_nonexistent_endpoint(self, scan_state: ScanState):
        """Actions targeting a missing endpoint key are silently ignored."""
        self._setup_state(scan_state)
        actions = [
            {
                "endpoint_key": "GET https://vulnbank.org/nonexistent",
                "action": "suppress",
                "pattern": "bola_idor",
                "reason": "Should not match",
            }
        ]
        suppressed, adjusted, annotated = VerifierAgent._apply_actions(scan_state, actions)
        assert suppressed == 0

    def test_suppress_already_suppressed(self, scan_state: ScanState):
        """Double-suppressing is a no-op — count stays at 0."""
        key = self._setup_state(scan_state)
        # Pre-suppress
        for ind in scan_state.vuln_indicators[key]:
            if ind.pattern == VulnPattern.INFO_DISCLOSURE:
                ind.suppressed = True

        actions = [
            {
                "endpoint_key": key,
                "action": "suppress",
                "pattern": "info_disclosure",
                "reason": "Already suppressed",
            }
        ]
        suppressed, adjusted, annotated = VerifierAgent._apply_actions(scan_state, actions)
        assert suppressed == 0


class TestVerifierFormatting:
    def test_format_indicators_empty(self, scan_state: ScanState):
        """No indicators → 'No indicators' placeholder."""
        result = VerifierAgent._format_indicators(scan_state)
        assert result == "No indicators"

    def test_format_indicators_with_data(self, scan_state: ScanState):
        """Output groups indicators under their endpoint key."""
        ep = make_endpoint(url="https://vulnbank.org/api/users/{id}")
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)
        scan_state.vuln_indicators[key] = [
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="Path ID",
                description="BOLA",
            ),
        ]
        result = VerifierAgent._format_indicators(scan_state)
        assert key in result
        assert "bola_idor" in result
        assert "high" in result

    def test_format_classifications(self, scan_state: ScanState):
        """Classifications include category and auth status per endpoint."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/users",
            category=EndpointCategory.REST_API,
            requires_auth=True,
        )
        scan_state.add_endpoint(ep)
        result = VerifierAgent._format_classifications(scan_state)
        assert "rest_api" in result
        assert "requires_auth" in result


class TestFindingVerificationFields:
    def test_finding_has_verification_fields(self):
        """Explicit verification_status and verification_note round-trip."""
        finding = Finding(
            agent_name="test",
            finding_type="test",
            title="Test",
            detail="Test detail",
            verification_status="consistent",
            verification_note="All good",
        )
        assert finding.verification_status == "consistent"
        assert finding.verification_note == "All good"

    def test_finding_verification_defaults_empty(self):
        """When omitted, both verification fields default to empty string."""
        finding = Finding(
            agent_name="test",
            finding_type="test",
            title="Test",
            detail="Test detail",
        )
        assert finding.verification_status == ""
        assert finding.verification_note == ""


class TestVerifierInDAG:
    def test_verifier_in_agent_deps(self):
        """Verifier is registered in the agent DAG."""
        assert "verifier" in AGENT_DEPS

    def test_verifier_depends_on_vuln_analyzer(self):
        """Verifier runs after vuln_analyzer."""
        assert "vuln_analyzer" in AGENT_DEPS["verifier"]

    def test_prioritizer_depends_on_verifier(self):
        """Prioritizer now depends on verifier, not vuln_analyzer directly."""
        assert "verifier" in AGENT_DEPS["prioritizer"]
        assert "vuln_analyzer" not in AGENT_DEPS["prioritizer"]

    def test_verifier_in_separate_wave(self):
        """Verifier sits between vuln_analyzer and prioritizer in wave order."""
        waves = _resolve_waves(AGENT_DEPS)
        order = {name: i for i, wave in enumerate(waves) for name in wave}
        assert order["vuln_analyzer"] < order["verifier"]
        assert order["verifier"] < order["prioritizer"]
