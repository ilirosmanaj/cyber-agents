"""Tests for report helper functions — formatting, scaling, and fallback."""

from __future__ import annotations

from src.ghost_hunter.models import (
    AttackSurfaceEntry,
    DiscoverySource,
    Endpoint,
    EndpointCategory,
    Finding,
    RiskLevel,
    ScanState,
    TechFingerprint,
    VulnIndicator,
    VulnPattern,
)
from src.ghost_hunter.report import (
    CWE_REFERENCES,
    _BASE_REPORT_TOKENS,
    _MAX_REPORT_TOKENS,
    _TOKENS_PER_FINDING,
    _build_attack_surface_table,
    _build_cwe_reference_block,
    _build_fallback_report,
    _build_report_context,
    _build_tech_fingerprint_section,
    _compute_max_tokens,
    _format_finding,
)


class TestComputeMaxTokens:
    def test_small_scan_uses_base(self):
        """A scan with 0 findings should return the base token count."""
        assert _compute_max_tokens(0) == _BASE_REPORT_TOKENS

    def test_scales_with_findings(self):
        """10 findings should add 10 * _TOKENS_PER_FINDING to the base."""
        expected = _BASE_REPORT_TOKENS + 10 * _TOKENS_PER_FINDING
        assert _compute_max_tokens(10) == expected

    def test_caps_at_maximum(self):
        """Even with 500 findings, the result shouldn't exceed _MAX_REPORT_TOKENS."""
        assert _compute_max_tokens(500) == _MAX_REPORT_TOKENS

    def test_boundary_at_cap(self):
        """When findings push exactly to the cap, it should equal the cap."""
        exact = (_MAX_REPORT_TOKENS - _BASE_REPORT_TOKENS) // _TOKENS_PER_FINDING
        assert _compute_max_tokens(exact) <= _MAX_REPORT_TOKENS
        assert _compute_max_tokens(exact + 1) == _MAX_REPORT_TOKENS


class TestFormatFinding:
    def test_basic_finding(self):
        """A finding without validation should have no tag."""
        f = Finding(
            agent_name="vuln_analyzer",
            finding_type="info_disclosure",
            title="Debug endpoint exposed",
            detail="Werkzeug debugger accessible",
            severity=RiskLevel.CRITICAL,
        )
        result = _format_finding(f)
        assert "[CRITICAL]" in result
        assert "Debug endpoint exposed" in result
        assert "[VALIDATED]" not in result
        assert "[REFUTED]" not in result

    def test_validated_finding(self):
        """A validated finding gets the [VALIDATED] tag."""
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="Confirmed issue",
            detail="detail",
            validated=True,
        )
        assert "[VALIDATED]" in _format_finding(f)

    def test_refuted_finding(self):
        """A refuted finding gets the [REFUTED] tag."""
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="False positive",
            detail="detail",
            validated=False,
        )
        assert "[REFUTED]" in _format_finding(f)

    def test_evidence_included(self):
        """When evidence is present, it should appear in the output."""
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="Test",
            detail="detail",
            evidence="HTTP 200 with admin panel",
        )
        result = _format_finding(f)
        assert "Evidence: HTTP 200 with admin panel" in result

    def test_no_evidence_omitted(self):
        """When evidence is empty, 'Evidence:' line should not appear."""
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="Test",
            detail="detail",
        )
        assert "Evidence:" not in _format_finding(f)


class TestBuildCweReferenceBlock:
    def test_contains_all_patterns(self):
        """Every VulnPattern value should have a CWE mapping."""
        block = _build_cwe_reference_block()
        for pattern in VulnPattern:
            assert pattern.value in CWE_REFERENCES, f"Missing CWE for {pattern.value}"
            assert pattern.value in block

    def test_format_is_arrow_separated(self):
        """Each line should follow 'pattern → CWE-XXX' format."""
        block = _build_cwe_reference_block()
        for line in block.strip().split("\n")[1:]:
            assert "→" in line


class TestBuildTechFingerprint:
    def test_full_fingerprint(self):
        """All fields present should produce a multi-line block."""
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            tech_fingerprint=TechFingerprint(
                server="nginx/1.18",
                frameworks=["Flask"],
                technologies=["Python"],
                security_headers={"X-Frame-Options": "DENY"},
                missing_security_headers=["Content-Security-Policy"],
                cookies=["session_id"],
            ),
        )
        result = _build_tech_fingerprint_section(state)
        assert "nginx/1.18" in result
        assert "Flask" in result
        assert "Content-Security-Policy" in result

    def test_empty_fingerprint(self):
        """No tech data should produce an empty string."""
        state = ScanState(target="test.com", base_url="https://test.com")
        assert _build_tech_fingerprint_section(state) == ""


class TestBuildAttackSurfaceTable:
    def test_empty_surface_returns_empty(self):
        """No attack surface entries should return empty string."""
        state = ScanState(target="test.com", base_url="https://test.com")
        assert _build_attack_surface_table(state) == ""

    def test_table_has_header_and_rows(self):
        """With entries, the table should have a header row and data rows."""
        ep = Endpoint(
            url="https://test.com/api/users",
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        entry = AttackSurfaceEntry(
            endpoint=ep,
            category=EndpointCategory.REST_API,
            risk_level=RiskLevel.HIGH,
            priority_rank=1,
            rationale="test",
            vuln_indicators=[
                VulnIndicator(
                    pattern=VulnPattern.BOLA_IDOR,
                    confidence=RiskLevel.HIGH,
                    evidence="test",
                    description="test",
                ),
            ],
        )
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            attack_surface=[entry],
        )
        table = _build_attack_surface_table(state)
        assert "| #" in table
        assert "bola_idor" in table
        assert "/api/users" in table


class TestBuildFallbackReport:
    def test_includes_header(self):
        """Fallback should contain target and duration."""
        state = ScanState(target="test.com", base_url="https://test.com")
        report = _build_fallback_report(state, duration=12.5)
        assert "test.com" in report
        assert "12.5s" in report
        assert "AI report synthesis failed" in report

    def test_includes_critical_findings(self):
        """Critical findings should appear in the fallback."""
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            findings=[
                Finding(
                    agent_name="vuln_analyzer",
                    finding_type="info_disclosure",
                    title="Debug endpoint exposed",
                    detail="Werkzeug debugger at /console",
                    severity=RiskLevel.CRITICAL,
                    evidence="HTTP 200",
                ),
                Finding(
                    agent_name="test",
                    finding_type="test",
                    title="Minor issue",
                    detail="Not important",
                    severity=RiskLevel.LOW,
                ),
            ],
        )
        report = _build_fallback_report(state, duration=5.0)
        assert "Debug endpoint exposed" in report
        assert "Evidence: HTTP 200" in report
        assert "Minor issue" not in report

    def test_includes_tech_fingerprint(self):
        """Fallback should include tech fingerprint if available."""
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            tech_fingerprint=TechFingerprint(server="Apache/2.4"),
        )
        report = _build_fallback_report(state, duration=3.0)
        assert "Apache/2.4" in report
        assert "Target Profile" in report

    def test_includes_attack_surface_table(self):
        """Fallback should include the attack surface table if entries exist."""
        ep = Endpoint(
            url="https://test.com/admin",
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        entry = AttackSurfaceEntry(
            endpoint=ep,
            category=EndpointCategory.ADMIN_ENDPOINT,
            risk_level=RiskLevel.CRITICAL,
            priority_rank=1,
            rationale="admin panel",
        )
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            attack_surface=[entry],
        )
        report = _build_fallback_report(state, duration=2.0)
        assert "Attack Surface Map" in report
        assert "/admin" in report


class TestBuildReportContext:
    def test_includes_cwe_block(self):
        """Report context sent to the LLM should include the CWE reference map."""
        state = ScanState(target="test.com", base_url="https://test.com")
        context = _build_report_context(state, duration=10.0)
        assert "CWE REFERENCE MAP" in context
        assert "CWE-639" in context

    def test_scan_metadata_present(self):
        """Basic scan metadata should appear at the top of the context."""
        state = ScanState(
            target="vulnbank.org",
            base_url="https://vulnbank.org",
            agents_completed=["passive_recon", "web_crawler"],
        )
        context = _build_report_context(state, duration=42.0)
        assert "vulnbank.org" in context
        assert "42.0s" in context
        assert "passive_recon" in context
