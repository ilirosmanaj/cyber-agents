"""Tests for report helper functions — formatting, scaling, and fallback."""

from __future__ import annotations

from src.ghost_hunter.models import (
    AttackSurfaceEntry,
    DiscoverySource,
    Endpoint,
    EndpointCategory,
    Finding,
    RiskLevel,
    ScanInsight,
    ScanState,
    ScanStrategy,
    TechFingerprint,
    VulnIndicator,
    VulnPattern,
)
from src.ghost_hunter.report import (
    CWE_REFERENCES,
    _BASE_REPORT_TOKENS,
    _MAX_REPORT_TOKENS,
    _TOKENS_PER_FINDING,
    _append_missing_findings,
    _augment_report,
    _build_attack_surface_context,
    _build_attack_surface_table,
    _build_chain_analysis,
    _build_cwe_reference_block,
    _build_enriched_findings,
    _build_fallback_report,
    _build_report_context,
    _build_section_instructions,
    _build_strategy_section,
    _build_tech_fingerprint_section,
    _build_vuln_pattern_summary,
    _compute_max_tokens,
    _ensure_base_url,
    _format_finding_enriched,
    _is_login_token_false_positive,
    _lookup_cwe_for_finding,
    _lookup_tests_for_finding,
    _splice_deterministic_table,
)


def _minimal_state() -> ScanState:
    """Create a minimal ScanState for tests that need one."""
    return ScanState(target="test.com", base_url="https://test.com")


def _state_with_attack_surface() -> ScanState:
    """State with an endpoint and matching attack surface entry."""
    ep = Endpoint(
        url="https://test.com/api/users",
        method="GET",
        discovered_by=DiscoverySource.CRAWL,
        parameters=["id"],
        requires_auth=True,
    )
    entry = AttackSurfaceEntry(
        endpoint=ep,
        category=EndpointCategory.REST_API,
        risk_level=RiskLevel.HIGH,
        priority_rank=1,
        rationale="IDOR candidate",
        suggested_tests=["curl -H 'Auth: tok' https://test.com/api/users?id=2"],
        vuln_indicators=[
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="id param in path",
                description="test",
            ),
        ],
    )
    state = ScanState(
        target="test.com",
        base_url="https://test.com",
        attack_surface=[entry],
    )
    state.add_endpoint(ep)
    return state


# ---- TestComputeMaxTokens ----


class TestComputeMaxTokens:
    def test_small_scan_uses_base(self):
        assert _compute_max_tokens(0) == _BASE_REPORT_TOKENS

    def test_scales_with_findings(self):
        expected = _BASE_REPORT_TOKENS + 10 * _TOKENS_PER_FINDING
        assert _compute_max_tokens(10) == expected

    def test_caps_at_maximum(self):
        assert _compute_max_tokens(500) == _MAX_REPORT_TOKENS

    def test_boundary_at_cap(self):
        exact = (_MAX_REPORT_TOKENS - _BASE_REPORT_TOKENS) // _TOKENS_PER_FINDING
        assert _compute_max_tokens(exact) <= _MAX_REPORT_TOKENS
        assert _compute_max_tokens(exact + 1) == _MAX_REPORT_TOKENS


# ---- TestFormatFindingEnriched ----


class TestFormatFindingEnriched:
    def test_cwe_attached(self):
        """vuln_bola_idor finding_type should produce CWE-639 tag."""
        f = Finding(
            agent_name="test",
            finding_type="vuln_bola_idor",
            title="BOLA_IDOR — GET https://test.com/api/users",
            detail="IDOR on user id",
            severity=RiskLevel.HIGH,
        )
        result = _format_finding_enriched(f)
        assert "CWE-639" in result

    def test_suggested_tests_included(self):
        """When the attack surface has tests, they show up in the output."""
        state = _state_with_attack_surface()
        f = Finding(
            agent_name="test",
            finding_type="vuln_bola_idor",
            title="BOLA_IDOR — GET https://test.com/api/users",
            detail="IDOR",
            severity=RiskLevel.HIGH,
        )
        result = _format_finding_enriched(f, state)
        assert "Suggested tests:" in result
        assert "curl" in result

    def test_verification_status_shown(self):
        """Verification status + note should appear when set."""
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="Test finding",
            detail="detail",
            verification_status="consistent",
            verification_note="confirmed by verifier",
        )
        result = _format_finding_enriched(f)
        assert "Verification: consistent" in result
        assert "confirmed by verifier" in result

    def test_validated_tag(self):
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="Confirmed",
            detail="d",
            validated=True,
        )
        assert "[VALIDATED]" in _format_finding_enriched(f)

    def test_refuted_tag(self):
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="False pos",
            detail="d",
            validated=False,
        )
        assert "[REFUTED]" in _format_finding_enriched(f)


# ---- TestLookupCweForFinding ----


class TestLookupCweForFinding:
    def test_direct_finding_type_match(self):
        """finding_type 'bola_idor' maps to CWE-639."""
        f = Finding(
            agent_name="test",
            finding_type="bola_idor",
            title="test",
            detail="d",
        )
        assert "CWE-639" in _lookup_cwe_for_finding(f)

    def test_vuln_prefix_stripped(self):
        """finding_type 'vuln_ssrf' maps via stripping 'vuln_' prefix."""
        f = Finding(
            agent_name="test",
            finding_type="vuln_ssrf",
            title="test",
            detail="d",
        )
        assert "CWE-918" in _lookup_cwe_for_finding(f)

    def test_title_keyword_fallback(self):
        """Falls back to keyword matching in title."""
        f = Finding(
            agent_name="test",
            finding_type="unknown_type",
            title="Detected race condition in endpoint",
            detail="d",
        )
        assert "CWE-362" in _lookup_cwe_for_finding(f)

    def test_no_match_returns_empty(self):
        """Unrecognized type and title returns empty string."""
        f = Finding(
            agent_name="test",
            finding_type="something_unknown",
            title="Generic finding title",
            detail="d",
        )
        assert _lookup_cwe_for_finding(f) == ""


# ---- TestLookupTestsForFinding ----


class TestLookupTestsForFinding:
    def test_matching_endpoint(self):
        """Finding with a matching attack surface entry gets its tests."""
        state = _state_with_attack_surface()
        f = Finding(
            agent_name="test",
            finding_type="vuln_bola_idor",
            title="BOLA_IDOR — GET https://test.com/api/users",
            detail="d",
        )
        tests = _lookup_tests_for_finding(f, state)
        assert len(tests) == 1
        assert "curl" in tests[0]

    def test_no_match_returns_empty(self):
        """No attack surface match means no tests."""
        state = _minimal_state()
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="TEST — GET https://test.com/nonexistent",
            detail="d",
        )
        assert _lookup_tests_for_finding(f, state) == []

    def test_no_dash_separator_returns_empty(self):
        """Title without ' — ' can't be parsed, so no tests."""
        state = _state_with_attack_surface()
        f = Finding(
            agent_name="test",
            finding_type="test",
            title="Some general finding",
            detail="d",
        )
        assert _lookup_tests_for_finding(f, state) == []


# ---- TestBuildCweReferenceBlock ----


class TestBuildCweReferenceBlock:
    def test_contains_all_patterns(self):
        block = _build_cwe_reference_block()
        for pattern in VulnPattern:
            assert pattern.value in CWE_REFERENCES, f"Missing CWE for {pattern.value}"
            assert pattern.value in block

    def test_format_is_arrow_separated(self):
        block = _build_cwe_reference_block()
        for line in block.strip().split("\n")[1:]:
            assert "→" in line


# ---- TestBuildTechFingerprint ----


class TestBuildTechFingerprint:
    def test_full_fingerprint(self):
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
        state = ScanState(target="test.com", base_url="https://test.com")
        assert _build_tech_fingerprint_section(state) == ""


# ---- TestBuildAttackSurfaceTable ----


class TestBuildAttackSurfaceTable:
    def test_empty_surface_returns_empty(self):
        state = ScanState(target="test.com", base_url="https://test.com")
        assert _build_attack_surface_table(state) == ""

    def test_table_has_header_and_rows(self):
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

    def test_deduplicates_indicators(self):
        """Duplicate vuln patterns in a single entry are deduplicated."""
        ep = Endpoint(
            url="https://test.com/api/accounts",
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        entry = AttackSurfaceEntry(
            endpoint=ep,
            category=EndpointCategory.REST_API,
            risk_level=RiskLevel.CRITICAL,
            priority_rank=1,
            rationale="test",
            vuln_indicators=[
                VulnIndicator(
                    pattern=VulnPattern.BOLA_IDOR,
                    confidence=RiskLevel.HIGH,
                    evidence="e1",
                    description="d1",
                ),
                VulnIndicator(
                    pattern=VulnPattern.BOLA_IDOR,
                    confidence=RiskLevel.CRITICAL,
                    evidence="e2",
                    description="d2",
                ),
                VulnIndicator(
                    pattern=VulnPattern.JWT_WEAKNESS,
                    confidence=RiskLevel.HIGH,
                    evidence="e3",
                    description="d3",
                ),
                VulnIndicator(
                    pattern=VulnPattern.BOLA_IDOR,
                    confidence=RiskLevel.HIGH,
                    evidence="e4",
                    description="d4",
                ),
            ],
        )
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            attack_surface=[entry],
        )
        table = _build_attack_surface_table(state)
        # Should have bola_idor once, then jwt_weakness — not bola_idor three times
        assert "bola_idor, jwt_weakness" in table
        assert "bola_idor, bola_idor" not in table

    def test_filters_malformed_urls(self):
        """URLs with malformed characters like commas or quotes are excluded."""
        good_ep = Endpoint(
            url="https://test.com/api/valid",
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        bad_ep = Endpoint(
            url='https://test.com/g,"',
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        entries = [
            AttackSurfaceEntry(
                endpoint=bad_ep,
                category=EndpointCategory.UNKNOWN,
                risk_level=RiskLevel.INFO,
                priority_rank=1,
                rationale="artifact",
            ),
            AttackSurfaceEntry(
                endpoint=good_ep,
                category=EndpointCategory.REST_API,
                risk_level=RiskLevel.HIGH,
                priority_rank=2,
                rationale="valid",
            ),
        ]
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            attack_surface=entries,
        )
        table = _build_attack_surface_table(state)
        assert "/api/valid" in table
        assert '/g,"' not in table
        # Re-ranked: the valid entry should be rank 1
        assert "| 1 " in table

    def test_filters_aws_metadata_paths(self):
        """AWS metadata paths on target host are excluded."""
        real_ep = Endpoint(
            url="https://test.com/api/users",
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        meta_ep = Endpoint(
            url="https://test.com/latest/meta-data/iam/security-credentials/",
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        entries = [
            AttackSurfaceEntry(
                endpoint=real_ep,
                category=EndpointCategory.REST_API,
                risk_level=RiskLevel.HIGH,
                priority_rank=1,
                rationale="real",
            ),
            AttackSurfaceEntry(
                endpoint=meta_ep,
                category=EndpointCategory.HEALTH_CHECK,
                risk_level=RiskLevel.INFO,
                priority_rank=2,
                rationale="aws",
            ),
        ]
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            attack_surface=entries,
        )
        table = _build_attack_surface_table(state)
        assert "/api/users" in table
        assert "meta-data" not in table

    def test_all_entries_filtered_returns_empty(self):
        """If all entries are filtered out, returns empty string."""
        bad_ep = Endpoint(
            url='https://test.com/g,"',
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        entry = AttackSurfaceEntry(
            endpoint=bad_ep,
            category=EndpointCategory.UNKNOWN,
            risk_level=RiskLevel.INFO,
            priority_rank=1,
            rationale="bad",
        )
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            attack_surface=[entry],
        )
        assert _build_attack_surface_table(state) == ""


# ---- TestBuildEnrichedFindings ----


class TestBuildEnrichedFindings:
    def test_groups_by_severity(self):
        """Each severity level gets its own header block."""
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            findings=[
                Finding(
                    agent_name="test",
                    finding_type="test",
                    title="Critical issue",
                    detail="d",
                    severity=RiskLevel.CRITICAL,
                ),
                Finding(
                    agent_name="test",
                    finding_type="test",
                    title="High issue",
                    detail="d",
                    severity=RiskLevel.HIGH,
                ),
                Finding(
                    agent_name="test",
                    finding_type="test",
                    title="Low issue",
                    detail="d",
                    severity=RiskLevel.LOW,
                ),
            ],
        )
        result = _build_enriched_findings(state)
        assert "FINDINGS — CRITICAL (1 total) — INCLUDE ALL IN REPORT" in result
        assert "FINDINGS — HIGH (1 total) — INCLUDE ALL IN REPORT" in result
        assert "FINDINGS — LOW (1 total) — INCLUDE ALL IN REPORT" in result

    def test_severity_order(self):
        """CRITICAL section appears before HIGH, which appears before LOW."""
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            findings=[
                Finding(
                    agent_name="test",
                    finding_type="t",
                    title="Low",
                    detail="d",
                    severity=RiskLevel.LOW,
                ),
                Finding(
                    agent_name="test",
                    finding_type="t",
                    title="Critical",
                    detail="d",
                    severity=RiskLevel.CRITICAL,
                ),
            ],
        )
        result = _build_enriched_findings(state)
        crit_pos = result.find("CRITICAL")
        low_pos = result.find("LOW")
        assert crit_pos < low_pos

    def test_empty_findings(self):
        state = _minimal_state()
        assert _build_enriched_findings(state) == ""


# ---- TestBuildChainAnalysis ----


class TestBuildChainAnalysis:
    def test_explicit_chain(self):
        """Two indicators with the same chain_id are grouped together."""
        state = _minimal_state()
        state.vuln_indicators = {
            "GET https://test.com/api/upload": [
                VulnIndicator(
                    pattern=VulnPattern.FILE_UPLOAD,
                    confidence=RiskLevel.HIGH,
                    evidence="upload endpoint",
                    description="d",
                    chain_id="abc123",
                ),
            ],
            "GET https://test.com/api/fetch": [
                VulnIndicator(
                    pattern=VulnPattern.SSRF,
                    confidence=RiskLevel.HIGH,
                    evidence="ssrf via fetch",
                    description="d",
                    chain_id="abc123",
                ),
            ],
        }
        result = _build_chain_analysis(state)
        assert "VULNERABILITY CHAINS:" in result
        assert "Chain abc123" in result
        assert "file_upload" in result
        assert "ssrf" in result

    def test_implicit_chain(self):
        """ssrf + file_upload on one endpoint = implicit chain."""
        state = _minimal_state()
        state.vuln_indicators = {
            "POST https://test.com/api/data": [
                VulnIndicator(
                    pattern=VulnPattern.SSRF,
                    confidence=RiskLevel.HIGH,
                    evidence="ssrf evidence",
                    description="d",
                ),
                VulnIndicator(
                    pattern=VulnPattern.FILE_UPLOAD,
                    confidence=RiskLevel.HIGH,
                    evidence="upload evidence",
                    description="d",
                ),
            ],
        }
        result = _build_chain_analysis(state)
        assert "Implicit chain at POST https://test.com/api/data" in result
        assert "ssrf" in result
        assert "file_upload" in result

    def test_empty_returns_empty(self):
        state = _minimal_state()
        assert _build_chain_analysis(state) == ""

    def test_suppressed_excluded(self):
        """Suppressed indicators shouldn't produce chains."""
        state = _minimal_state()
        state.vuln_indicators = {
            "GET https://test.com/x": [
                VulnIndicator(
                    pattern=VulnPattern.SSRF,
                    confidence=RiskLevel.HIGH,
                    evidence="e",
                    description="d",
                    chain_id="ch1",
                    suppressed=True,
                ),
            ],
        }
        assert _build_chain_analysis(state) == ""


# ---- TestBuildStrategySection ----


class TestBuildStrategySection:
    def test_with_strategy(self):
        state = _minimal_state()
        state.scan_strategy = ScanStrategy(
            focus_areas=["auth_flows", "graphql"],
            tech_hypotheses=["Flask", "JWT"],
            scan_depth="deep",
            priority_patterns=["bola_idor"],
        )
        result = _build_strategy_section(state)
        assert "SCAN STRATEGY" in result
        assert "auth_flows" in result
        assert "Flask" in result
        assert "deep" in result

    def test_with_insights(self):
        state = _minimal_state()
        state.scan_insights = [
            ScanInsight(
                phase="recon",
                summary="Found API docs.",
                key_signals=["openapi"],
            ),
        ]
        result = _build_strategy_section(state)
        assert "SCAN INSIGHTS" in result
        assert "Found API docs" in result

    def test_empty_state(self):
        state = _minimal_state()
        assert _build_strategy_section(state) == ""


# ---- TestBuildSectionInstructions ----


class TestBuildSectionInstructions:
    def test_includes_finding_count(self):
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            findings=[
                Finding(
                    agent_name="t",
                    finding_type="t",
                    title="c",
                    detail="d",
                    severity=RiskLevel.CRITICAL,
                ),
                Finding(
                    agent_name="t",
                    finding_type="t",
                    title="h",
                    detail="d",
                    severity=RiskLevel.HIGH,
                ),
            ],
        )
        result = _build_section_instructions(state)
        assert "ALL 2 critical+high findings" in result

    def test_methodology_mentions_planner(self):
        state = _minimal_state()
        state.scan_strategy = ScanStrategy(focus_areas=["auth"])
        result = _build_section_instructions(state)
        assert "planner strategy" in result

    def test_medium_low_section_included(self):
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            findings=[
                Finding(
                    agent_name="t",
                    finding_type="t",
                    title="m",
                    detail="d",
                    severity=RiskLevel.MEDIUM,
                ),
            ],
        )
        result = _build_section_instructions(state)
        assert "Medium & Low Findings" in result


# ---- TestBuildAttackSurfaceContext ----


class TestBuildAttackSurfaceContext:
    def test_returns_empty_for_no_surface(self):
        state = _minimal_state()
        assert _build_attack_surface_context(state) == ""

    def test_limits_entries(self):
        ep = Endpoint(
            url="https://test.com/a",
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        entries = [
            AttackSurfaceEntry(
                endpoint=ep,
                category=EndpointCategory.REST_API,
                risk_level=RiskLevel.HIGH,
                priority_rank=i,
                rationale="test",
            )
            for i in range(1, 6)
        ]
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            attack_surface=entries,
        )
        result = _build_attack_surface_context(state, limit=3)
        assert "top 3 of 5" in result
        # Only ranks 1-3 should appear
        assert "#1 " in result
        assert "#3 " in result
        assert "#4 " not in result


# ---- TestBuildVulnPatternSummary ----


class TestBuildVulnPatternSummary:
    def test_counts_active_indicators(self):
        state = _minimal_state()
        state.vuln_indicators = {
            "ep1": [
                VulnIndicator(
                    pattern=VulnPattern.BOLA_IDOR,
                    confidence=RiskLevel.HIGH,
                    evidence="e",
                    description="d",
                ),
                VulnIndicator(
                    pattern=VulnPattern.BOLA_IDOR,
                    confidence=RiskLevel.HIGH,
                    evidence="e2",
                    description="d",
                    suppressed=True,
                ),
            ],
        }
        result = _build_vuln_pattern_summary(state)
        assert "Total active indicators: 1" in result
        assert "Suppressed (false positives): 1" in result


# ---- TestSpliceDeterministicTable ----


class TestSpliceDeterministicTable:
    def test_replaces_llm_table(self):
        """Whatever the LLM wrote for the table gets swapped out."""
        state = _state_with_attack_surface()
        llm_report = (
            "## Executive Summary\nSummary text.\n\n"
            "## Attack Surface Map\n\n"
            "| Bad | Table |\n|-----|-------|\n| x | y |\n\n"
            "## Recommendations\nFix things."
        )
        result = _splice_deterministic_table(llm_report, state)
        assert "| # | Risk | Method | URL | Category | Indicators |" in result
        assert "| Bad | Table |" not in result
        assert "## Recommendations" in result

    def test_inserts_when_missing(self):
        """Missing section header? Insert before Recommendations."""
        state = _state_with_attack_surface()
        report = "## Executive Summary\nSummary.\n\n## Recommendations\nFix."
        result = _splice_deterministic_table(report, state)
        assert "## Attack Surface Map" in result
        assert "| # | Risk | Method |" in result
        # Recommendations still present
        assert "## Recommendations" in result

    def test_appends_when_no_anchor(self):
        """If no anchor sections exist, appends at end."""
        state = _state_with_attack_surface()
        report = "## Executive Summary\nJust a summary."
        result = _splice_deterministic_table(report, state)
        assert "## Attack Surface Map" in result
        assert result.index("## Attack Surface Map") > report.index("## Executive Summary")

    def test_noop_without_attack_surface(self):
        """No attack surface entries means report unchanged."""
        state = _minimal_state()
        report = "## Report\nContent."
        assert _splice_deterministic_table(report, state) == report


# ---- TestAugmentReport ----


class TestAugmentReport:
    def test_deterministic_table_spliced(self):
        """Old LLM table should be gone, deterministic one in its place."""
        state = _state_with_attack_surface()
        report = (
            "## Executive Summary\nhttps://test.com\n\n"
            "## Attack Surface Map\n\n| Old | Table |\n\n"
            "## Recommendations\nDone."
        )
        result = _augment_report(report, state)
        assert "| # | Risk | Method |" in result
        assert "| Old | Table |" not in result

    def test_missing_findings_appended(self):
        """An SSRF finding the LLM skipped should appear in Additional Findings."""
        state = _minimal_state()
        state.findings = [
            Finding(
                agent_name="test",
                finding_type="vuln_ssrf",
                title="SSRF — GET https://test.com/api/fetch",
                detail="SSRF on fetch endpoint",
                severity=RiskLevel.CRITICAL,
                evidence="HTTP 200 with internal IP",
            ),
        ]
        report = (
            "## Executive Summary\nhttps://test.com\n\n"
            "## Recommendations\nFix things."
        )
        result = _augment_report(report, state)
        assert "Additional Critical & High Findings" in result
        assert "ssrf" in result

    def test_present_findings_not_duplicated(self):
        """Already-mentioned finding shouldn't be appended twice."""
        state = _minimal_state()
        state.findings = [
            Finding(
                agent_name="test",
                finding_type="vuln_ssrf",
                title="SSRF — GET https://test.com/api/fetch",
                detail="SSRF on fetch endpoint",
                severity=RiskLevel.CRITICAL,
            ),
        ]
        report = (
            "## Executive Summary\nhttps://test.com\n\n"
            "## Findings\nSSRF — GET https://test.com/api/fetch\n\n"
            "## Recommendations\nFix."
        )
        result = _augment_report(report, state)
        assert "Additional Critical & High Findings" not in result

    def test_finding_url_in_table_still_appended(self):
        """URL appearing only in the table doesn't count as 'mentioned'."""
        state = _state_with_attack_surface()
        state.findings = [
            Finding(
                agent_name="test",
                finding_type="vuln_bola_idor",
                title="BOLA_IDOR — GET https://test.com/api/users",
                detail="IDOR on user endpoint",
                severity=RiskLevel.HIGH,
            ),
        ]
        report = (
            "## Executive Summary\nhttps://test.com summary\n\n"
            "## Critical & High Findings\n\nNo findings detailed.\n\n"
            "## Attack Surface Map\n\n"
            "| # | Risk | Method | URL | Category | Indicators |\n"
            "|----|------|--------|-----|----------|------------|\n"
            "| 1 | HIGH | GET | https://test.com/api/users | rest_api | bola_idor |\n\n"
            "## Recommendations\nFix things."
        )
        result = _augment_report(report, state)
        assert "Additional Critical & High Findings" in result
        assert "bola idor" in result

    def test_appended_findings_include_tests(self):
        """Appended findings should carry their curl commands."""
        state = _state_with_attack_surface()
        state.findings = [
            Finding(
                agent_name="test",
                finding_type="vuln_bola_idor",
                title="BOLA_IDOR — GET https://test.com/api/users",
                detail="IDOR",
                severity=RiskLevel.HIGH,
            ),
        ]
        report = (
            "## Executive Summary\nhttps://test.com\n\n"
            "## Attack Surface Map\n\n| table |\n\n"
            "## Recommendations\nFix."
        )
        result = _augment_report(report, state)
        assert "Test:" in result
        assert "curl" in result

    def test_groups_by_endpoint(self):
        """Two BOLA findings on /api/accounts shouldn't produce two entries."""
        ep = Endpoint(
            url="https://test.com/api/accounts",
            method="GET",
            discovered_by=DiscoverySource.CRAWL,
        )
        state = _minimal_state()
        state.add_endpoint(ep)
        state.findings = [
            Finding(
                agent_name="test",
                finding_type="vuln_bola_idor",
                title="BOLA_IDOR — GET https://test.com/api/accounts",
                detail="Path param BOLA",
                severity=RiskLevel.HIGH,
                evidence="path contains object ID",
            ),
            Finding(
                agent_name="test",
                finding_type="vuln_bola_idor",
                title="BOLA_IDOR_QUERY — GET https://test.com/api/accounts",
                detail="Query param BOLA",
                severity=RiskLevel.HIGH,
                evidence="query param id",
            ),
        ]
        report = (
            "## Executive Summary\nhttps://test.com\n\n"
            "## Attack Surface Map\n\n| table |\n\n"
            "## Recommendations\nFix."
        )
        result = _augment_report(report, state)
        # Should have one grouped entry for GET https://test.com/api/accounts
        assert result.count("GET https://test.com/api/accounts") == 1
        # Both evidence items should be combined
        assert "path contains object ID" in result
        assert "query param id" in result

    def test_placed_before_attack_surface(self):
        """Additional findings go before the table, not after."""
        state = _minimal_state()
        state.findings = [
            Finding(
                agent_name="test",
                finding_type="vuln_ssrf",
                title="SSRF — GET https://test.com/api/fetch",
                detail="SSRF",
                severity=RiskLevel.CRITICAL,
            ),
        ]
        report = (
            "## Executive Summary\nhttps://test.com\n\n"
            "## Attack Surface Map\n\ntable here\n\n"
            "## Recommendations\nFix."
        )
        result = _augment_report(report, state)
        additional_pos = result.find("Additional Critical & High Findings")
        surface_pos = result.find("## Attack Surface Map")
        assert additional_pos < surface_pos

    def test_login_token_false_positive_filtered(self):
        """Token in /login response is normal — shouldn't be appended."""
        state = _minimal_state()
        state.findings = [
            Finding(
                agent_name="test",
                finding_type="excessive_data_exposure",
                title="EXCESSIVE_DATA_EXPOSURE — POST https://test.com/login",
                detail="Token in response",
                severity=RiskLevel.HIGH,
            ),
        ]
        report = (
            "## Executive Summary\nhttps://test.com\n\n"
            "## Recommendations\nFix."
        )
        result = _augment_report(report, state)
        assert "Additional Critical & High Findings" not in result

    def test_base_url_added_when_missing(self):
        """Report without the target URL gets it injected."""
        state = _minimal_state()
        report = "## Executive Summary\nSome summary.\n\n## Recommendations\nFix."
        result = _augment_report(report, state)
        assert "**Base URL:** https://test.com" in result

    def test_base_url_not_added_when_present(self):
        """Don't double-add the URL if it's already there."""
        state = _minimal_state()
        report = "## Executive Summary\nTarget: https://test.com\n"
        result = _augment_report(report, state)
        # Should appear exactly once (the original)
        assert result.count("https://test.com") == 1


# ---- TestLoginTokenFalsePositive ----


class TestLoginTokenFalsePositive:
    def test_login_endpoint_flagged(self):
        f = Finding(
            agent_name="test",
            finding_type="excessive_data_exposure",
            title="EXCESSIVE_DATA_EXPOSURE — POST https://test.com/login",
            detail="d",
        )
        assert _is_login_token_false_positive(f) is True

    def test_auth_endpoint_flagged(self):
        f = Finding(
            agent_name="test",
            finding_type="vuln_excessive_data_exposure",
            title="EXCESSIVE_DATA_EXPOSURE — POST https://test.com/api/auth/token",
            detail="d",
        )
        assert _is_login_token_false_positive(f) is True

    def test_non_auth_endpoint_not_flagged(self):
        f = Finding(
            agent_name="test",
            finding_type="excessive_data_exposure",
            title="EXCESSIVE_DATA_EXPOSURE — GET https://test.com/api/users",
            detail="d",
        )
        assert _is_login_token_false_positive(f) is False

    def test_non_excessive_data_type_not_flagged(self):
        f = Finding(
            agent_name="test",
            finding_type="vuln_bola_idor",
            title="BOLA_IDOR — POST https://test.com/login",
            detail="d",
        )
        assert _is_login_token_false_positive(f) is False


# ---- TestEnsureBaseUrl ----


class TestEnsureBaseUrl:
    def test_inserts_after_exec_summary(self):
        state = _minimal_state()
        report = "## Executive Summary\nSummary text.\n\nMore content."
        result = _ensure_base_url(report, state)
        assert "**Base URL:** https://test.com" in result
        # Should be after header line
        assert result.index("**Base URL:**") > result.index("## Executive Summary")

    def test_prepends_when_no_exec_summary(self):
        state = _minimal_state()
        report = "# Some Report\nContent."
        result = _ensure_base_url(report, state)
        assert result.startswith("**Base URL:** https://test.com")


# ---- TestBuildFallbackReport ----


class TestBuildFallbackReport:
    def test_includes_header(self):
        state = ScanState(target="test.com", base_url="https://test.com")
        report = _build_fallback_report(state, duration=12.5)
        assert "test.com" in report
        assert "12.5s" in report
        assert "AI report synthesis failed" in report

    def test_includes_critical_findings_with_cwe(self):
        """Fallback should show CWE next to critical findings."""
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
        assert "CWE-200" in report

    def test_includes_suggested_tests(self):
        """curl commands from the attack surface make it into the fallback."""
        state = _state_with_attack_surface()
        state.findings = [
            Finding(
                agent_name="test",
                finding_type="vuln_bola_idor",
                title="BOLA_IDOR — GET https://test.com/api/users",
                detail="IDOR",
                severity=RiskLevel.HIGH,
            ),
        ]
        report = _build_fallback_report(state, duration=3.0)
        assert "Tests:" in report
        assert "curl" in report

    def test_includes_tech_fingerprint(self):
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            tech_fingerprint=TechFingerprint(server="Apache/2.4"),
        )
        report = _build_fallback_report(state, duration=3.0)
        assert "Apache/2.4" in report
        assert "Target Profile" in report

    def test_includes_attack_surface_table(self):
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


# ---- TestBuildReportContext ----


class TestBuildReportContext:
    def test_includes_cwe_block(self):
        state = ScanState(target="test.com", base_url="https://test.com")
        context = _build_report_context(state, duration=10.0)
        assert "CWE REFERENCE MAP" in context
        assert "CWE-639" in context

    def test_scan_metadata_present(self):
        state = ScanState(
            target="vulnbank.org",
            base_url="https://vulnbank.org",
            agents_completed=["passive_recon", "web_crawler"],
        )
        context = _build_report_context(state, duration=42.0)
        assert "vulnbank.org" in context
        assert "42.0s" in context
        assert "passive_recon" in context

    def test_includes_finding_example(self):
        """The few-shot example should be in the prompt context."""
        state = _minimal_state()
        context = _build_report_context(state, duration=1.0)
        assert "EXAMPLE FINDING FORMAT" in context

    def test_includes_enriched_findings(self):
        """Severity headers like 'FINDINGS — CRITICAL' should appear."""
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            findings=[
                Finding(
                    agent_name="test",
                    finding_type="test",
                    title="Critical issue",
                    detail="d",
                    severity=RiskLevel.CRITICAL,
                ),
            ],
        )
        context = _build_report_context(state, duration=1.0)
        assert "FINDINGS — CRITICAL (1 total) — INCLUDE ALL IN REPORT" in context

    def test_includes_section_instructions(self):
        """Section list for the LLM should be at the end."""
        state = _minimal_state()
        context = _build_report_context(state, duration=1.0)
        assert "REPORT SECTIONS (generate in this order):" in context

    def test_includes_strategy_when_present(self):
        """Planner strategy makes it into the context if set."""
        state = _minimal_state()
        state.scan_strategy = ScanStrategy(
            focus_areas=["graphql"],
            scan_depth="deep",
        )
        context = _build_report_context(state, duration=1.0)
        assert "SCAN STRATEGY" in context
        assert "graphql" in context

    def test_attack_surface_includes_indicator_evidence(self):
        ep = Endpoint(
            url="https://test.com/api/chat",
            method="POST",
            discovered_by=DiscoverySource.CRAWL,
            parameters=["message"],
            requires_auth=True,
            response_fields=["response"],
        )
        indicator = VulnIndicator(
            pattern=VulnPattern.PROMPT_INJECTION,
            confidence=RiskLevel.HIGH,
            evidence="AI path: /api/chat; input params: message",
            description="AI endpoint accepts free-text input",
        )
        entry = AttackSurfaceEntry(
            endpoint=ep,
            category=EndpointCategory.REST_API,
            risk_level=RiskLevel.HIGH,
            priority_rank=1,
            rationale="AI endpoint",
            vuln_indicators=[indicator],
        )
        state = ScanState(
            target="test.com",
            base_url="https://test.com",
            attack_surface=[entry],
        )
        context = _build_report_context(state, duration=5.0)
        assert "AI path: /api/chat; input params: message" in context
        assert "AI endpoint accepts free-text input" in context
        assert "Params: message" in context
        assert "Auth: requires_auth" in context
        assert "Response fields: response" in context
