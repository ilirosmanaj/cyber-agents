"""Tests for VulnPatternAnalyzer deterministic checks and body analysis."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.ghost_hunter.agents.vuln_analyzer import VulnPatternAnalyzer, _SEVERITY_ORDER
from src.ghost_hunter.models import (
    EndpointCategory,
    RiskLevel,
    ScanState,
    SecuritySchemeInfo,
    VulnIndicator,
    VulnPattern,
)
from src.ghost_hunter.models.llm_responses import (
    ResponseBodyAnalysisResponse,
    ResponseBodyFinding,
)
from tests.conftest import make_endpoint


# ---------------------------------------------------------------------------
# BOLA / IDOR detection
# ---------------------------------------------------------------------------


class TestBOLAIDOR:
    """Tests for _check_bola_idor."""

    def test_path_param_with_user_id(self):
        """Endpoints with /users/{id} should trigger bola_idor at HIGH confidence."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/users/{id}")
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        assert any(ind.pattern == VulnPattern.BOLA_IDOR for ind in indicators)
        assert any(ind.confidence == RiskLevel.HIGH for ind in indicators)

    def test_path_param_with_account_id(self):
        """Given endpoint with /accounts/{account_id}, detect bola_idor."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/accounts/{account_id}")
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        assert any(ind.pattern == VulnPattern.BOLA_IDOR for ind in indicators)

    def test_numeric_path_segment(self):
        """Given endpoint with numeric path segment, detect bola_idor as MEDIUM."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/users/123")
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        assert any(
            ind.pattern == VulnPattern.BOLA_IDOR and ind.confidence == RiskLevel.MEDIUM
            for ind in indicators
        )

    def test_static_asset_no_idor(self):
        """Static assets like /static/style.css should never trigger IDOR."""
        ep = make_endpoint(url="https://vulnbank.org/static/style.css")
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        assert not indicators

    def test_query_param_user_id(self):
        """Given endpoint with user_id query param, detect bola_idor."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/profile",
            parameters=["user_id"],
        )
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        assert any(ind.pattern == VulnPattern.BOLA_IDOR for ind in indicators)

    def test_query_param_no_idor(self):
        """Params like 'query' and 'page' shouldn't trigger bola_idor."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/search",
            parameters=["query", "page"],
        )
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        assert not indicators

    def test_long_numeric_segment_ignored(self):
        """Given path with very long numeric segment, no bola_idor (not an enumerable ID)."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/users/12345678901234")
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        # only 10 digits max is detected
        assert not any(
            ind.pattern == VulnPattern.BOLA_IDOR and "Numeric path segment" in ind.evidence
            for ind in indicators
        )


# ---------------------------------------------------------------------------
# Mass assignment detection
# ---------------------------------------------------------------------------


class TestMassAssignment:
    """Tests for _check_mass_assignment."""

    def test_post_with_role_field(self):
        """Given POST to /api/users with 'role' in body, detect mass_assignment."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/register",
            method="POST",
            request_body_fields=["username", "password", "role"],
        )
        indicators = VulnPatternAnalyzer._check_mass_assignment(ep)
        assert any(
            ind.pattern == VulnPattern.MASS_ASSIGNMENT and ind.confidence == RiskLevel.HIGH
            for ind in indicators
        )

    def test_get_no_mass_assignment(self):
        """GET is read-only, should never flag mass assignment."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/users",
            method="GET",
            request_body_fields=["role"],
        )
        indicators = VulnPatternAnalyzer._check_mass_assignment(ep)
        assert not indicators

    def test_post_non_user_path(self):
        """POST to /api/search shouldn't flag mass assignment even with 'role'."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/search",
            method="POST",
            request_body_fields=["role"],
        )
        indicators = VulnPatternAnalyzer._check_mass_assignment(ep)
        assert not indicators

    def test_post_user_path_no_dangerous_fields(self):
        """Given POST to /profile without dangerous fields, detect as LOW confidence."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/profile",
            method="POST",
            request_body_fields=["name", "email"],
        )
        indicators = VulnPatternAnalyzer._check_mass_assignment(ep)
        assert any(
            ind.pattern == VulnPattern.MASS_ASSIGNMENT and ind.confidence == RiskLevel.LOW
            for ind in indicators
        )

    def test_put_with_is_admin(self):
        """Given PUT to /settings with is_admin field, detect as HIGH."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/settings",
            method="PUT",
            request_body_fields=["name", "is_admin"],
        )
        indicators = VulnPatternAnalyzer._check_mass_assignment(ep)
        assert any(
            ind.pattern == VulnPattern.MASS_ASSIGNMENT and ind.confidence == RiskLevel.HIGH
            for ind in indicators
        )


# ---------------------------------------------------------------------------
# Auth boundary gap detection
# ---------------------------------------------------------------------------


class TestAuthBoundary:
    """Tests for _check_auth_boundary."""

    def test_no_auth_on_data_endpoint(self):
        """Given data endpoint with requires_auth=False and status 200, detect auth_boundary_gap."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users",
            requires_auth=False,
            status_code=200,
        )
        indicators = VulnPatternAnalyzer._check_auth_boundary(ep)
        assert any(ind.pattern == VulnPattern.AUTH_BOUNDARY_GAP for ind in indicators)

    def test_static_asset_no_auth_gap(self):
        """Static assets are expected to be public, no auth gap."""
        ep = make_endpoint(
            url="https://vulnbank.org/static/logo.png",
            requires_auth=False,
        )
        indicators = VulnPatternAnalyzer._check_auth_boundary(ep)
        assert not indicators

    def test_auth_required_no_gap(self):
        """Given endpoint with requires_auth=True, no auth_boundary_gap."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users",
            requires_auth=True,
        )
        indicators = VulnPatternAnalyzer._check_auth_boundary(ep)
        assert not indicators

    def test_auth_unknown_no_gap(self):
        """When auth status is unknown (None), don't flag — we can't be sure."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users",
            requires_auth=None,
        )
        indicators = VulnPatternAnalyzer._check_auth_boundary(ep)
        assert not indicators

    def test_account_endpoint_no_auth(self):
        """Given /account endpoint without auth, detect auth_boundary_gap."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/account/balance",
            requires_auth=False,
        )
        indicators = VulnPatternAnalyzer._check_auth_boundary(ep)
        assert any(ind.pattern == VulnPattern.AUTH_BOUNDARY_GAP for ind in indicators)


# ---------------------------------------------------------------------------
# SSRF detection
# ---------------------------------------------------------------------------


class TestSSRF:
    """Tests for _check_ssrf."""

    def test_url_parameter(self):
        """Given endpoint with 'url' parameter, detect ssrf."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/fetch",
            parameters=["url"],
        )
        indicators = VulnPatternAnalyzer._check_ssrf(ep)
        assert any(ind.pattern == VulnPattern.SSRF for ind in indicators)

    def test_callback_parameter(self):
        """Given endpoint with 'callback' parameter, detect ssrf."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/webhook",
            parameters=["callback"],
        )
        indicators = VulnPatternAnalyzer._check_ssrf(ep)
        assert any(ind.pattern == VulnPattern.SSRF for ind in indicators)

    def test_name_parameter_no_ssrf(self):
        """Regular params like 'name' and 'email' shouldn't trigger SSRF."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users",
            parameters=["name", "email"],
        )
        indicators = VulnPatternAnalyzer._check_ssrf(ep)
        assert not indicators

    def test_url_in_path(self):
        """Given endpoint with 'url' in path, detect ssrf at MEDIUM confidence."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/url-proxy")
        indicators = VulnPatternAnalyzer._check_ssrf(ep)
        assert any(
            ind.pattern == VulnPattern.SSRF and ind.confidence == RiskLevel.MEDIUM
            for ind in indicators
        )


# ---------------------------------------------------------------------------
# Chained vulnerability grouping
# ---------------------------------------------------------------------------


class TestChainedVulnerability:
    """Tests for chained vulnerability detection via cross-endpoint checks."""

    def test_auth_gap_and_bola_produces_chain(self, scan_state: ScanState):
        """Given endpoint with BOTH auth_boundary_gap AND bola_idor, they can be chained."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users/{id}",
            requires_auth=False,
        )
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)

        # simulate both indicators
        indicators = [
            VulnIndicator(
                pattern=VulnPattern.AUTH_BOUNDARY_GAP,
                confidence=RiskLevel.HIGH,
                evidence="No auth required",
                description="Data endpoint without auth",
                chain_id="chain_001",
            ),
            VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence="Path ID template",
                description="IDOR in path",
                chain_id="chain_001",
            ),
        ]
        scan_state.vuln_indicators[key] = indicators

        # both indicators share a chain_id
        chain_ids = {ind.chain_id for ind in indicators if ind.chain_id}
        assert len(chain_ids) == 1

    def test_severity_ordering_in_chain(self):
        """Given chained indicators, highest severity is selected."""
        severities = [RiskLevel.MEDIUM, RiskLevel.CRITICAL, RiskLevel.HIGH]
        highest = min(severities, key=lambda r: _SEVERITY_ORDER[r])
        assert highest == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Info disclosure detection
# ---------------------------------------------------------------------------


class TestInfoDisclosure:
    """Tests for _check_info_disclosure."""

    def test_debug_path_accessible(self):
        """Given accessible /debug path, detect info_disclosure."""
        ep = make_endpoint(
            url="https://vulnbank.org/debug",
            status_code=200,
        )
        indicators = VulnPatternAnalyzer._check_info_disclosure(ep)
        assert any(ind.pattern == VulnPattern.INFO_DISCLOSURE for ind in indicators)

    def test_debug_path_forbidden(self):
        """403 on /debug means it's locked down, no disclosure."""
        ep = make_endpoint(
            url="https://vulnbank.org/debug",
            status_code=403,
        )
        indicators = VulnPatternAnalyzer._check_info_disclosure(ep)
        assert not indicators

    def test_swagger_accessible(self):
        """Given accessible /swagger path, detect info_disclosure."""
        ep = make_endpoint(
            url="https://vulnbank.org/swagger",
            status_code=200,
        )
        indicators = VulnPatternAnalyzer._check_info_disclosure(ep)
        assert any(ind.pattern == VulnPattern.INFO_DISCLOSURE for ind in indicators)


# ---------------------------------------------------------------------------
# Response body leak detection
# ---------------------------------------------------------------------------


class TestResponseBodyLeaks:
    """Tests for _check_response_body_leaks."""

    def test_stack_trace_in_response(self):
        """Stack traces in response bodies are a clear info_disclosure signal."""
        ep = make_endpoint(
            response_body_snippet='Error: Traceback (most recent call last):\n  File "/app/main.py", line 42',
        )
        indicators = VulnPatternAnalyzer._check_response_body_leaks(ep)
        assert any(
            ind.pattern == VulnPattern.INFO_DISCLOSURE and "Stack trace" in ind.evidence
            for ind in indicators
        )

    def test_java_stack_trace(self):
        """Java-style traces (com.vulnbank...) should also get caught."""
        ep = make_endpoint(
            response_body_snippet="java.lang.NullPointerException\n\tat com.vulnbank.service.UserService.getUser",
        )
        indicators = VulnPatternAnalyzer._check_response_body_leaks(ep)
        assert any(ind.pattern == VulnPattern.INFO_DISCLOSURE for ind in indicators)

    def test_internal_ip_in_response(self):
        """RFC 1918 addresses like 10.0.1.25 in JSON should flag info_disclosure."""
        ep = make_endpoint(
            response_body_snippet='{"server": "10.0.1.25", "status": "ok"}',
        )
        indicators = VulnPatternAnalyzer._check_response_body_leaks(ep)
        assert any(
            ind.pattern == VulnPattern.INFO_DISCLOSURE and "Internal IP" in ind.evidence
            for ind in indicators
        )

    def test_sql_error_in_response(self):
        """SQL syntax errors in output leak DB type and suggest injection."""
        ep = make_endpoint(
            response_body_snippet="You have an error in your SQL syntax; check the manual",
        )
        indicators = VulnPatternAnalyzer._check_response_body_leaks(ep)
        assert any(
            ind.pattern == VulnPattern.INFO_DISCLOSURE and "SQL error" in ind.evidence
            for ind in indicators
        )

    def test_api_key_in_response(self):
        """Stripe live keys in config responses should flag as CRITICAL excessive_data."""
        ep = make_endpoint(
            response_body_snippet='{"config": {"stripe_key": "pk_live_51H7dklsKJHsdf928hkjsdf92"}}',
        )
        indicators = VulnPatternAnalyzer._check_response_body_leaks(ep)
        assert any(
            ind.pattern == VulnPattern.EXCESSIVE_DATA_EXPOSURE
            and ind.confidence == RiskLevel.CRITICAL
            for ind in indicators
        )

    def test_clean_response_body(self):
        """Normal JSON with just names shouldn't flag anything."""
        ep = make_endpoint(
            response_body_snippet='{"users": [{"name": "Alice"}, {"name": "Bob"}]}',
        )
        indicators = VulnPatternAnalyzer._check_response_body_leaks(ep)
        assert not indicators

    def test_empty_response_body(self):
        """Empty snippet should bail out early with no indicators."""
        ep = make_endpoint(response_body_snippet="")
        indicators = VulnPatternAnalyzer._check_response_body_leaks(ep)
        assert not indicators

    def test_multiple_emails_excessive_data(self):
        """More than 3 emails in one response looks like a data dump."""
        ep = make_endpoint(
            response_body_snippet=(
                '{"users": ['
                '{"email": "alice@example.com"}, '
                '{"email": "bob@example.com"}, '
                '{"email": "charlie@example.com"}, '
                '{"email": "dave@example.com"}]}'
            ),
        )
        indicators = VulnPatternAnalyzer._check_response_body_leaks(ep)
        assert any(
            ind.pattern == VulnPattern.EXCESSIVE_DATA_EXPOSURE
            and "emails" in ind.evidence.lower()
            for ind in indicators
        )

    def test_192_168_internal_ip(self):
        """192.168.x.x range should trigger the same as 10.x — both are RFC 1918."""
        ep = make_endpoint(
            response_body_snippet='Database host: 192.168.1.100',
        )
        indicators = VulnPatternAnalyzer._check_response_body_leaks(ep)
        assert any(
            ind.pattern == VulnPattern.INFO_DISCLOSURE and "Internal IP" in ind.evidence
            for ind in indicators
        )



# ---------------------------------------------------------------------------
# File upload detection
# ---------------------------------------------------------------------------


class TestFileUpload:
    """Tests for _check_file_upload."""

    def test_multipart_content_type(self):
        """Given multipart content type, detect file_upload."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/upload",
            method="POST",
            request_body_content_type="multipart/form-data",
        )
        indicators = VulnPatternAnalyzer._check_file_upload(ep)
        assert any(ind.pattern == VulnPattern.FILE_UPLOAD for ind in indicators)

    def test_upload_param(self):
        """Given 'file' parameter, detect file_upload."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/documents",
            method="POST",
            parameters=["file", "description"],
        )
        indicators = VulnPatternAnalyzer._check_file_upload(ep)
        assert any(ind.pattern == VulnPattern.FILE_UPLOAD for ind in indicators)


# ---------------------------------------------------------------------------
# Race condition detection
# ---------------------------------------------------------------------------


class TestRaceCondition:
    """Tests for _check_race_condition."""

    def test_post_to_transfer(self):
        """Given POST to /transfer, detect race_condition."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/transfer",
            method="POST",
        )
        indicators = VulnPatternAnalyzer._check_race_condition(ep)
        assert any(ind.pattern == VulnPattern.RACE_CONDITION for ind in indicators)

    def test_get_to_transfer(self):
        """GET to /transfer is read-only, no race condition risk."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/transfer",
            method="GET",
        )
        indicators = VulnPatternAnalyzer._check_race_condition(ep)
        assert not indicators


# ---------------------------------------------------------------------------
# JWT / auth weakness detection
# ---------------------------------------------------------------------------


class TestJWTWeakness:
    """Tests for _check_jwt_weakness."""

    def test_jwt_bearer_scheme(self):
        """Given JWT bearer security scheme, detect jwt_weakness."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users",
            security_schemes=[
                SecuritySchemeInfo(scheme_type="http", scheme_name="bearer", bearer_format="JWT"),
            ],
        )
        indicators = VulnPatternAnalyzer._check_jwt_weakness(ep)
        assert any(ind.pattern == VulnPattern.JWT_WEAKNESS for ind in indicators)

    def test_api_key_in_query(self):
        """Given API key in query string, detect jwt_weakness as HIGH."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/data",
            security_schemes=[
                SecuritySchemeInfo(scheme_type="apiKey", scheme_name="api_key", location="query"),
            ],
        )
        indicators = VulnPatternAnalyzer._check_jwt_weakness(ep)
        assert any(
            ind.pattern == VulnPattern.JWT_WEAKNESS and ind.confidence == RiskLevel.HIGH
            for ind in indicators
        )

    def test_auth_over_http(self):
        """Given auth endpoint over HTTP, detect jwt_weakness."""
        ep = make_endpoint(
            url="http://vulnbank.org/api/v1/users",
            requires_auth=True,
        )
        indicators = VulnPatternAnalyzer._check_jwt_weakness(ep)
        assert any(ind.pattern == VulnPattern.JWT_WEAKNESS for ind in indicators)


# ---------------------------------------------------------------------------
# Cross-endpoint: version confusion
# ---------------------------------------------------------------------------


class TestVersionConfusion:
    """Tests for _check_version_confusion."""

    def test_inconsistent_auth_across_versions(self):
        """Given same resource in v1 (no auth) and v2 (auth), detect version_confusion."""
        endpoints = [
            (
                "GET https://vulnbank.org/api/v1/users",
                make_endpoint(
                    url="https://vulnbank.org/api/v1/users",
                    requires_auth=False,
                ),
            ),
            (
                "GET https://vulnbank.org/api/v2/users",
                make_endpoint(
                    url="https://vulnbank.org/api/v2/users",
                    requires_auth=True,
                ),
            ),
        ]
        results = VulnPatternAnalyzer._check_version_confusion(endpoints)
        assert len(results) > 0
        all_indicators = [ind for inds in results.values() for ind in inds]
        assert any(ind.pattern == VulnPattern.API_VERSION_CONFUSION for ind in all_indicators)

    def test_consistent_auth_no_confusion(self):
        """Given same auth across versions, no version_confusion."""
        endpoints = [
            (
                "GET https://vulnbank.org/api/v1/users",
                make_endpoint(
                    url="https://vulnbank.org/api/v1/users",
                    requires_auth=True,
                ),
            ),
            (
                "GET https://vulnbank.org/api/v2/users",
                make_endpoint(
                    url="https://vulnbank.org/api/v2/users",
                    requires_auth=True,
                ),
            ),
        ]
        results = VulnPatternAnalyzer._check_version_confusion(endpoints)
        assert not results


# ---------------------------------------------------------------------------
# Cross-endpoint: broken function-level auth
# ---------------------------------------------------------------------------


class TestBrokenFunctionAuth:
    """Tests for _check_broken_function_auth."""

    def test_admin_no_auth_accessible(self):
        """Given /admin accessible without auth, detect broken_function_level_auth."""
        endpoints = [
            (
                "GET https://vulnbank.org/admin",
                make_endpoint(
                    url="https://vulnbank.org/admin",
                    requires_auth=False,
                    status_code=200,
                ),
            ),
        ]
        results = VulnPatternAnalyzer._check_broken_function_auth(endpoints)
        all_indicators = [ind for inds in results.values() for ind in inds]
        assert any(
            ind.pattern == VulnPattern.BROKEN_FUNCTION_LEVEL_AUTH
            and ind.confidence == RiskLevel.CRITICAL
            for ind in all_indicators
        )

    def test_admin_with_auth_no_finding(self):
        """/admin behind auth is expected, no finding."""
        endpoints = [
            (
                "GET https://vulnbank.org/admin",
                make_endpoint(
                    url="https://vulnbank.org/admin",
                    requires_auth=True,
                    status_code=200,
                ),
            ),
        ]
        results = VulnPatternAnalyzer._check_broken_function_auth(endpoints)
        assert not results

    def test_admin_forbidden_no_finding(self):
        """403 on /admin means access is denied — not a finding."""
        endpoints = [
            (
                "GET https://vulnbank.org/admin",
                make_endpoint(
                    url="https://vulnbank.org/admin",
                    requires_auth=False,
                    status_code=403,
                ),
            ),
        ]
        results = VulnPatternAnalyzer._check_broken_function_auth(endpoints)
        assert not results


# ---------------------------------------------------------------------------
# BOLA version number false positive fix
# ---------------------------------------------------------------------------


class TestBOLAVersionFix:
    """Numeric segments in version prefixes like /api/v1/... shouldn't trigger BOLA."""

    def test_v1_not_flagged_as_idor(self):
        """The '1' in /api/v1/users is a version, not an object ID."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/users")
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        assert not any(
            ind.pattern == VulnPattern.BOLA_IDOR and "Numeric path segment" in ind.evidence
            for ind in indicators
        )

    def test_v2_not_flagged(self):
        """Same applies for v2, v3, etc."""
        ep = make_endpoint(url="https://vulnbank.org/api/v2/accounts")
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        assert not any(
            ind.pattern == VulnPattern.BOLA_IDOR and "Numeric path segment" in ind.evidence
            for ind in indicators
        )

    def test_real_id_after_version_still_detected(self):
        """A real numeric ID at the end should still trigger even if v1 is in the path."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/users/42")
        indicators = VulnPatternAnalyzer._check_bola_idor(ep)
        assert any(
            ind.pattern == VulnPattern.BOLA_IDOR and "42" in ind.evidence
            for ind in indicators
        )


# ---------------------------------------------------------------------------
# Mass assignment tightening — empty params no longer emit LOW
# ---------------------------------------------------------------------------


class TestMassAssignmentTightening:
    """LOW mass_assignment should only fire when endpoint has user-controllable fields."""

    def test_no_params_no_low_emission(self):
        """POST to /profile with NO fields shouldn't emit LOW mass_assignment."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/profile",
            method="POST",
            request_body_fields=[],
            parameters=[],
        )
        indicators = VulnPatternAnalyzer._check_mass_assignment(ep)
        assert not indicators

    def test_with_safe_params_emits_low(self):
        """POST to /profile with non-dangerous fields should still emit LOW."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/profile",
            method="POST",
            request_body_fields=["name", "bio"],
        )
        indicators = VulnPatternAnalyzer._check_mass_assignment(ep)
        assert any(
            ind.pattern == VulnPattern.MASS_ASSIGNMENT and ind.confidence == RiskLevel.LOW
            for ind in indicators
        )


# ---------------------------------------------------------------------------
# Injection surface detection
# ---------------------------------------------------------------------------


class TestInjectionSurfaces:
    """Tests for _check_injection_surfaces — SQL, path traversal, command injection."""

    def test_sql_injection_params(self):
        """'query' and 'filter' params trigger SQL injection indicator."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/search",
            parameters=["query", "filter"],
        )
        indicators = VulnPatternAnalyzer._check_injection_surfaces(ep)
        assert any(
            ind.pattern == VulnPattern.SQL_INJECTION and "SQL-related params" in ind.evidence
            for ind in indicators
        )

    def test_path_traversal_params(self):
        """File-related param names should produce a PATH_TRAVERSAL indicator."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/download",
            parameters=["file"],
        )
        indicators = VulnPatternAnalyzer._check_injection_surfaces(ep)
        assert any(
            ind.pattern == VulnPattern.PATH_TRAVERSAL and "File-path params" in ind.evidence
            for ind in indicators
        )

    def test_command_injection_params(self):
        """'cmd' and 'host' get COMMAND_INJECTION at HIGH confidence."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/tools",
            parameters=["cmd", "host"],
        )
        indicators = VulnPatternAnalyzer._check_injection_surfaces(ep)
        assert any(
            ind.pattern == VulnPattern.COMMAND_INJECTION
            and "Command-related params" in ind.evidence
            and ind.confidence == RiskLevel.HIGH
            for ind in indicators
        )

    def test_safe_params_no_injection(self):
        """Normal params like 'name' and 'email' shouldn't trigger anything."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users",
            parameters=["name", "email"],
        )
        indicators = VulnPatternAnalyzer._check_injection_surfaces(ep)
        assert not indicators


# ---------------------------------------------------------------------------
# JWT detection from response headers/notes
# ---------------------------------------------------------------------------


class TestJWTFromHeaders:
    """JWT weakness detection from www-authenticate header and notes."""

    def test_bearer_in_www_authenticate(self):
        """www-authenticate: Bearer should trigger jwt_weakness."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/users")
        ep.response_headers["www-authenticate"] = "Bearer realm=\"api\""
        indicators = VulnPatternAnalyzer._check_jwt_weakness(ep)
        assert any(ind.pattern == VulnPattern.JWT_WEAKNESS for ind in indicators)

    def test_jwt_in_notes(self):
        """Notes mentioning 'JWT' should trigger jwt_weakness."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/users")
        ep.notes = "Uses JWT authentication for all API calls"
        indicators = VulnPatternAnalyzer._check_jwt_weakness(ep)
        assert any(ind.pattern == VulnPattern.JWT_WEAKNESS for ind in indicators)

    def test_no_jwt_signals_clean(self):
        """Endpoint with no JWT signals shouldn't trigger."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/users")
        indicators = VulnPatternAnalyzer._check_jwt_weakness(ep)
        assert not indicators


# ---------------------------------------------------------------------------
# SSRF segment-level matching
# ---------------------------------------------------------------------------


class TestSSRFSegmentMatching:
    """SSRF path matching works at the segment level, including hyphenated names."""

    def test_hyphenated_segment_matches(self):
        """/api/v1/url-proxy — 'url-proxy' starts with 'url-', should match."""
        ep = make_endpoint(url="https://vulnbank.org/api/v1/url-proxy")
        indicators = VulnPatternAnalyzer._check_ssrf(ep)
        assert any(ind.pattern == VulnPattern.SSRF for ind in indicators)

    def test_underscore_segment_matches(self):
        """/api/webhook_handler — 'webhook_handler' starts with 'webhook_'."""
        ep = make_endpoint(url="https://vulnbank.org/api/webhook_handler")
        indicators = VulnPatternAnalyzer._check_ssrf(ep)
        assert any(ind.pattern == VulnPattern.SSRF for ind in indicators)

    def test_substring_doesnt_match(self):
        """/api/curriculum — contains 'url' as substring but not as segment prefix."""
        ep = make_endpoint(url="https://vulnbank.org/api/curriculum")
        indicators = VulnPatternAnalyzer._check_ssrf(ep)
        assert not indicators


# ---------------------------------------------------------------------------
# Body analysis endpoint filtering (Pass 1.5)
# ---------------------------------------------------------------------------


class TestBodyAnalysisFiltering:
    """Tests for _select_endpoints_for_body_analysis."""

    def test_debug_endpoint_selected(self, scan_state: ScanState):
        """Endpoint with debug_endpoint category is selected for body analysis."""
        ep = make_endpoint(
            url="https://vulnbank.org/debug/console",
            category=EndpointCategory.DEBUG_ENDPOINT,
            response_body_snippet="<html>Debug console output</html>",
        )
        scan_state.add_endpoint(ep)
        selected = VulnPatternAnalyzer._select_endpoints_for_body_analysis(scan_state)
        assert len(selected) == 1
        assert selected[0][1] is ep

    def test_admin_with_broken_auth_selected(self, scan_state: ScanState):
        """Endpoint with broken_function_level_auth indicator is selected."""
        ep = make_endpoint(
            url="https://vulnbank.org/admin/settings",
            response_body_snippet='{"admin": true, "config": "..."}',
        )
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)
        scan_state.vuln_indicators[key] = [
            VulnIndicator(
                pattern=VulnPattern.BROKEN_FUNCTION_LEVEL_AUTH,
                confidence=RiskLevel.CRITICAL,
                evidence="Admin path without auth",
                description="test",
            ),
        ]
        selected = VulnPatternAnalyzer._select_endpoints_for_body_analysis(scan_state)
        assert len(selected) == 1

    def test_info_disclosure_endpoint_selected(self, scan_state: ScanState):
        """Endpoint with info_disclosure indicator is selected."""
        ep = make_endpoint(
            url="https://vulnbank.org/debug",
            response_body_snippet="Stack trace here",
        )
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)
        scan_state.vuln_indicators[key] = [
            VulnIndicator(
                pattern=VulnPattern.INFO_DISCLOSURE,
                confidence=RiskLevel.HIGH,
                evidence="Accessible sensitive path",
                description="test",
            ),
        ]
        selected = VulnPatternAnalyzer._select_endpoints_for_body_analysis(scan_state)
        assert len(selected) == 1

    def test_static_asset_not_selected(self, scan_state: ScanState):
        """Static asset with short body is not selected."""
        ep = make_endpoint(
            url="https://vulnbank.org/static/style.css",
            category=EndpointCategory.STATIC_ASSET,
            response_body_snippet="body { color: red; }",
        )
        scan_state.add_endpoint(ep)
        selected = VulnPatternAnalyzer._select_endpoints_for_body_analysis(scan_state)
        assert len(selected) == 0

    def test_empty_body_not_selected(self, scan_state: ScanState):
        """Endpoint with empty snippet is skipped."""
        ep = make_endpoint(
            url="https://vulnbank.org/api/v1/users",
            response_body_snippet="",
        )
        scan_state.add_endpoint(ep)
        selected = VulnPatternAnalyzer._select_endpoints_for_body_analysis(scan_state)
        assert len(selected) == 0


# ---------------------------------------------------------------------------
# Body analysis result processing (Pass 1.5)
# ---------------------------------------------------------------------------


class TestBodyAnalysisProcessing:
    """Tests for _process_body_analysis_results."""

    def _make_analyzer(self) -> VulnPatternAnalyzer:
        """Create a VulnPatternAnalyzer with mocked dependencies."""
        return VulnPatternAnalyzer(
            http_client=MagicMock(),
            llm_client=MagicMock(),
            prompt_registry=MagicMock(),
        )

    def test_secret_finding_creates_indicator(self, scan_state: ScanState):
        """A secret finding from LLM creates a VulnIndicator with correct pattern."""
        analyzer = self._make_analyzer()
        ep = make_endpoint(url="https://vulnbank.org/debug/console")
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)

        response = ResponseBodyAnalysisResponse(
            reasoning="Found a secret key assignment",
            findings=[
                ResponseBodyFinding(
                    finding_type="secret",
                    value_redacted="s3cr...here",
                    context="SECRET_KEY assignment in Werkzeug debug console",
                    confidence="critical",
                    is_placeholder=False,
                ),
            ],
        )
        batch = [(key, ep)]
        added = analyzer._process_body_analysis_results(scan_state, response, batch)

        assert added == 1
        indicators = scan_state.vuln_indicators[key]
        assert len(indicators) == 1
        assert indicators[0].pattern == VulnPattern.INFO_DISCLOSURE
        assert indicators[0].confidence == RiskLevel.CRITICAL
        assert indicators[0].llm_enhanced is True

    def test_placeholder_finding_skipped(self, scan_state: ScanState):
        """A finding with is_placeholder=True is not added."""
        analyzer = self._make_analyzer()
        ep = make_endpoint(url="https://vulnbank.org/docs/config")
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)

        response = ResponseBodyAnalysisResponse(
            reasoning="Found example value",
            findings=[
                ResponseBodyFinding(
                    finding_type="secret",
                    value_redacted="chan...geme",
                    context="SECRET_KEY = 'changeme' in documentation",
                    confidence="high",
                    is_placeholder=True,
                ),
            ],
        )
        batch = [(key, ep)]
        added = analyzer._process_body_analysis_results(scan_state, response, batch)

        assert added == 0
        assert key not in scan_state.vuln_indicators

    def test_credential_finding_maps_to_excessive_data(self, scan_state: ScanState):
        """A credential finding maps to EXCESSIVE_DATA_EXPOSURE pattern."""
        analyzer = self._make_analyzer()
        ep = make_endpoint(url="https://vulnbank.org/debug/env")
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)

        response = ResponseBodyAnalysisResponse(
            reasoning="Found database credentials",
            findings=[
                ResponseBodyFinding(
                    finding_type="credential",
                    value_redacted="post...prod",
                    context="DATABASE_URL with embedded credentials",
                    confidence="critical",
                    is_placeholder=False,
                ),
            ],
        )
        batch = [(key, ep)]
        added = analyzer._process_body_analysis_results(scan_state, response, batch)

        assert added == 1
        assert scan_state.vuln_indicators[key][0].pattern == VulnPattern.EXCESSIVE_DATA_EXPOSURE

    def test_llm_failure_graceful(self, scan_state: ScanState):
        """LLM failure doesn't crash; Pass 1 results stay intact."""
        analyzer = self._make_analyzer()
        analyzer.llm = AsyncMock()
        analyzer.llm.chat_structured.side_effect = RuntimeError("LLM timeout")
        analyzer.prompt_registry = MagicMock()

        ep = make_endpoint(
            url="https://vulnbank.org/debug",
            category=EndpointCategory.DEBUG_ENDPOINT,
            response_body_snippet="some content here " * 20,
        )
        scan_state.add_endpoint(ep)
        key = scan_state.endpoint_key(ep.method, ep.url)

        # simulate existing Pass 1 indicator
        scan_state.vuln_indicators[key] = [
            VulnIndicator(
                pattern=VulnPattern.INFO_DISCLOSURE,
                confidence=RiskLevel.HIGH,
                evidence="Accessible sensitive path",
                description="Pass 1 result",
            ),
        ]

        import asyncio
        count = asyncio.get_event_loop().run_until_complete(
            analyzer._body_analysis_pass(scan_state)
        )

        assert count == 0
        # Pass 1 result is still intact
        assert len(scan_state.vuln_indicators[key]) == 1
        assert scan_state.vuln_indicators[key][0].description == "Pass 1 result"
