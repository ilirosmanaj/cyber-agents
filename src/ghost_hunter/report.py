"""LLM-synthesized penetration test report from completed scan state."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

from src.ghost_hunter.clients.llm import LLMClient
from src.ghost_hunter.models import (
    AttackSurfaceEntry,
    Endpoint,
    Finding,
    RiskLevel,
    ScanState,
    VulnIndicator,
)
from src.ghost_hunter.prompts import PromptRegistry

logger = logging.getLogger(__name__)

# 32k cap matches most LLM provider output limits
_TOKENS_PER_FINDING = 400
_BASE_REPORT_TOKENS = 8000
_MAX_REPORT_TOKENS = 32768
_SNIPPET_MAX_CHARS = 300
# Info disclosure (e.g. debug console) often exposes secrets; show full response
_SNIPPET_MAX_CHARS_INFO_DISCLOSURE = 8000

CWE_REFERENCES: dict[str, str] = {
    "bola_idor": "CWE-639 (Authorization Bypass Through User-Controlled Key)",
    "mass_assignment": "CWE-915 (Improperly Controlled Modification of Dynamically-Determined Object Attributes)",
    "ssrf": "CWE-918 (Server-Side Request Forgery)",
    "file_upload": "CWE-434 (Unrestricted Upload of File with Dangerous Type)",
    "jwt_weakness": "CWE-347 (Improper Verification of Cryptographic Signature)",
    "api_version_confusion": "CWE-436 (Interpretation Conflict)",
    "race_condition": "CWE-362 (Race Condition)",
    "prompt_injection": "CWE-94 (Code Injection)",
    "info_disclosure": "CWE-200 (Exposure of Sensitive Information)",
    "auth_boundary_gap": "CWE-862 (Missing Authorization)",
    "excessive_data_exposure": "CWE-213 (Exposure of Sensitive Information Due to Incompatible Policies)",
    "broken_function_level_auth": "CWE-285 (Improper Authorization)",
    "chained_vulnerability": "CWE-20 (Improper Input Validation)",
    "sql_injection": "CWE-89 (SQL Injection)",
    "path_traversal": "CWE-22 (Path Traversal)",
    "command_injection": "CWE-78 (OS Command Injection)",
}

_AGENT_DISPLAY_NAMES: dict[str, str] = {
    "passive_recon": "Passive Reconnaissance",
    "web_crawler": "Web Crawling",
    "planner": "Scan Planning",
    "api_discovery": "API Discovery",
    "js_analyzer": "JavaScript Analysis",
    "hypothesis": "Hypothesis Testing",
    "classifier": "Endpoint Classification",
    "vuln_analyzer": "Vulnerability Analysis",
    "verifier": "Finding Verification",
    "prioritizer": "Risk Prioritization",
}


_FINDING_EXAMPLE = """\
EXAMPLE FINDING FORMAT (follow this structure for every critical/high finding):

### BOLA/IDOR — GET https://target.com/api/v1/accounts/{id} **[HIGH]**
**CWE:** CWE-639 (Authorization Bypass Through User-Controlled Key)
**Confidence:** Likely — deterministic pattern match + LLM analysis (no active exploitation)
**Endpoint:** GET https://target.com/api/v1/accounts/{id}
**Parameters:** id (path)
**Auth:** requires_auth

**Evidence:**
Endpoint returns full account details including balance, email, SSN when accessed \
with any valid session token. No ownership check on the `id` parameter.

**Attack Scenario:**
1. Attacker authenticates as user A and receives session token
2. Attacker sends GET /api/v1/accounts/78432 with their own session token
3. Server returns victim's account details (balance, email, SSN) without verifying ownership

**Impact:**
Any authenticated user can read any other user's financial data by iterating account IDs \
at GET /api/v1/accounts/{id}. Exposed fields: balance, email, SSN.

**Remediation:**
Add server-side ownership check in the /api/v1/accounts/{id} handler: verify that the \
authenticated user's ID matches the requested account's owner_id before returning data. \
Return 403 if mismatch.

**Test Command:**
```
curl https://target.com/api/v1/accounts/78432 -H "Authorization: Bearer eyJhbG..."
```

NOTE: id is a PATH parameter so the value 78432 goes in the URL, not in a header or body. \
Always match parameter location (path/query/body) to the curl structure. \
Use concrete example values, not bare <placeholders>."""


def _format_auth_status(ep: Endpoint) -> str | None:
    """Return 'requires_auth', 'no_auth', or None."""
    if ep.requires_auth is True:
        return "requires_auth"
    if ep.requires_auth is False:
        return "no_auth"
    return None


def _endpoint_context_for_finding(f: Finding, state: ScanState) -> str:
    """Look up endpoint metadata from a finding title and format as context lines."""
    ep = _lookup_endpoint_from_finding(f, state)
    if ep is None:
        return ""

    parts: list[str] = []
    if ep.parameters:
        parts.append(f"\n  Params: {', '.join(ep.parameters)}")
    auth = _format_auth_status(ep)
    if auth:
        parts.append(f"\n  Auth: {auth}")
    if ep.response_fields:
        parts.append(f"\n  Response fields: {', '.join(ep.response_fields)}")
    if ep.response_body_snippet:
        s = ep.response_body_snippet
        if len(s) > _SNIPPET_MAX_CHARS:
            s = s[:_SNIPPET_MAX_CHARS] + "..."
        parts.append(f"\n  Response snippet: {s}")
    return "".join(parts)


def _lookup_endpoint_from_finding(f: Finding, state: ScanState) -> Endpoint | None:
    """Extract endpoint key from finding title and look up in state.

    Finding titles follow the format "PATTERN — METHOD URL"
    (optionally prefixed with "[LLM] ").
    """
    if " — " not in f.title:
        return None
    after_dash = f.title.split(" — ", 1)[1]
    parts = after_dash.split(" ", 1)
    if len(parts) != 2:
        return None
    method, url = parts
    return state.find_endpoint(method, url)


def _compute_max_tokens(finding_count: int) -> int:
    """Scale LLM max_tokens based on number of findings."""
    scaled = _BASE_REPORT_TOKENS + (finding_count * _TOKENS_PER_FINDING)
    return min(scaled, _MAX_REPORT_TOKENS)


def _build_cwe_reference_block() -> str:
    """Format the CWE reference map as context for the LLM."""
    lines = [f"  {pattern} → {cwe}" for pattern, cwe in CWE_REFERENCES.items()]
    return "CWE REFERENCE MAP:\n" + "\n".join(lines)


def _build_tech_fingerprint_section(state: ScanState) -> str:
    """Format tech fingerprint data for report context or fallback."""
    fp = state.tech_fingerprint
    lines = []
    if fp.server:
        lines.append(f"Server: {fp.server}")
    if fp.frameworks:
        lines.append(f"Frameworks: {', '.join(fp.frameworks)}")
    if fp.technologies:
        lines.append(f"Technologies: {', '.join(fp.technologies)}")
    if fp.security_headers:
        lines.append(f"Security headers: {', '.join(fp.security_headers.keys())}")
    if fp.missing_security_headers:
        lines.append(f"Missing headers: {', '.join(fp.missing_security_headers)}")
    if fp.cookies:
        lines.append(f"Cookies: {', '.join(fp.cookies)}")
    return "\n".join(lines)


def _format_indicator_lines(indicators: list[VulnIndicator]) -> str:
    """Format vuln indicators as indented context lines."""
    lines: list[str] = []
    for v in indicators:
        lines.append(
            f"    - {v.pattern.value} ({v.confidence.value.upper()}): {v.evidence}"
        )
        if v.description:
            lines.append(f"      {v.description}")
    return "\n".join(lines)


_METADATA_PATH_PREFIXES = ("/latest/meta-data", "/latest/api/token")
_MALFORMED_URL_CHARS = set(',;"\'<>{}')


def _is_valid_surface_entry(entry: AttackSurfaceEntry) -> bool:
    """Filter out malformed URLs and misclassified internal paths."""
    url = entry.endpoint.url
    if _MALFORMED_URL_CHARS & set(url):
        return False
    path = urlparse(url).path
    for prefix in _METADATA_PATH_PREFIXES:
        if path.startswith(prefix):
            return False
    return True


def _is_auth_endpoint_false_positive(entry: AttackSurfaceEntry) -> bool:
    """Auth endpoints whose only indicator is excessive_data_exposure are expected behavior."""
    if entry.category.value != "auth_endpoint":
        return False
    return all(v.pattern.value == "excessive_data_exposure" for v in entry.vuln_indicators)


def _build_attack_surface_table(state: ScanState) -> str:
    """Format attack surface entries as a markdown table."""
    if not state.attack_surface:
        return ""
    rows = [
        "| # | Risk | Method | URL | Category | Indicators |",
        "|----|------|--------|-----|----------|------------|",
    ]
    rank = 0
    for entry in state.attack_surface:
        if not _is_valid_surface_entry(entry):
            continue
        if not entry.vuln_indicators:
            continue
        if _is_auth_endpoint_false_positive(entry):
            continue
        rank += 1
        vulns = ", ".join(
            dict.fromkeys(v.pattern.value for v in entry.vuln_indicators)
        )
        rows.append(
            f"| {rank} "
            f"| {entry.risk_level.value.upper()} "
            f"| {entry.endpoint.method} "
            f"| {entry.endpoint.url} "
            f"| {entry.category.value} "
            f"| {vulns or 'none'} |"
        )
    if rank == 0:
        return ""
    return "\n".join(rows)


def _lookup_cwe_for_finding(f: Finding) -> str:
    """Map a finding's type or title to a CWE reference string.

    Checks finding_type (stripping 'vuln_' prefix) then falls back to
    keyword matching against the title.
    """
    finding_type = f.finding_type.removeprefix("vuln_")

    if finding_type in CWE_REFERENCES:
        return CWE_REFERENCES[finding_type]

    title_lower = f.title.lower()
    for pattern, cwe in CWE_REFERENCES.items():
        if pattern.replace("_", " ") in title_lower:
            return cwe

    return ""


def _lookup_tests_for_finding(f: Finding, state: ScanState) -> list[str]:
    """Cross-reference a finding with attack surface to find suggested tests."""
    ep = _lookup_endpoint_from_finding(f, state)
    if ep is None:
        return []

    for entry in state.attack_surface:
        if entry.endpoint.url == ep.url and entry.endpoint.method == ep.method:
            return entry.suggested_tests

    return []


def _format_finding_enriched(f: Finding, state: ScanState | None = None) -> str:
    """Format a single finding with CWE, tests, and verification status."""
    validated_tag = ""
    if f.validated is True:
        validated_tag = " [VALIDATED]"
    elif f.validated is False:
        validated_tag = " [REFUTED]"

    cwe = _lookup_cwe_for_finding(f)
    cwe_tag = f" | {cwe}" if cwe else ""

    line = (
        f"- [{f.severity.value.upper()}]{validated_tag}{cwe_tag} {f.title}\n"
        f"  Detail: {f.detail}"
    )
    if f.evidence:
        line += f"\n  Evidence: {f.evidence}"

    if f.verification_status:
        line += f"\n  Verification: {f.verification_status}"
        if f.verification_note:
            line += f" — {f.verification_note}"

    if state is not None:
        line += _endpoint_context_for_finding(f, state)
        tests = _lookup_tests_for_finding(f, state)
        if tests:
            line += f"\n  Suggested tests: {' | '.join(tests)}"

    return line


def _build_enriched_findings(state: ScanState) -> str:
    """Group findings by severity with CWE, tests, and verification info."""
    severity_order = [
        RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM,
        RiskLevel.LOW, RiskLevel.INFO,
    ]
    groups: dict[RiskLevel, list[Finding]] = {}
    for f in state.findings:
        if _is_false_positive(f, state):
            continue
        groups.setdefault(f.severity, []).append(f)

    sections: list[str] = []
    for sev in severity_order:
        findings = groups.get(sev, [])
        if not findings:
            continue
        header = (
            f"FINDINGS — {sev.value.upper()} ({len(findings)} total)"
            f" — INCLUDE ALL IN REPORT"
        )
        lines = [header]
        for f in findings:
            lines.append(_format_finding_enriched(f, state))
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _build_strategy_section(state: ScanState) -> str:
    """Format planner strategy and scan insights."""
    lines: list[str] = []

    if state.scan_strategy:
        s = state.scan_strategy
        lines.append("SCAN STRATEGY (from planner agent):")
        if s.focus_areas:
            lines.append(f"  Focus areas: {', '.join(s.focus_areas)}")
        if s.tech_hypotheses:
            lines.append(f"  Tech hypotheses: {', '.join(s.tech_hypotheses)}")
        lines.append(f"  Scan depth: {s.scan_depth}")
        if s.priority_patterns:
            lines.append(f"  Priority patterns: {', '.join(s.priority_patterns)}")

    insights = state.insights_context()
    if insights and insights != "No prior insights.":
        lines.append(f"\nSCAN INSIGHTS:\n{insights}")

    return "\n".join(lines)


def _build_chain_analysis(state: ScanState) -> str:
    """Find explicit (chain_id) and implicit (multi-pattern) vuln chains."""
    chains: dict[str, list[tuple[str, VulnIndicator]]] = {}
    for ep_key, indicators in state.vuln_indicators.items():
        for ind in indicators:
            if ind.chain_id and not ind.suppressed:
                chains.setdefault(ind.chain_id, []).append((ep_key, ind))

    # implicit: same endpoint with 2+ different high-severity patterns
    implicit: list[tuple[str, list[VulnIndicator]]] = []
    explicit_pairs = {
        (ep_key, ind.pattern.value)
        for members in chains.values()
        for ep_key, ind in members
    }

    for ep_key, indicators in state.vuln_indicators.items():
        high_sev = [
            ind for ind in indicators
            if not ind.suppressed
            and ind.confidence in (RiskLevel.CRITICAL, RiskLevel.HIGH)
            and (ep_key, ind.pattern.value) not in explicit_pairs
        ]
        patterns = {ind.pattern for ind in high_sev}
        if len(patterns) >= 2:
            implicit.append((ep_key, high_sev))

    if not chains and not implicit:
        return ""

    lines = ["VULNERABILITY CHAINS:"]

    for chain_id, members in chains.items():
        endpoints = sorted({ep_key for ep_key, _ in members})
        patterns = [ind.pattern.value for _, ind in members]
        lines.append(f"\nChain {chain_id}:")
        lines.append(f"  Endpoints: {', '.join(endpoints)}")
        lines.append(f"  Patterns: {' → '.join(patterns)}")
        for ep_key, ind in members:
            lines.append(
                f"  - [{ind.confidence.value.upper()}] "
                f"{ind.pattern.value} at {ep_key}: {ind.evidence}"
            )

    for ep_key, indicators in implicit:
        patterns = sorted({ind.pattern.value for ind in indicators})
        lines.append(f"\nImplicit chain at {ep_key}:")
        lines.append(f"  Patterns: {', '.join(patterns)}")
        for ind in indicators:
            lines.append(
                f"  - [{ind.confidence.value.upper()}] "
                f"{ind.pattern.value}: {ind.evidence}"
            )

    return "\n".join(lines)


def _build_section_instructions(state: ScanState) -> str:
    """Build the section list for the LLM based on what data is available."""
    severity_counts: dict[str, int] = {}
    for f in state.findings:
        sev = f.severity.value.upper()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    critical_high = severity_counts.get("CRITICAL", 0) + severity_counts.get("HIGH", 0)

    lines = [
        "REPORT SECTIONS (generate in this order):",
        "1. ## Executive Summary — 3-5 sentences: scope, findings count by severity, "
        "overall risk, most critical discovery.",
        "2. ## Scope & Limitations — What was tested (passive recon, crawling, API spec "
        "parsing, JS analysis, pattern matching, LLM analysis). What was NOT tested: "
        "authenticated endpoint behavior, business logic beyond crawlable paths, "
        "rate limiting, DoS.",
    ]

    if state.scan_strategy:
        lines.append(
            "3. ## Methodology — Techniques used, planner strategy assessment. "
            "Include scan depth and focus areas."
        )
    else:
        lines.append("3. ## Methodology — Techniques used.")

    lines.append(
        "4. ## Target Profile — Technology fingerprint, frameworks, security headers."
    )
    lines.append(
        f"5. ## Critical & High Findings — ALL {critical_high} critical+high findings. "
        f"Each needs: CWE, confidence, endpoint, evidence, attack scenario "
        f"(numbered steps), impact, remediation, test command."
    )

    medium_low = severity_counts.get("MEDIUM", 0) + severity_counts.get("LOW", 0)
    if medium_low:
        lines.append(
            f"6. ## Medium & Low Findings — Summarize {medium_low} remaining findings."
        )

    lines.extend([
        "7. ## Attack Surface Map — Write ONLY a brief summary paragraph. "
        "The table will be auto-inserted.",
        "8. ## Vulnerability Pattern Analysis — Distribution, chained vulns, "
        "LLM-enhanced vs deterministic, suppressed false positives.",
        "9. ## Recommendations — Prioritized roadmap: immediate (critical), "
        "short-term (high), medium-term (medium).",
    ])

    return "\n".join(lines)


def _build_attack_surface_context(state: ScanState, limit: int = 25) -> str:
    """Format top attack surface entries as text for the LLM context."""
    if not state.attack_surface:
        return ""

    surface_lines: list[str] = []
    for entry in state.attack_surface[:limit]:
        if _is_auth_endpoint_false_positive(entry):
            continue
        ep = entry.endpoint
        auth = _format_auth_status(ep) or "auth_unknown"
        params = ", ".join(ep.parameters) if ep.parameters else "none"
        resp_fields = ", ".join(ep.response_fields) if ep.response_fields else "none"
        tests = " | ".join(entry.suggested_tests)

        block = (
            f"#{entry.priority_rank} [{entry.risk_level.value.upper()}] "
            f"{ep.method} {ep.url}\n"
            f"  Category: {entry.category.value} | Auth: {auth}\n"
            f"  Params: {params}\n"
            f"  Response fields: {resp_fields}"
        )

        if entry.vuln_indicators:
            block += "\n  Indicators:\n" + _format_indicator_lines(
                entry.vuln_indicators
            )

        block += f"\n  Rationale: {entry.rationale}"
        block += f"\n  Tests: {tests or 'none'}"
        surface_lines.append(block)

    total = len(state.attack_surface)
    header = f"ATTACK SURFACE (top {min(limit, total)} of {total}):"
    return header + "\n" + "\n".join(surface_lines)


def _build_vuln_pattern_summary(state: ScanState) -> str:
    """Summarise vulnerability indicator distribution."""
    pattern_counts: dict[str, int] = {}
    suppressed_count = 0
    llm_enhanced_count = 0
    all_indicators = itertools.chain.from_iterable(state.vuln_indicators.values())
    for ind in all_indicators:
        if ind.suppressed:
            suppressed_count += 1
            continue
        pattern_counts[ind.pattern.value] = (
            pattern_counts.get(ind.pattern.value, 0) + 1
        )
        if ind.llm_enhanced:
            llm_enhanced_count += 1

    distribution = ", ".join(
        f"{v} {k}"
        for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1])
    )
    return (
        f"VULNERABILITY PATTERN SUMMARY:\n"
        f"Total active indicators: {sum(pattern_counts.values())}\n"
        f"Suppressed (false positives): {suppressed_count}\n"
        f"LLM-enhanced: {llm_enhanced_count}\n"
        f"Distribution: {distribution}"
    )


def _insert_section_before_anchor(
    report: str,
    header: str,
    body: str,
    anchors: tuple[str, ...],
) -> str:
    """Insert a new section before the first matching anchor, or append at end."""
    for anchor in anchors:
        pos = report.find(anchor)
        if pos != -1:
            return report[:pos] + f"{header}\n\n{body}\n\n" + report[pos:]
    return report + f"\n\n{header}\n\n{body}\n"


def _splice_deterministic_table(report: str, state: ScanState) -> str:
    """Replace any LLM-generated attack surface table with the deterministic one."""
    table = _build_attack_surface_table(state)
    if not table:
        return report

    marker = "## Attack Surface Map"
    idx = report.find(marker)

    if idx == -1:
        return _insert_section_before_anchor(
            report, marker, table,
            anchors=("## Recommendations", "## Vulnerability Pattern"),
        )

    end_of_header = report.find("\n", idx)
    if end_of_header == -1:
        end_of_header = len(report)

    next_section = report.find("\n## ", end_of_header)
    if next_section == -1:
        return report[:end_of_header] + f"\n\n{table}\n"

    return (
        report[:end_of_header]
        + f"\n\n{table}\n"
        + report[next_section:]
    )


_TAIL_SECTION_MARKERS = (
    "## Attack Surface Map",
    "## Vulnerability Pattern",
    "## Recommendations",
)


def _find_tail_section(report: str) -> int:
    """Find the position of the first tail section marker, or -1."""
    for marker in _TAIL_SECTION_MARKERS:
        idx = report.find(marker)
        if idx != -1:
            return idx
    return -1


def _augment_report(report: str, state: ScanState) -> str:
    """Post-LLM quality fixes: deterministic table, missing findings, base URL."""
    report = _append_missing_findings(report, state)
    report = _splice_deterministic_table(report, state)
    report = _ensure_base_url(report, state)
    return report


_SEVERITY_RANK = {
    RiskLevel.CRITICAL: 0,
    RiskLevel.HIGH: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.LOW: 3,
    RiskLevel.INFO: 4,
}

class _ExpectedBehaviorRule(NamedTuple):
    finding_types: tuple[str, ...]
    path_keywords: tuple[str, ...]
    reason: str


_EXPECTED_BEHAVIOR_RULES: tuple[_ExpectedBehaviorRule, ...] = (
    _ExpectedBehaviorRule(
        finding_types=("excessive_data_exposure", "vuln_excessive_data_exposure"),
        path_keywords=("/login", "/auth", "/token", "/signin", "/oauth"),
        reason="auth endpoints return tokens by design",
    ),
    _ExpectedBehaviorRule(
        finding_types=("info_disclosure", "vuln_info_disclosure"),
        path_keywords=("/swagger", "/api-docs", "/redoc", "/docs", "/openapi"),
        reason="documentation endpoints expose API specs by design",
    ),
    _ExpectedBehaviorRule(
        finding_types=("bola_idor", "vuln_bola_idor"),
        path_keywords=("/products", "/posts", "/articles", "/categories", "/public"),
        reason="public resource listing is expected behavior",
    ),
    _ExpectedBehaviorRule(
        finding_types=("mass_assignment", "vuln_mass_assignment"),
        path_keywords=("/register", "/signup", "/create-account"),
        reason="registration endpoints accept user-provided fields by design",
    ),
)


def _is_expected_behavior(f: Finding) -> bool:
    """True if the finding matches a known false-positive pattern (e.g. tokens in /login)."""
    title_lower = f.title.lower()
    detail_lower = f.detail.lower()
    for rule in _EXPECTED_BEHAVIOR_RULES:
        if f.finding_type not in rule.finding_types:
            continue
        if any(kw in title_lower or kw in detail_lower for kw in rule.path_keywords):
            return True
    return False


_DENIAL_PHRASES: tuple[str, ...] = (
    "access denied",
    "permission denied",
    "not authorized",
    "unauthorized access",
    "forbidden",
    "authentication required",
    "login required",
    "insufficient permissions",
    "insufficient privileges",
    "loopback only",
    "internal use only",
    "internal resource",
    "not allowed",
    "request denied",
    "ip not allowed",
    "ip not whitelisted",
)


def _is_response_denial(f: Finding, state: ScanState) -> bool:
    """True if the endpoint's actual response shows it denied access."""
    if f.validated is True:
        return False

    ep = _lookup_endpoint_from_finding(f, state)
    if ep is None:
        return False

    # status code >= 400 means server denied the request
    if ep.status_code is not None and ep.status_code >= 400:
        return True

    # catches the HTTP 200 + JSON error body anti-pattern
    snippet = ep.response_body_snippet
    if not snippet:
        return False

    snippet_lower = snippet.lower()
    return any(phrase in snippet_lower for phrase in _DENIAL_PHRASES)


def _is_false_positive(f: Finding, state: ScanState) -> bool:
    return _is_expected_behavior(f) or _is_response_denial(f, state)


def _confidence_summary(findings: list[Finding]) -> str:
    """Return a confidence or verification line from the first finding with status info."""
    for f in findings:
        if f.validated is True:
            note = f" — {f.validation_evidence}" if f.validation_evidence else ""
            return f"**Confidence:** Confirmed [VALIDATED]{note}"
        if f.validated is False:
            note = f" — {f.verification_note}" if f.verification_note else ""
            return f"**Confidence:** Refuted [REFUTED]{note}"
        if f.verification_status:
            note = f" — {f.verification_note}" if f.verification_note else ""
            return f"**Verification:** {f.verification_status}{note}"
    return ""


def _expand_endpoint_lists(details: list[str]) -> list[str]:
    """Turn 'Endpoints: url1, url2, ...' into a bulleted list."""
    result: list[str] = []
    for detail in details:
        if "Endpoints: " not in detail:
            result.append(detail)
            continue
        before, _, url_csv = detail.partition("Endpoints: ")
        urls = [u.strip() for u in url_csv.split(", ") if u.strip()]
        if len(urls) < 3:
            result.append(detail)
            continue
        result.append(before.rstrip())
        result.append("")
        result.append(f"**Affected endpoints** ({len(urls)}):")
        result.extend(f"- {u}" for u in urls)
    return result


def _format_finding_group(
    ep_key: str, findings: list[Finding], state: ScanState, index: int = 0,
) -> str:
    """Format a group of findings sharing one endpoint as a markdown subsection."""
    worst = min(
        findings,
        key=lambda x: _SEVERITY_RANK.get(x.severity, len(_SEVERITY_RANK)),
    )
    sev = worst.severity.value.upper()

    types = list(dict.fromkeys(f.finding_type for f in findings))
    type_label = ", ".join(
        t.removeprefix("vuln_").replace("_", " ") for t in types
    )

    prefix = f"{index}. " if index else ""
    lines: list[str] = [f"#### {prefix}{type_label.upper()} — {ep_key} **[{sev}]**"]

    cwe_parts = list(dict.fromkeys(
        cwe for f in findings if (cwe := _lookup_cwe_for_finding(f))
    ))
    if cwe_parts:
        lines.append(f"**CWE:** {'; '.join(cwe_parts)}")

    ep = _lookup_endpoint_from_finding(worst, state)
    if ep is not None:
        lines.append(f"**Endpoint:** {ep.method} {ep.url}")
        if ep.parameters:
            lines.append(f"**Parameters:** {', '.join(ep.parameters)}")
        auth = _format_auth_status(ep)
        if auth:
            lines.append(f"**Auth:** {auth}")
        if ep.response_body_snippet:
            snippet = ep.response_body_snippet
            limit = (
                _SNIPPET_MAX_CHARS_INFO_DISCLOSURE
                if any("info_disclosure" in t for t in types)
                else _SNIPPET_MAX_CHARS
            )
            if len(snippet) > limit:
                snippet = snippet[:limit] + "..."
            lines.extend(("", "**Response:**", f"`{snippet}`"))

    details = list(dict.fromkeys(f.detail for f in findings if f.detail))
    if details:
        lines.extend(("", "**Detail:**"))
        lines.extend(_expand_endpoint_lists(details))

    evidences = list(dict.fromkeys(
        f.evidence for f in findings if f.evidence
    ))
    if evidences:
        lines.extend(("", "**Evidence:**", "; ".join(evidences)))

    confidence = _confidence_summary(findings)
    if confidence:
        lines.extend(("", confidence))

    all_tests = list(dict.fromkeys(
        t for f in findings for t in _lookup_tests_for_finding(f, state)
    ))
    if all_tests:
        lines.append("")
        lines.append("**Test Commands:**")
        for t in all_tests:
            lines.append(f"```\n{t}\n```")

    lines.extend(("", "---", ""))
    return "\n".join(lines)


def _append_missing_findings(report: str, state: ScanState) -> str:
    """Insert missing critical/high findings grouped by endpoint.

    Multiple findings for the same endpoint are merged into one entry.
    The section is placed right after the LLM's Critical & High Findings
    section (before Attack Surface Map) to keep all findings together.
    """
    critical_high = [
        f for f in state.findings
        if f.severity in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        and not _is_false_positive(f, state)
    ]
    if not critical_high:
        return report

    cut = _find_tail_section(report)
    check_region = report[:cut] if cut != -1 else report

    missing: list[Finding] = []
    for f in critical_high:
        ep = _lookup_endpoint_from_finding(f, state)
        url = ep.url if ep else ""
        if f.title not in check_region and (not url or url not in check_region):
            missing.append(f)

    if not missing:
        return report

    # group by endpoint so each URL appears once
    grouped: dict[str, list[Finding]] = {}
    for f in missing:
        ep = _lookup_endpoint_from_finding(f, state)
        key = f"{ep.method} {ep.url}" if ep else f.title
        grouped.setdefault(key, []).append(f)

    section = "\n\n### Additional Critical & High Findings\n\n"
    section += "*The following findings were not fully detailed above:*\n\n"

    for idx, (ep_key, findings) in enumerate(grouped.items(), start=1):
        section += _format_finding_group(
            ep_key=ep_key, findings=findings, state=state, index=idx,
        )

    insert_before = _find_tail_section(report)

    if insert_before != -1:
        report = report[:insert_before] + section + "\n" + report[insert_before:]
    else:
        report += section

    return report


def _ensure_base_url(report: str, state: ScanState) -> str:
    """Add base URL near the top of the report if it's not already present."""
    if state.base_url in report:
        return report

    exec_idx = report.find("## Executive Summary")
    if exec_idx != -1:
        end_of_line = report.find("\n", exec_idx)
        if end_of_line != -1:
            report = (
                report[:end_of_line + 1]
                + f"\n**Base URL:** {state.base_url}\n"
                + report[end_of_line + 1:]
            )
            return report

    return f"**Base URL:** {state.base_url}\n\n" + report


def _build_report_context(state: ScanState, duration: float) -> str:
    """Build the full scan context for the LLM report synthesis prompt."""
    sections: list[str] = []

    sections.append(
        f"SCAN ID: {state.scan_id}\n"
        f"TARGET: {state.target}\n"
        f"BASE URL: {state.base_url}\n"
        f"DURATION: {duration:.1f}s\n"
        f"TECHNIQUES USED: {', '.join(_AGENT_DISPLAY_NAMES.get(a, a) for a in state.agents_completed)}\n"
        f"TOTAL ENDPOINTS: {len(state.endpoints)}\n"
        f"TOTAL FINDINGS: {len(state.findings)}\n"
        f"ATTACK SURFACE ENTRIES: {len(state.attack_surface)}"
    )

    strategy = _build_strategy_section(state)
    if strategy:
        sections.append(strategy)

    tech_block = _build_tech_fingerprint_section(state)
    if tech_block:
        sections.append("TECH FINGERPRINT:\n" + tech_block)

    sections.append(_FINDING_EXAMPLE)

    enriched = _build_enriched_findings(state)
    if enriched:
        sections.append(enriched)

    chains = _build_chain_analysis(state)
    if chains:
        sections.append(chains)

    surface = _build_attack_surface_context(state)
    if surface:
        sections.append(surface)

    sections.append(_build_vuln_pattern_summary(state))
    sections.append(_build_cwe_reference_block())
    sections.append(_build_section_instructions(state))

    return "\n\n---\n\n".join(sections)


def _build_fallback_report(state: ScanState, duration: float) -> str:
    """Fallback report for when LLM synthesis fails."""
    parts = [
        f"# Ghost Hunter — Scan Report\n\n"
        f"**Target:** {state.target}  \n"
        f"**Base URL:** {state.base_url}  \n"
        f"**Duration:** {duration:.1f}s  \n"
        f"**Endpoints:** {len(state.endpoints)}  \n"
        f"**Findings:** {len(state.findings)}  \n\n"
        f"*AI report synthesis failed. Showing raw findings and attack surface.*\n"
    ]

    tech_block = _build_tech_fingerprint_section(state)
    if tech_block:
        parts.append(f"\n## Target Profile\n\n{tech_block}\n")

    critical_high = [
        f for f in state.findings
        if f.severity in (RiskLevel.CRITICAL, RiskLevel.HIGH)
    ]
    if critical_high:
        parts.append("\n## Critical & High Findings\n")
        for f in critical_high:
            cwe = _lookup_cwe_for_finding(f)
            cwe_text = f" | {cwe}" if cwe else ""
            parts.append(
                f"\n- **[{f.severity.value.upper()}]** {f.title}{cwe_text}: {f.detail}"
            )
            if f.evidence:
                parts.append(f"  Evidence: {f.evidence}")
            tests = _lookup_tests_for_finding(f, state)
            if tests:
                parts.append(f"  Tests: {' | '.join(tests)}")

    surface_table = _build_attack_surface_table(state)
    if surface_table:
        parts.append(f"\n\n## Attack Surface Map\n\n{surface_table}\n")

    return "\n".join(parts)


async def generate_report(
    state: ScanState,
    llm_client: LLMClient,
    duration: float,
    prompt_registry: PromptRegistry | None = None,
) -> str:
    """Generate a markdown pentest report via LLM and write it to disk.

    Returns the output file path.
    """
    context = _build_report_context(state, duration)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    filename = str(output_dir / f"ghost_hunter_report_{state.scan_id}.md")

    max_tokens = _compute_max_tokens(len(state.findings))

    registry = prompt_registry or PromptRegistry()
    report_prompt = registry.get("report").system_prompt

    messages = [
        {"role": "system", "content": report_prompt},
        {
            "role": "user",
            "content": (
                f"Generate a penetration test report for the following scan:\n\n"
                f"{context}"
            ),
        },
    ]

    try:
        response = await llm_client.chat(
            messages=messages,
            name="report_synthesis",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        report = _augment_report(response.content, state)
        Path(filename).write_text(report)
        logger.info("Report written to %s", filename)
        return filename

    except Exception as e:
        logger.error("Report generation failed: %s", e)
        Path(filename).write_text(_build_fallback_report(state, duration))
        return filename
