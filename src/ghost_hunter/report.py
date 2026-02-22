"""LLM-synthesized penetration test report from completed scan state."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

from src.ghost_hunter.clients.llm import LLMClient
from src.ghost_hunter.models import Finding, RiskLevel, ScanState

logger = logging.getLogger(__name__)

_MAX_FINDINGS_IN_CONTEXT = 50
_MAX_ATTACK_SURFACE_ENTRIES = 20
_MAX_VULN_INDICATORS_PER_ENTRY = 5
_MAX_SUGGESTED_TESTS_PER_ENTRY = 2
_MAX_FALLBACK_FINDINGS = 20

_TOKENS_PER_FINDING = 120
_BASE_REPORT_TOKENS = 3000
_MAX_REPORT_TOKENS = 8192

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
}

REPORT_SYSTEM_PROMPT = """\
CONTEXT:
You are synthesizing a penetration test report from the results of an automated \
security assessment. The assessment was performed by Ghost Hunter, a multi-agent \
system that combines deterministic pattern matching with LLM-powered semantic \
analysis to discover, classify, and analyze API vulnerabilities.

IMPORTANT — CONFIDENCE LEVELS:
All findings in this report are from the discovery and reconnaissance phase. \
No findings have been confirmed through active exploitation. When presenting \
findings, use these confidence labels:
- **Confirmed** — only if the finding data includes [VALIDATED]
- **Likely** — strong evidence from multiple signals (deterministic check + LLM analysis)
- **Suspected** — single signal or pattern match only
If a finding is marked [REFUTED], note it was disproven during validation.

ROLE:
Senior penetration tester writing a report for both technical and executive audiences. \
Follow OWASP and PTES conventions.

ACTION:
Generate a complete markdown report with these sections:

1. **Executive Summary** — 3-5 sentences: scope, findings count by severity, overall \
risk posture, most critical discovery.

2. **Scope & Limitations** — What was tested (passive recon, crawling, API spec parsing, \
JS analysis, pattern matching, LLM analysis). What was NOT tested: authenticated \
endpoint behavior, business logic beyond crawlable paths, rate limiting, DoS resilience. \
State that findings are discovery-phase and have not been confirmed through exploitation \
unless marked [VALIDATED].

3. **Methodology** — Which agents ran, what techniques were used. Briefly describe the \
multi-agent pipeline.

4. **Target Profile** — Technology fingerprint, frameworks detected, security headers \
present/missing, security posture assessment.

5. **Critical & High Findings** — Each finding with:
   - Title, severity badge, and CWE reference (use the CWE REFERENCE MAP provided)
   - Confidence level (Confirmed / Likely / Suspected)
   - Endpoint(s) affected
   - Evidence
   - Impact description
   - Remediation recommendation

6. **Attack Surface Map** — Top-20 prioritized endpoints as a markdown table with \
risk level, category, and vulnerability indicators.

7. **Vulnerability Pattern Analysis** — Distribution of vulnerability types, chained \
vulnerabilities, LLM-enhanced vs deterministic findings, suppressed false positives.

8. **Recommendations** — Prioritized remediation roadmap: immediate (critical), \
short-term (high), medium-term (medium), long-term improvements.

FORMAT:
Output raw markdown. Use ## for sections, ### for subsections. Use tables, bullet lists, \
and code blocks where appropriate. Include severity badges like **[CRITICAL]**, **[HIGH]**, \
**[MEDIUM]**, **[LOW]**, **[INFO]**.

TONE:
Precise and actionable. Reference specific URLs, parameters, and response data. \
Be clear about confidence levels — don't present pattern matches as confirmed exploits.
"""


def _format_finding(f: Finding) -> str:
    """Format a single finding as a context line for the LLM."""
    validated_tag = ""
    if f.validated is True:
        validated_tag = " [VALIDATED]"
    elif f.validated is False:
        validated_tag = " [REFUTED]"

    line = (
        f"- [{f.severity.value.upper()}]{validated_tag} {f.title}\n"
        f"  Detail: {f.detail}"
    )
    if f.evidence:
        line += f"\n  Evidence: {f.evidence}"
    return line


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


def _build_attack_surface_table(state: ScanState) -> str:
    """Format top attack surface entries as a markdown table."""
    if not state.attack_surface:
        return ""
    rows = [
        "| # | Risk | Method | URL | Category | Indicators |",
        "|----|------|--------|-----|----------|------------|",
    ]
    for entry in state.attack_surface[:_MAX_ATTACK_SURFACE_ENTRIES]:
        vulns = ", ".join(
            v.pattern.value for v in entry.vuln_indicators[:_MAX_VULN_INDICATORS_PER_ENTRY]
        )
        rows.append(
            f"| {entry.priority_rank} "
            f"| {entry.risk_level.value.upper()} "
            f"| {entry.endpoint.method} "
            f"| {entry.endpoint.url} "
            f"| {entry.category.value} "
            f"| {vulns or 'none'} |"
        )
    return "\n".join(rows)


def _build_report_context(state: ScanState, duration: float) -> str:
    """Build the full scan context for the LLM report synthesis prompt."""
    sections: list[str] = []

    sections.append(
        f"SCAN ID: {state.scan_id}\n"
        f"TARGET: {state.target}\n"
        f"BASE URL: {state.base_url}\n"
        f"DURATION: {duration:.1f}s\n"
        f"AGENTS COMPLETED: {', '.join(state.agents_completed)}\n"
        f"TOTAL ENDPOINTS: {len(state.endpoints)}\n"
        f"TOTAL FINDINGS: {len(state.findings)}\n"
        f"ATTACK SURFACE ENTRIES: {len(state.attack_surface)}"
    )

    tech_block = _build_tech_fingerprint_section(state)
    if tech_block:
        sections.append("TECH FINGERPRINT:\n" + tech_block)

    findings_lines = [_format_finding(f) for f in state.findings[:_MAX_FINDINGS_IN_CONTEXT]]
    if findings_lines:
        sections.append("ALL FINDINGS:\n" + "\n".join(findings_lines))

    surface_lines = []
    for entry in state.attack_surface[:_MAX_ATTACK_SURFACE_ENTRIES]:
        vulns = ", ".join(
            v.pattern.value for v in entry.vuln_indicators[:_MAX_VULN_INDICATORS_PER_ENTRY]
        )
        tests = " | ".join(entry.suggested_tests[:_MAX_SUGGESTED_TESTS_PER_ENTRY])
        surface_lines.append(
            f"#{entry.priority_rank} [{entry.risk_level.value.upper()}] "
            f"{entry.endpoint.method} {entry.endpoint.url}\n"
            f"  Category: {entry.category.value} | Vulns: {vulns or 'none'}\n"
            f"  Rationale: {entry.rationale}\n"
            f"  Tests: {tests or 'none'}"
        )
    if surface_lines:
        sections.append("ATTACK SURFACE (top 20):\n" + "\n".join(surface_lines))

    pattern_counts: dict[str, int] = {}
    suppressed_count = 0
    llm_enhanced_count = 0
    all_indicators = itertools.chain.from_iterable(state.vuln_indicators.values())
    for ind in all_indicators:
        if ind.suppressed:
            suppressed_count += 1
            continue
        pattern_counts[ind.pattern.value] = pattern_counts.get(ind.pattern.value, 0) + 1
        if ind.llm_enhanced:
            llm_enhanced_count += 1

    distribution = ", ".join(
        f"{v} {k}" for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1])
    )
    sections.append(
        f"VULNERABILITY PATTERN SUMMARY:\n"
        f"Total active indicators: {sum(pattern_counts.values())}\n"
        f"Suppressed (false positives): {suppressed_count}\n"
        f"LLM-enhanced: {llm_enhanced_count}\n"
        f"Distribution: {distribution}"
    )

    sections.append(_build_cwe_reference_block())

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
        for f in critical_high[:_MAX_FALLBACK_FINDINGS]:
            parts.append(f"\n- **[{f.severity.value.upper()}]** {f.title}: {f.detail}")
            if f.evidence:
                parts.append(f"  Evidence: {f.evidence}")

    surface_table = _build_attack_surface_table(state)
    if surface_table:
        parts.append(f"\n\n## Attack Surface Map\n\n{surface_table}\n")

    return "\n".join(parts)


async def generate_report(
    state: ScanState,
    llm_client: LLMClient,
    duration: float,
) -> str:
    """Generate a markdown pentest report via LLM and write it to disk.

    Returns the output file path.
    """
    context = _build_report_context(state, duration)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    filename = str(output_dir / f"ghost_hunter_report_{state.scan_id}.md")

    max_tokens = _compute_max_tokens(len(state.findings))

    messages = [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
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
            temperature=0.3,
        )

        Path(filename).write_text(response.content)
        logger.info("Report written to %s", filename)
        return filename

    except Exception as e:
        logger.error("Report generation failed: %s", e)
        Path(filename).write_text(_build_fallback_report(state, duration))
        return filename
