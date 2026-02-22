"""LLM-synthesized penetration test report from completed scan state."""

from __future__ import annotations

import logging
from pathlib import Path

from src.ghost_hunter.clients.llm import LLMClient
from src.ghost_hunter.models import Finding, RiskLevel, ScanState

logger = logging.getLogger(__name__)

# keep report context within LLM token limits
_MAX_FINDINGS_IN_CONTEXT = 50
_MAX_ATTACK_SURFACE_ENTRIES = 20
_MAX_VULN_INDICATORS_PER_ENTRY = 5
_MAX_SUGGESTED_TESTS_PER_ENTRY = 2
_MAX_FALLBACK_FINDINGS = 20

REPORT_SYSTEM_PROMPT = """\
CONTEXT:
You are synthesizing a professional penetration test report from the results of \
an automated AI-driven security assessment. The assessment was performed by Ghost Hunter, \
a multi-agent system that combines deterministic pattern matching with LLM-powered \
semantic analysis to discover, classify, and validate API vulnerabilities.

ROLE:
Senior penetration tester writing a report for both technical and executive audiences. \
You follow industry conventions (OWASP, PTES) and write with the authority of someone \
who has validated these findings through active probing.

ACTION:
Generate a complete markdown penetration test report with these sections:

1. **Executive Summary** — 3-5 sentences: scope, key findings count by severity, overall \
risk posture, most critical discovery. Written for a CISO or VP of Engineering.

2. **Methodology** — Which agents ran, what techniques were used (passive recon, crawling, \
API spec parsing, JS analysis, LLM hypothesis, classification, two-pass vuln analysis, \
active probe validation). Emphasize the AI-driven pipeline approach.

3. **Target Profile** — Technology fingerprint, frameworks detected, security headers \
present/missing, overall security posture assessment.

4. **Critical & High Findings** — Each finding with:
   - Title and severity badge
   - Endpoint(s) affected
   - Evidence (from deterministic checks, LLM analysis, or active probes)
   - Impact description
   - Remediation recommendation
   Group validated findings (from probe_validator) prominently.

5. **Attack Surface Map** — Top-20 prioritized endpoints with risk level, category, \
and vulnerability indicators. Format as a markdown table.

6. **Vulnerability Pattern Analysis** — Distribution of vulnerability types, chained \
vulnerabilities, LLM-enhanced findings vs deterministic, suppressed false positives.

7. **Recommendations** — Prioritized remediation roadmap: immediate (critical), short-term \
(high), medium-term (medium), and long-term improvements.

FORMAT:
Output raw markdown. Use ## for sections, ### for subsections. Use tables, bullet lists, \
and code blocks where appropriate. Include severity badges like **[CRITICAL]**, **[HIGH]**, \
**[MEDIUM]**, **[LOW]**, **[INFO]**.

TONE:
Professional, precise, actionable. No hedging — state findings with confidence based on \
the evidence. Reference specific URLs, parameters, and response data.
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

    # tech fingerprint
    fp = state.tech_fingerprint
    tech_lines = []
    if fp.server:
        tech_lines.append(f"Server: {fp.server}")
    if fp.frameworks:
        tech_lines.append(f"Frameworks: {', '.join(fp.frameworks)}")
    if fp.technologies:
        tech_lines.append(f"Technologies: {', '.join(fp.technologies)}")
    if fp.security_headers:
        tech_lines.append(f"Security headers: {', '.join(fp.security_headers.keys())}")
    if fp.missing_security_headers:
        tech_lines.append(f"Missing headers: {', '.join(fp.missing_security_headers)}")
    if fp.cookies:
        tech_lines.append(f"Cookies: {', '.join(fp.cookies)}")
    if tech_lines:
        sections.append("TECH FINGERPRINT:\n" + "\n".join(tech_lines))

    # findings
    findings_lines = [_format_finding(f) for f in state.findings[:_MAX_FINDINGS_IN_CONTEXT]]
    if findings_lines:
        sections.append("ALL FINDINGS:\n" + "\n".join(findings_lines))

    # attack surface
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

    # vuln indicator summary
    pattern_counts: dict[str, int] = {}
    suppressed_count = 0
    llm_enhanced_count = 0
    for indicators in state.vuln_indicators.values():
        for ind in indicators:
            if ind.suppressed:
                suppressed_count += 1
                continue
            pattern_counts[ind.pattern.value] = pattern_counts.get(ind.pattern.value, 0) + 1
            if ind.llm_enhanced:
                llm_enhanced_count += 1

    validated_count = sum(1 for f in state.findings if f.validated is True)

    distribution = ", ".join(
        f"{v} {k}" for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1])
    )
    sections.append(
        f"VULNERABILITY PATTERN SUMMARY:\n"
        f"Total active indicators: {sum(pattern_counts.values())}\n"
        f"Suppressed (false positives): {suppressed_count}\n"
        f"LLM-enhanced: {llm_enhanced_count}\n"
        f"Probe-validated: {validated_count}\n"
        f"Distribution: {distribution}"
    )

    return "\n\n---\n\n".join(sections)


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

    messages = [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Generate a professional penetration test report for the following scan:\n\n"
                f"{context}"
            ),
        },
    ]

    try:
        response = await llm_client.chat(
            messages=messages,
            name="report_synthesis",
            max_tokens=4096,
            temperature=0.3,
        )
        report_content = response.choices[0].message.content

        header = (
            f"# Ghost Hunter — Penetration Test Report\n\n"
            f"**Scan ID:** {state.scan_id}  \n"
            f"**Target:** {state.target}  \n"
            f"**Base URL:** {state.base_url}  \n"
            f"**Duration:** {duration:.1f}s  \n"
            f"**Agents:** {len(state.agents_completed)}  \n"
            f"**Endpoints:** {len(state.endpoints)}  \n"
            f"**Findings:** {len(state.findings)}  \n\n"
            f"---\n\n"
        )

        Path(filename).write_text(header + report_content)
        logger.info("Report written to %s", filename)
        return filename

    except Exception as e:
        logger.error("Report generation failed: %s", e)
        critical_findings = [
            f for f in state.findings
            if f.severity in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        ]
        fallback = (
            f"# Ghost Hunter — Scan Report\n\n"
            f"**Target:** {state.target}  \n"
            f"**Duration:** {duration:.1f}s  \n\n"
            f"*AI report synthesis failed. Showing raw critical/high findings:*\n\n"
        )
        for f in critical_findings[:_MAX_FALLBACK_FINDINGS]:
            fallback += f"- **[{f.severity.value.upper()}]** {f.title}: {f.detail}\n"
        Path(filename).write_text(fallback)
        return filename
