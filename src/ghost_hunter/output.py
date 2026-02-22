"""Rich console output and JSON report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.ghost_hunter.models import AgentResult, RiskLevel, ScanState

console = Console()

RISK_COLORS = {
    RiskLevel.CRITICAL: "bold red",
    RiskLevel.HIGH: "red",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.LOW: "cyan",
    RiskLevel.INFO: "dim",
}

BANNER = r"""
   _____ _               _     _   _             _
  / ____| |             | |   | | | |           | |
 | |  __| |__   ___  ___| |_  | |_| |_   _ _ __ | |_ ___ _ __
 | | |_ | '_ \ / _ \/ __| __| |  _  | | | | '_ \| __/ _ \ '__|
 | |__| | | | | (_) \__ \ |_  | | | | |_| | | | | ||  __/ |
  \_____|_| |_|\___/|___/\__| |_| |_|\__,_|_| |_|\__\___|_|
"""


def print_banner(target: str) -> None:
    console.print(Text(BANNER, style="bold cyan"))
    console.print(
        Panel(
            f"[bold]Target:[/bold] {target}\n"
            f"[bold]Mode:[/bold] Discovery & Recon Only",
            title="Ghost Hunter",
            border_style="cyan",
        )
    )
    console.print()


def print_agent_step(agent_name: str, reason: str, result: AgentResult) -> None:
    status = "[green]OK[/green]" if result.success else "[red]FAIL[/red]"
    console.print(
        f"  [{status}] [bold]{agent_name}[/bold] "
        f"({result.duration_seconds:.1f}s) — "
        f"{len(result.endpoints_found)} endpoints, "
        f"{len(result.findings)} findings"
    )
    if reason:
        console.print(f"       [dim]{reason}[/dim]")
    for error in result.errors:
        console.print(f"       [red]Error: {error}[/red]")


def print_attack_surface(state: ScanState) -> None:
    if not state.attack_surface:
        console.print("\n[yellow]No attack surface entries to display.[/yellow]")
        return

    console.print()
    table = Table(
        title="Attack Surface Map",
        title_style="bold cyan",
        show_lines=True,
        expand=True,
    )
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Risk", width=8)
    table.add_column("Method", width=7)
    table.add_column("Endpoint", min_width=30)
    table.add_column("Category", width=15)
    table.add_column("Vuln Patterns", width=18)
    table.add_column("Rationale", min_width=25)
    table.add_column("Suggested Tests", min_width=20)

    for entry in state.attack_surface:
        risk_style = RISK_COLORS.get(entry.risk_level, "")
        tests = "\n".join(f"- {t}" for t in entry.suggested_tests[:3])
        vulns = "\n".join(
            f"- {v.pattern.value}" for v in entry.vuln_indicators[:3]
        )

        table.add_row(
            str(entry.priority_rank),
            Text(entry.risk_level.value.upper(), style=risk_style),
            entry.endpoint.method,
            entry.endpoint.url,
            entry.category.value,
            vulns or "-",
            entry.rationale[:100],
            tests or "-",
        )

    console.print(table)


def _build_vuln_summary(state: ScanState) -> dict[str, int]:
    counts: dict[str, int] = {}
    for indicators in state.vuln_indicators.values():
        for ind in indicators:
            counts[ind.pattern.value] = counts.get(ind.pattern.value, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def print_summary(state: ScanState, duration: float) -> None:
    console.print()

    sources: dict[str, int] = {}
    for ep in state.endpoints.values():
        sources[ep.discovered_by.value] = sources.get(ep.discovered_by.value, 0) + 1

    source_lines = "\n".join(
        f"  {v:>3} from {k}" for k, v in sorted(sources.items(), key=lambda x: -x[1])
    )

    risk_dist: dict[str, int] = {}
    for entry in state.attack_surface:
        risk_dist[entry.risk_level.value] = risk_dist.get(entry.risk_level.value, 0) + 1
    risk_lines = ", ".join(f"{v} {k}" for k, v in risk_dist.items())

    vuln_dist = _build_vuln_summary(state)
    vuln_lines = ", ".join(f"{v} {k}" for k, v in vuln_dist.items())
    vuln_ep_count = len(state.vuln_indicators)

    summary_text = (
        f"[bold]Total endpoints:[/bold] {len(state.endpoints)}\n"
        f"[bold]Attack surface entries:[/bold] {len(state.attack_surface)}\n"
        f"[bold]Findings:[/bold] {len(state.findings)}\n"
        f"[bold]Duration:[/bold] {duration:.1f}s\n\n"
        f"[bold]Endpoints by source:[/bold]\n{source_lines}\n\n"
        f"[bold]Risk distribution:[/bold] {risk_lines or 'N/A'}"
    )

    if vuln_lines:
        summary_text += (
            f"\n\n[bold]Vulnerability patterns:[/bold] "
            f"{vuln_ep_count} endpoints flagged\n  {vuln_lines}"
        )

    console.print(
        Panel(
            summary_text,
            title="Scan Summary",
            border_style="green",
        )
    )


def write_json_report(state: ScanState, duration: float) -> str:
    """Write full JSON report and return the file path."""
    report: dict[str, Any] = {
        "scan_id": state.scan_id,
        "target": state.target,
        "base_url": state.base_url,
        "duration_seconds": round(duration, 2),
        "summary": {
            "total_endpoints": len(state.endpoints),
            "total_findings": len(state.findings),
            "attack_surface_entries": len(state.attack_surface),
            "agents_completed": state.agents_completed,
        },
        "tech_fingerprint": state.tech_fingerprint.model_dump(),
        "endpoints": [
            ep.model_dump() for ep in state.endpoints.values()
        ],
        "findings": [f.model_dump() for f in state.findings],
        "attack_surface": [
            {
                "priority_rank": e.priority_rank,
                "risk_level": e.risk_level.value,
                "category": e.category.value,
                "url": e.endpoint.url,
                "method": e.endpoint.method,
                "rationale": e.rationale,
                "suggested_tests": e.suggested_tests,
                "vuln_indicators": [
                    {
                        "pattern": v.pattern.value,
                        "confidence": v.confidence.value,
                        "evidence": v.evidence,
                        "description": v.description,
                    }
                    for v in e.vuln_indicators
                ],
            }
            for e in state.attack_surface
        ],
        "vuln_pattern_summary": _build_vuln_summary(state),
        "blocked_paths": state.blocked_paths,
        "errors": state.errors,
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    filename = str(output_dir / f"ghost_hunter_report_{state.scan_id}.json")
    Path(filename).write_text(json.dumps(report, indent=2, default=str))
    console.print(f"\n[green]JSON report written to {filename}[/green]")
    return filename


def print_report_path(report_path: str) -> None:
    """Print the path to the generated AI pentest report."""
    console.print(f"[green bold]AI Pentest Report written to {report_path}[/green bold]")
