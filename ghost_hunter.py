#!/usr/bin/env python3
"""Ghost Hunter — AI-powered API endpoint discovery and attack surface mapping.

Usage:
    uv run ghost_hunter.py <target>
    uv run ghost_hunter.py vulnbank.org --rate-limit 2.0
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone, timedelta

import click

from src.ghost_hunter.clients import AdaptiveHttpClient, LLMClient, create_trace, flush_langfuse, init_langfuse
from src.ghost_hunter.config import settings
from src.ghost_hunter.models import ScanState
from src.ghost_hunter.orchestrator import Orchestrator
from src.ghost_hunter.output import (
    console,
    print_attack_surface,
    print_banner,
    print_report_path,
    print_summary,
    write_json_report,
)
from src.ghost_hunter.report import generate_report


def _normalize_target(target: str) -> tuple[str, str]:
    """Return (display_target, base_url) from user input."""
    target = target.strip().rstrip("/")
    if target.startswith(("http://", "https://")):
        base_url = target
        display = target.split("//", 1)[1]
    else:
        display = target
        base_url = f"https://{target}"
    return display, base_url


async def _run_scan(target: str, rate_limit: float | None, proxy: str | None) -> None:
    display, base_url = _normalize_target(target)

    if rate_limit is not None:
        settings.default_rate_limit = rate_limit
    if proxy is not None:
        settings.proxy = proxy

    print_banner(display)

    if not settings.llm_api_key and not settings.groq_api_key:
        console.print("[red]Error: LLM_API_KEY (or GROQ_API_KEY) not set. Copy .env.example to .env and add your key.[/red]")
        sys.exit(1)

    init_langfuse()

    state = ScanState(target=display, base_url=base_url)

    vienna_tz = timezone(timedelta(hours=1))
    timestamp = datetime.now(tz=vienna_tz).strftime("%d%m%Y-%H:%M")
    create_trace(
        name=f"{display}_{timestamp}",
        session_id=state.scan_id,
        metadata={"target": display, "base_url": base_url},
    )

    http_client = AdaptiveHttpClient(base_url=base_url, rate_limit=rate_limit, proxy=proxy)
    llm_client = LLMClient()

    console.print("[bold]Starting scan...[/bold]\n")
    start = time.monotonic()

    try:
        orchestrator = Orchestrator(
            http_client=http_client,
            llm_client=llm_client,
            state=state,
        )
        state = await orchestrator.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Scan interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Scan error: {e}[/red]")
        logging.exception("Scan failed")
    finally:
        await http_client.close()

    duration = time.monotonic() - start

    print_attack_surface(state)
    print_summary(state, duration)
    write_json_report(state, duration)

    report_path = await generate_report(
        state=state, llm_client=llm_client, duration=duration,
        prompt_registry=orchestrator.prompt_registry,
    )
    print_report_path(report_path)

    flush_langfuse()


@click.command()
@click.argument("target")
@click.option("--rate-limit", type=float, default=None, help="Max requests per second (default: 5.0)")
@click.option("--proxy", type=str, default=None, help="HTTP/SOCKS5 proxy URL")
@click.option("--max-depth", type=int, default=None, help="Max crawl depth (default: 3)")
@click.option("--max-pages", type=int, default=None, help="Max pages to crawl (default: 100)")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def main(
    target: str,
    rate_limit: float | None,
    proxy: str | None,
    max_depth: int | None,
    max_pages: int | None,
    verbose: bool,
) -> None:
    """Ghost Hunter — discover API endpoints and map the attack surface of TARGET."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if verbose:
        logging.getLogger("src.ghost_hunter").setLevel(logging.DEBUG)

    if max_depth is not None:
        settings.max_crawl_depth = max_depth
    if max_pages is not None:
        settings.max_pages = max_pages

    asyncio.run(_run_scan(target, rate_limit, proxy))


if __name__ == "__main__":
    main()
