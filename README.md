# Ghost Hunter

AI-powered API endpoint discovery and attack surface mapping tool. Given any target domain, Ghost Hunter discovers API endpoints, classifies them, and produces a prioritized attack surface map using LLM-driven agents orchestrated by Groq.

**Discovery and recon only — no exploitation.**

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (get one at https://console.groq.com)

# 3. Run a scan
uv run ghost_hunter.py <target>
```

## Usage

```bash
# Basic scan
uv run ghost_hunter.py vulnbank.org

# With options
uv run ghost_hunter.py vulnbank.org --rate-limit 2.0 --max-depth 5 --max-pages 200

# Verbose logging
uv run ghost_hunter.py vulnbank.org -v

# With proxy
uv run ghost_hunter.py vulnbank.org --proxy socks5://127.0.0.1:9050
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--rate-limit` | Max HTTP requests per second | 5.0 |
| `--max-depth` | Max crawl depth | 3 |
| `--max-pages` | Max pages to crawl | 100 |
| `--proxy` | HTTP/SOCKS5 proxy URL | None |
| `-v, --verbose` | Enable debug logging | Off |

## Output

- **Console**: Rich-formatted attack surface table with priority rankings
- **JSON report**: `ghost_hunter_report_<scan_id>.json` with full structured results
- **Langfuse**: Optional LLM observability traces (configure keys in `.env`)

## Architecture

The tool runs 7 specialized agents orchestrated by an LLM planner:

| Agent | Type | Purpose |
|-------|------|---------|
| PassiveRecon | Deterministic | robots.txt, sitemap, headers, security.txt |
| WebCrawler | Deterministic | BFS crawl for links, forms, scripts |
| APIDiscovery | Hybrid | OpenAPI specs, common paths, version enum, LLM guessing |
| JSAnalyzer | Deterministic | Regex extraction of API routes from JS bundles |
| Hypothesis | LLM | Hypothesizes undiscovered endpoints, validates with HEAD requests |
| Classifier | LLM | Categorizes endpoints by type and auth requirements |
| Prioritizer | LLM | Ranks attack surface with rationale and suggested tests |

## Project Structure

```
ghost_hunter.py              # CLI entry point
src/ghost_hunter/
├── config.py                # Settings from environment variables
├── orchestrator.py          # LLM tool-use loop coordinating agents
├── output.py                # Rich console output + JSON reports
├── clients/                 # HTTP, LLM, and tracing clients
│   ├── http.py              # Adaptive rate-limited httpx client
│   ├── llm.py               # Groq wrapper with retry + tracing
│   └── tracing.py           # Langfuse initialization and helpers
├── models/                  # Pydantic data models
│   ├── enums.py             # DiscoverySource, RiskLevel, EndpointCategory
│   ├── endpoints.py         # Endpoint, Finding, TechFingerprint
│   ├── scan.py              # ScanState, AgentResult
│   └── attack_surface.py    # AttackSurfaceEntry
└── agents/                  # Specialized discovery agents
    ├── base.py              # BaseAgent ABC with auto-tracing
    ├── passive_recon.py
    ├── web_crawler.py
    ├── api_discovery.py
    ├── js_analyzer.py
    ├── hypothesis.py
    ├── classifier.py
    └── prioritizer.py
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Groq API key (free tier works)
