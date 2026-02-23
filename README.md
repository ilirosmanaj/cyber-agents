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
- **Markdown report**: LLM-generated pentest report following OWASP/PTES conventions
- **JSON report**: `ghost_hunter_report_<scan_id>.json` with full structured results
- **Langfuse**: Optional LLM observability traces (configure keys in `.env`)

## Architecture

The tool runs 9 specialized agents orchestrated by a DAG-based scheduler:

| Agent | Type | Purpose |
|-------|------|---------|
| PassiveRecon | Deterministic | robots.txt, sitemap, headers, security.txt |
| WebCrawler | Deterministic | BFS crawl for links, forms, scripts |
| APIDiscovery | Hybrid | OpenAPI specs, common paths, version enum, LLM guessing |
| JSAnalyzer | Hybrid | Regex + LLM extraction of API routes from JS bundles |
| Hypothesis | LLM | Hypothesizes undiscovered endpoints, validates with HEAD requests |
| Classifier | LLM | Categorizes endpoints by type and auth requirements |
| VulnAnalyzer | Hybrid | Two-pass vulnerability detection (deterministic + LLM) |
| Verifier | LLM | Cross-checks findings for consistency, flags re-analysis |
| Prioritizer | LLM | Ranks attack surface with rationale and suggested tests |

Agents run in dependency-ordered waves (e.g., crawler before classifier, classifier before prioritizer). The orchestrator handles parallelism within each wave.

### Prompt Registry

All LLM system prompts live in versioned YAML files under `src/ghost_hunter/prompts/`. Each prompt follows the CRAFT structure (Context, Role, Action, Format, Tone) and includes few-shot examples. This separation makes prompt iteration easier — you can edit prompts without touching agent code.

```bash
# List all prompts
ls src/ghost_hunter/prompts/

# Edit a prompt
$EDITOR src/ghost_hunter/prompts/classifier.yaml
```

### Confidence-Gated Re-analysis

When the LLM self-reports low confidence on a response, `chat_structured()` automatically retries once with the first response as additional context. Controlled by two settings in `.env`:

| Setting | Description | Default |
|---------|-------------|---------|
| `REANALYSIS_ENABLED` | Enable confidence-gated retries | `true` |
| `REANALYSIS_CONFIDENCE_THRESHOLD` | Retry below this score (0.0-1.0) | `0.6` |

### Inter-Agent Feedback

The verifier agent can request re-analysis of specific endpoints by upstream agents (classifier, vuln_analyzer). After the verifier wave completes, the orchestrator runs targeted re-analysis before moving to prioritization. Each endpoint can only be re-analyzed once to prevent loops.

## Evals

Golden-dataset evaluation tests measure how well the agent pipelines handle specific scenarios. They mock the LLM response and verify the agent writes the correct output to `ScanState`.

```bash
# Run all evals
uv run pytest tests/test_evals.py -v

# Run evals for a specific agent
uv run pytest tests/test_evals.py -k classifier -v
uv run pytest tests/test_evals.py -k vuln_analyzer -v
uv run pytest tests/test_evals.py -k prioritizer -v

# Run scoring function unit tests
uv run pytest tests/test_evals.py -k TestScoring -v
```

Golden datasets live in `tests/evals/` and contain realistic (input, expected_output) pairs. Each case includes a pre-built LLM response so tests run without API calls.

**Adding a new eval case:**

1. Add a new dict to the relevant `tests/evals/golden_*.py` file
2. Include `endpoints`, `llm_response` (the mocked LLM output), and `expected` results
3. Run `uv run pytest tests/test_evals.py -v` to verify

Scoring functions in `tests/evals/scoring.py`:
- `classification_accuracy` — fraction of correct (category, auth) pairs
- `vuln_precision_recall` — precision, recall, F1 over vuln indicator strings
- `risk_rank_correlation` — Spearman-style rank correlation for prioritization ordering

## Project Structure

```
ghost_hunter.py              # CLI entry point
src/ghost_hunter/
├── config.py                # Settings from environment variables
├── orchestrator.py          # DAG-based agent scheduler with reanalysis loop
├── prompts.py               # PromptRegistry — loads versioned YAML prompts
├── output.py                # Rich console output + JSON reports
├── report.py                # LLM-powered pentest report generation
├── prompts/                 # Versioned YAML prompt files (CRAFT format)
│   ├── classifier.yaml
│   ├── prioritizer.yaml
│   ├── vuln_analyzer.yaml
│   ├── verifier.yaml
│   ├── planner.yaml
│   ├── hypothesis.yaml
│   ├── api_discovery.yaml
│   ├── js_analyzer.yaml
│   ├── report.yaml
│   └── insight.yaml
├── clients/                 # HTTP, LLM, and tracing clients
│   ├── http.py              # Adaptive rate-limited httpx client
│   ├── llm.py               # OpenAI-compatible wrapper with retry, confidence gating
│   └── tracing.py           # Langfuse initialization and helpers
├── models/                  # Pydantic data models
│   ├── enums.py             # DiscoverySource, RiskLevel, EndpointCategory
│   ├── endpoints.py         # Endpoint, Finding, TechFingerprint
│   ├── llm_responses.py     # Structured output models for all LLM calls
│   ├── scan.py              # ScanState, AgentResult
│   └── attack_surface.py    # AttackSurfaceEntry, VulnIndicator
└── agents/                  # Specialized discovery agents
    ├── base.py              # BaseAgent ABC with tracing + prompt registry
    ├── passive_recon.py
    ├── web_crawler.py
    ├── api_discovery.py
    ├── js_analyzer.py
    ├── hypothesis.py
    ├── classifier.py
    ├── vuln_analyzer.py
    ├── verifier.py
    └── prioritizer.py
tests/
├── test_evals.py            # Parametrized eval tests for 3 agents
└── evals/
    ├── scoring.py           # Accuracy, precision/recall, rank correlation
    ├── golden_classifier.py
    ├── golden_vuln_analyzer.py
    └── golden_prioritizer.py
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Groq API key (free tier works)
