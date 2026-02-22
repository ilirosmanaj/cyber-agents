"""LLM-driven orchestrator — tool-use loop that coordinates agents."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.ghost_hunter.agents import get_agent_registry
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.clients import AdaptiveHttpClient, LLMClient, trace_span
from src.ghost_hunter.config import settings
from src.ghost_hunter.models import ScanState
from src.ghost_hunter.output import print_agent_step

logger = logging.getLogger(__name__)

DEFAULT_PHASE_ORDER = [
    "passive_recon",
    "web_crawler",
    "api_discovery",
    "js_analyzer",
    "hypothesis",
    "classifier",
    "vuln_analyzer",
    "prioritizer",
]

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_agent",
            "description": "Run a discovery/analysis agent. Available agents: "
            "passive_recon (robots.txt, sitemap, headers), "
            "web_crawler (BFS crawl for links/forms/scripts), "
            "api_discovery (OpenAPI specs, common paths, version enum, LLM guessing), "
            "js_analyzer (regex extraction from JS bundles), "
            "hypothesis (LLM hypothesizes + validates new endpoints), "
            "classifier (LLM classifies endpoints by type/auth), "
            "vuln_analyzer (structural vulnerability pattern detection), "
            "prioritizer (LLM ranks attack surface).",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": DEFAULT_PHASE_ORDER,
                        "description": "Which agent to run next",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this agent should run now",
                    },
                },
                "required": ["agent_name", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the scan. Call this when all useful agents have been run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


class Orchestrator:
    """LLM-driven orchestrator that coordinates agent execution."""

    def __init__(
        self,
        http_client: AdaptiveHttpClient,
        llm_client: LLMClient,
        state: ScanState,
    ):
        self.http = http_client
        self.llm = llm_client
        self.state = state
        self._registry = get_agent_registry()
        self._agents: dict[str, BaseAgent] = {}

        for name, cls in self._registry.items():
            self._agents[name] = cls(http_client=http_client, llm_client=llm_client)

    async def run(self) -> ScanState:
        """Execute the full scan with LLM-driven agent orchestration."""
        with trace_span("orchestrator", metadata={"target": self.state.target}):
            await self._orchestrate()
            await self._run_fallback_phases()

        return self.state

    async def _orchestrate(self) -> None:
        """LLM tool-use loop: LLM picks agents based on scan state."""
        messages = [
            {
                "role": "system",
                "content": (
                    "CONTEXT:\n"
                    "You are the orchestrator for Ghost Hunter, a multi-agent API security scanner. "
                    "You coordinate 8 specialized agents that form an information pipeline. Each agent "
                    "depends on outputs from prior agents.\n\n"
                    "ROLE:\n"
                    "You are a scan coordinator who understands information flow between security "
                    "analysis agents. You make strategic decisions about agent ordering, re-runs, "
                    "and skips based on intermediate results.\n\n"
                    "AGENT PIPELINE AND DEPENDENCIES:\n"
                    "1. passive_recon — Discovers robots.txt, sitemap, headers, tech fingerprint. "
                    "No dependencies. MUST run first.\n"
                    "2. web_crawler — BFS crawl for links, forms, scripts. Needs: passive_recon "
                    "(for base URL). Produces: endpoints, JS URLs.\n"
                    "3. api_discovery — OpenAPI specs, common paths, version enum, LLM guessing. "
                    "Needs: web_crawler (for known endpoints context). Produces: API endpoints.\n"
                    "4. js_analyzer — Regex extraction from JS bundles. REQUIRES: web_crawler "
                    "(for js_urls). SKIP if no JS files found.\n"
                    "5. hypothesis — LLM hypothesizes + validates new endpoints. Needs: all "
                    "discovery agents. Consider RE-RUN if >10 new endpoints from api_discovery.\n"
                    "6. classifier — LLM classifies endpoints by type/auth. REQUIRES: endpoints "
                    "from discovery phases. Produces: categories, auth flags.\n"
                    "7. vuln_analyzer — Two-pass vulnerability detection (deterministic + LLM). "
                    "REQUIRES: classifier output (categories, auth flags).\n"
                    "8. prioritizer — LLM ranks attack surface. REQUIRES: vuln_analyzer. "
                    "MUST run last.\n\n"
                    "ACTION:\n"
                    "Before each agent call, reason about:\n"
                    "- What information does this agent need? Has it been produced?\n"
                    "- What will this agent produce for downstream agents?\n"
                    "- Is a re-run warranted? (e.g., hypothesis after >10 new endpoints)\n"
                    "- Should js_analyzer be skipped? (no JS files found)\n"
                    "The 'reason' field in run_agent() should reflect strategic thinking, not just "
                    "'it is next in the pipeline'.\n\n"
                    "FORMAT:\n"
                    "Use run_agent() to execute agents, or finish() when all useful phases are done."
                ),
            },
            {
                "role": "user",
                "content": f"Starting scan of {self.state.target}.\n\nCurrent state:\n{self.state.summary()}",
            },
        ]

        for iteration in range(settings.max_orchestrator_iterations):
            try:
                response = await self.llm.chat(
                    messages=messages,
                    name=f"orchestrator_step_{iteration}",
                    tools=AGENT_TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.error("Orchestrator LLM call failed: %s", e)
                break

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                if choice.finish_reason == "stop":
                    logger.info("Orchestrator finished (no more tool calls)")
                    break
                messages.append({"role": "assistant", "content": msg.content or ""})
                continue

            # only include fields the Groq API accepts
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if fn_name == "finish":
                    logger.info("Orchestrator finished: %s", args.get("summary", ""))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Scan complete.",
                        }
                    )
                    return

                if fn_name == "run_agent":
                    agent_name = args.get("agent_name", "")
                    reason = args.get("reason", "")
                    result_text = await self._run_agent(agent_name, reason)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        }
                    )

    async def _run_agent(self, agent_name: str, reason: str) -> str:
        """Execute a single agent and return a summary for the LLM."""
        if agent_name not in self._agents:
            return f"Unknown agent: {agent_name}"

        agent = self._agents[agent_name]
        logger.info("Running agent: %s (reason: %s)", agent_name, reason)

        result = await agent.execute(self.state)
        self.state.merge_agent_result(result)

        print_agent_step(agent_name=agent_name, reason=reason, result=result)

        return (
            f"Agent '{agent_name}' completed in {result.duration_seconds:.1f}s.\n"
            f"Endpoints found: {len(result.endpoints_found)}\n"
            f"Findings: {len(result.findings)}\n"
            f"Errors: {result.errors or 'none'}\n\n"
            f"Updated state:\n{self.state.summary()}"
        )

    async def _run_fallback_phases(self) -> None:
        """Ensure all default phases have run, even if the LLM skipped them."""
        for phase in DEFAULT_PHASE_ORDER:
            if phase not in self.state.agents_completed:
                if phase == "js_analyzer" and not self.state.js_urls:
                    logger.info("Skipping js_analyzer fallback — no JS files found")
                    self.state.agents_completed.append(phase)
                    continue

                logger.info("Fallback: running skipped phase '%s'", phase)
                await self._run_agent(phase, "fallback — phase not run by orchestrator")
