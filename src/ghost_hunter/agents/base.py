"""Base agent with automatic Langfuse tracing."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

from src.ghost_hunter.clients import AdaptiveHttpClient, LLMClient, trace_span
from src.ghost_hunter.models import AgentResult, ScanState

if TYPE_CHECKING:
    from src.ghost_hunter.prompts import PromptRegistry

# cap on blocked paths included in LLM context to avoid token bloat
_MAX_BLOCKED_PATHS_IN_CONTEXT = 10

MAX_RESPONSE_BODY_SNIPPET = 4000
_TEXT_CONTENT_TYPES = ("text/", "json", "xml")


class BaseAgent(ABC):
    """Abstract base for all agents. Wraps run() in Langfuse span automatically."""

    name: str = "base"
    description: str = ""

    def __init__(
        self,
        http_client: AdaptiveHttpClient,
        llm_client: LLMClient,
        prompt_registry: PromptRegistry | None = None,
    ):
        self.http = http_client
        self.llm = llm_client
        self.prompt_registry = prompt_registry

    async def execute(self, state: ScanState) -> AgentResult:
        """Execute the agent with tracing. Sub-classes implement run()."""
        start = time.monotonic()

        with trace_span(self.name) as span:
            try:
                result = await self.run(state)
            except Exception as e:
                result = AgentResult(
                    agent_name=self.name,
                    success=False,
                    errors=[f"Agent {self.name} crashed: {e}"],
                )

            result.duration_seconds = time.monotonic() - start

            if span is not None:
                span.update(
                    metadata={
                        "endpoints_found": len(result.endpoints_found),
                        "findings": len(result.findings),
                        "errors": result.errors,
                        "duration_s": round(result.duration_seconds, 2),
                        "success": result.success,
                    }
                )

        return result

    @staticmethod
    def build_tech_context(state: ScanState) -> str:
        """Build detailed tech fingerprint string from scan state."""
        fp = state.tech_fingerprint
        lines = []
        if fp.server:
            lines.append(f"Server: {fp.server}")
        if fp.frameworks:
            lines.append(f"Frameworks: {', '.join(fp.frameworks)}")
        if fp.security_headers:
            lines.append(f"Security headers present: {', '.join(fp.security_headers.keys())}")
        if fp.missing_security_headers:
            lines.append(f"Missing security headers: {', '.join(fp.missing_security_headers)}")
        if fp.cookies:
            lines.append(f"Cookies: {', '.join(fp.cookies)}")
        if fp.technologies:
            lines.append(f"Technologies: {', '.join(fp.technologies)}")
        if state.blocked_paths:
            lines.append(
                f"Blocked (403) paths: {', '.join(state.blocked_paths[:_MAX_BLOCKED_PATHS_IN_CONTEXT])}"
            )
        return "\n".join(lines) or "Unknown"

    @staticmethod
    def build_insights_context(state: ScanState) -> str:
        """Format accumulated scan insights for LLM prompts."""
        return state.insights_context()

    @staticmethod
    def build_findings_context(state: ScanState, limit: int = 15) -> str:
        """Format existing findings as context for LLM prompts."""
        if not state.findings:
            return "None yet"
        return "\n".join(
            f"- [{f.severity.value}] {f.title}" for f in state.findings[:limit]
        )

    @staticmethod
    def extract_body_snippet(resp: httpx.Response) -> str:
        """Capture first N chars of text-based responses for analysis."""
        ct = resp.headers.get("content-type", "")
        if resp.status_code >= 300:
            return ""
        if not any(t in ct for t in _TEXT_CONTENT_TYPES):
            return ""
        return resp.text[:MAX_RESPONSE_BODY_SNIPPET]

    @abstractmethod
    async def run(self, state: ScanState) -> AgentResult:
        """Implement agent logic here. Tracing is handled by execute()."""
        ...
