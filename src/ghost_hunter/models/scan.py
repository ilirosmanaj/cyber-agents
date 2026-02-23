"""Scan state and agent result models."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from src.ghost_hunter.models.attack_surface import AttackSurfaceEntry, VulnIndicator
from src.ghost_hunter.models.endpoints import Endpoint, Finding, TechFingerprint
from src.ghost_hunter.models.insights import ScanInsight
from src.ghost_hunter.models.strategy import ScanStrategy


class AgentResult(BaseModel):
    agent_name: str
    success: bool = True
    endpoints_found: list[Endpoint] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: dict = Field(default_factory=dict)


class ScanState(BaseModel):
    scan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    target: str = ""
    base_url: str = ""
    endpoints: dict[str, Endpoint] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    tech_fingerprint: TechFingerprint = Field(default_factory=TechFingerprint)
    attack_surface: list[AttackSurfaceEntry] = Field(default_factory=list)
    vuln_indicators: dict[str, list[VulnIndicator]] = Field(default_factory=dict)
    blocked_paths: list[str] = Field(default_factory=list)
    js_urls: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    agents_completed: list[str] = Field(default_factory=list)
    scan_insights: list[ScanInsight] = Field(default_factory=list)
    scan_strategy: ScanStrategy | None = None

    def endpoint_key(self, method: str, url: str) -> str:
        return f"{method.upper()} {url.rstrip('/')}"

    def add_endpoint(self, ep: Endpoint) -> bool:
        """Add endpoint or merge into existing. Returns True if new."""
        key = self.endpoint_key(ep.method, ep.url)
        if key not in self.endpoints:
            self.endpoints[key] = ep
            return True
        self._merge_endpoint(self.endpoints[key], ep)
        return False

    @staticmethod
    def _merge_endpoint(existing: Endpoint, new: Endpoint) -> None:
        """Merge richer data from a new endpoint into the existing one."""
        if not existing.status_code and new.status_code:
            existing.status_code = new.status_code
        if not existing.content_type and new.content_type:
            existing.content_type = new.content_type
        if new.parameters:
            merged = list(dict.fromkeys(existing.parameters + new.parameters))
            existing.parameters = merged
        if new.requires_auth is not None and existing.requires_auth is None:
            existing.requires_auth = new.requires_auth
        if new.notes and new.notes not in existing.notes:
            existing.notes = f"{existing.notes}; {new.notes}" if existing.notes else new.notes
        if new.parameter_details:
            seen = {(p.name, p.location) for p in existing.parameter_details}
            for p in new.parameter_details:
                if (p.name, p.location) not in seen:
                    existing.parameter_details.append(p)
                    seen.add((p.name, p.location))
        if new.request_body_content_type and not existing.request_body_content_type:
            existing.request_body_content_type = new.request_body_content_type
        if new.request_body_fields:
            merged = list(dict.fromkeys(existing.request_body_fields + new.request_body_fields))
            existing.request_body_fields = merged
        if new.response_fields:
            merged = list(dict.fromkeys(existing.response_fields + new.response_fields))
            existing.response_fields = merged
        if new.security_schemes and not existing.security_schemes:
            existing.security_schemes = new.security_schemes
        if new.response_headers:
            existing.response_headers.update(new.response_headers)
        if new.response_body_snippet and not existing.response_body_snippet:
            existing.response_body_snippet = new.response_body_snippet

    def find_endpoint(self, method: str, url: str) -> Endpoint | None:
        """Find an endpoint by exact key, falling back to URL-only match."""
        key = self.endpoint_key(method, url)
        ep = self.endpoints.get(key)
        if ep is not None:
            return ep
        for v in self.endpoints.values():
            if v.url == url:
                return v
        return None

    def merge_agent_result(self, result: AgentResult) -> None:
        """Merge an agent's result into the shared scan state."""
        for ep in result.endpoints_found:
            self.add_endpoint(ep)
        self.findings.extend(result.findings)
        self.errors.extend(result.errors)
        if result.agent_name not in self.agents_completed:
            self.agents_completed.append(result.agent_name)

    def insights_context(self, max_insights: int = 5) -> str:
        """Format recent insights for LLM prompts."""
        if not self.scan_insights:
            return "No prior insights."
        recent = self.scan_insights[-max_insights:]
        lines: list[str] = []
        for insight in recent:
            lines.append(f"[{insight.phase}] {insight.summary}")
            if insight.key_signals:
                lines.append(f"  Signals: {', '.join(insight.key_signals)}")
            if insight.recommended_focus:
                lines.append(f"  Focus: {', '.join(insight.recommended_focus)}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Short summary for LLM context."""
        cats: dict[str, int] = {}
        for ep in self.endpoints.values():
            cat = ep.category.value if ep.category else "unclassified"
            cats[cat] = cats.get(cat, 0) + 1
        cat_str = ", ".join(f"{v} {k}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
        return (
            f"Target: {self.target}\n"
            f"Endpoints discovered: {len(self.endpoints)}\n"
            f"Categories: {cat_str or 'none yet'}\n"
            f"Findings: {len(self.findings)}\n"
            f"Blocked paths: {len(self.blocked_paths)}\n"
            f"JS files found: {len(self.js_urls)}\n"
            f"Agents completed: {', '.join(self.agents_completed) or 'none'}\n"
            f"Errors: {len(self.errors)}"
        )
