"""Classifier agent — LLM categorizes endpoints by type, auth requirement, and risk."""

from __future__ import annotations

import logging

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.models import (
    AgentResult,
    EndpointCategory,
    Finding,
    RiskLevel,
    ScanState,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 30


@register_agent
class ClassifierAgent(BaseAgent):
    name = "classifier"
    description = "Uses LLM to classify each endpoint by category (REST API, auth, admin, etc.) and auth requirements."

    async def run(self, state: ScanState) -> AgentResult:
        findings: list[Finding] = []
        errors: list[str] = []

        all_endpoints = list(state.endpoints.values())
        if not all_endpoints:
            return AgentResult(agent_name=self.name, success=True)

        tech_context = self.build_tech_context(state)
        findings_context = self.build_findings_context(state)

        classified_count = 0
        for i in range(0, len(all_endpoints), BATCH_SIZE):
            batch = all_endpoints[i : i + BATCH_SIZE]
            batch_data = [
                {
                    "url": ep.url,
                    "method": ep.method,
                    "status_code": ep.status_code,
                    "content_type": ep.content_type or "",
                    "parameters": ep.parameters,
                    "notes": ep.notes,
                }
                for ep in batch
            ]

            messages = [
                {
                    "role": "system",
                    "content": (
                        "CONTEXT:\n"
                        "You are classifying web application endpoints discovered through multiple methods "
                        "(crawling, OpenAPI spec parsing, JavaScript analysis, LLM hypothesis, common path "
                        "probing). Your classifications feed directly into the vulnerability analyzer, so "
                        "accurate categorization and auth assessment are critical for downstream security "
                        "analysis.\n\n"
                        "ROLE:\n"
                        "You are a web application security analyst specializing in attack surface mapping. "
                        "You understand REST conventions, authentication patterns, and how endpoint behavior "
                        "signals its purpose.\n\n"
                        "ACTION:\n"
                        "For each endpoint, follow these steps:\n"
                        "1. Examine the URL path, HTTP method, status code, content type, and parameters\n"
                        "2. Assign a category based on the definitions below\n"
                        "3. Assess authentication requirements based on the criteria below\n\n"
                        "CATEGORY DEFINITIONS:\n"
                        "- rest_api: RESTful data endpoints (CRUD on resources, JSON responses, parameterized paths)\n"
                        "- form_action: HTML form submission targets (POST with form-encoded data, login/contact forms)\n"
                        "- static_asset: CSS, JS, images, fonts, or other static files served without dynamic logic\n"
                        "- auth_endpoint: Login, logout, register, password reset, token refresh, OAuth callbacks\n"
                        "- admin_endpoint: Administrative panels, user management, system configuration, dashboards\n"
                        "- debug_endpoint: Debug tools, profilers, stack traces, actuator endpoints, phpinfo\n"
                        "- documentation: API docs, Swagger UI, ReDoc, developer guides, help pages\n"
                        "- health_check: Health, readiness, liveness probes (e.g., /health, /healthz, /readyz, /status)\n"
                        "- graphql: GraphQL query endpoints (typically POST /graphql with query body)\n"
                        "- unknown: Cannot determine purpose from available information\n\n"
                        "AUTH ASSESSMENT CRITERIA:\n"
                        "- true: Endpoint requires authentication. Signals: returns 401/403 without token, "
                        "path contains /user/ /account/ /profile/ /admin/, has security schemes, "
                        "operates on user-specific resources\n"
                        "- false: Endpoint is publicly accessible. Signals: returns 200 without credentials, "
                        "serves public content, is a login/register endpoint, health check, documentation\n"
                        "- null: Cannot determine. Use when there isn't enough evidence either way\n\n"
                        "FORMAT:\n"
                        "Respond with JSON:\n"
                        "{\n"
                        '  "reasoning": "Brief observation about patterns in this batch",\n'
                        '  "classifications": [\n'
                        '    {"url": "...", "method": "...", "category": "...", "requires_auth": true|false|null}\n'
                        "  ]\n"
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Target: {state.target}\n\n"
                        f"Tech Stack:\n{tech_context}\n\n"
                        f"Prior Findings:\n{findings_context}\n\n"
                        f"Endpoints to classify:\n{self._format_batch(batch_data)}"
                    ),
                },
            ]

            try:
                data = await self.llm.chat_json(
                    messages, name=f"classify_batch_{i // BATCH_SIZE}"
                )
                classifications = data.get("classifications", [])

                for cls in classifications:
                    url = cls.get("url", "")
                    method = cls.get("method", "GET")
                    category = cls.get("category", "unknown")
                    requires_auth = cls.get("requires_auth")

                    key = state.endpoint_key(method, url)
                    if key in state.endpoints:
                        try:
                            state.endpoints[key].category = EndpointCategory(category)
                        except ValueError:
                            state.endpoints[key].category = EndpointCategory.UNKNOWN
                        state.endpoints[key].requires_auth = requires_auth
                        classified_count += 1

            except Exception as e:
                errors.append(f"Classification batch {i // BATCH_SIZE} failed: {e}")

        findings.append(
            Finding(
                agent_name=self.name,
                finding_type="classification_complete",
                title=f"Classified {classified_count}/{len(all_endpoints)} endpoints",
                detail=self._category_summary(state),
                severity=RiskLevel.INFO,
            )
        )

        return AgentResult(
            agent_name=self.name,
            success=len(errors) == 0,
            endpoints_found=[],
            findings=findings,
            errors=errors,
        )

    def _category_summary(self, state: ScanState) -> str:
        cats: dict[str, int] = {}
        for ep in state.endpoints.values():
            cat = ep.category.value if ep.category else "unclassified"
            cats[cat] = cats.get(cat, 0) + 1
        return ", ".join(f"{v} {k}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))

    @staticmethod
    def _format_batch(batch: list[dict]) -> str:
        lines = []
        for i, ep in enumerate(batch, 1):
            params = f" params={ep['parameters']}" if ep["parameters"] else ""
            lines.append(
                f"  {i}. {ep['method']} {ep['url']} [{ep['status_code']}] "
                f"{ep['content_type']}{params}"
            )
        return "\n".join(lines)
