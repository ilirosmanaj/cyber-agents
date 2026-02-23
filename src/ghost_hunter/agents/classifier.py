"""Classifier agent — LLM categorizes endpoints by type, auth requirement, and risk."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.models import (
    AgentResult,
    Endpoint,
    EndpointCategory,
    Finding,
    RiskLevel,
    ScanState,
)
from src.ghost_hunter.models.llm_responses import ClassificationBatchResponse

logger = logging.getLogger(__name__)

BATCH_SIZE = 30

# max chars of notes sent to the LLM
_MAX_NOTES_LENGTH = 120

# max chars of response body snippet per endpoint in batch context
_MAX_SNIPPET_IN_BATCH = 200

_HEALTH_PATHS = re.compile(
    r"^/(?:health|healthz|readyz|status|ping|alive|ready)$", re.IGNORECASE
)
_STATIC_EXTENSIONS = re.compile(
    r"\.(?:js|css|png|jpe?g|gif|svg|ico|woff2?|ttf|eot|webp|avif|map)$",
    re.IGNORECASE,
)
_AUTH_PATHS = re.compile(
    r"(?:/login|/logout|/register|/signup|/signin|/auth|/oauth|/callback|"
    r"/password[_-]?reset|/token|/refresh)",
    re.IGNORECASE,
)
_ADMIN_PATHS = re.compile(
    r"(?:/admin|/dashboard|/management|/console)", re.IGNORECASE
)
_DEBUG_PATHS = re.compile(
    r"(?:/debug|/actuator|/phpinfo|/__debug__|/trace|/profiler)", re.IGNORECASE
)
_DOC_PATHS = re.compile(
    r"(?:/docs|/swagger|/redoc|/api-docs|/openapi|/documentation)", re.IGNORECASE
)
_GRAPHQL_PATHS = re.compile(r"(?:/graphql|/graphiql)", re.IGNORECASE)


def _pre_classify(ep: Endpoint) -> EndpointCategory | None:
    """Deterministic pre-classification for obvious endpoints."""
    path = urlparse(ep.url).path

    if _STATIC_EXTENSIONS.search(path):
        return EndpointCategory.STATIC_ASSET

    ct = (ep.content_type or "").split(";")[0].strip()
    if ct in ("application/javascript", "text/css", "image/png", "image/jpeg", "image/svg+xml"):
        return EndpointCategory.STATIC_ASSET

    if _HEALTH_PATHS.match(path):
        return EndpointCategory.HEALTH_CHECK

    if _GRAPHQL_PATHS.search(path):
        return EndpointCategory.GRAPHQL

    if _DOC_PATHS.search(path):
        return EndpointCategory.DOCUMENTATION

    if _DEBUG_PATHS.search(path):
        return EndpointCategory.DEBUG_ENDPOINT

    if _ADMIN_PATHS.search(path):
        return EndpointCategory.ADMIN_ENDPOINT

    if _AUTH_PATHS.search(path):
        return EndpointCategory.AUTH_ENDPOINT

    return None


@register_agent
class ClassifierAgent(BaseAgent):
    name = "classifier"
    description = "Uses LLM to classify each endpoint by category (REST API, auth, admin, etc.) and auth requirements."

    async def run(self, state: ScanState) -> AgentResult:
        findings: list[Finding] = []
        errors: list[str] = []

        all_endpoints = list(state.endpoints.items())
        if not all_endpoints:
            return AgentResult(agent_name=self.name, success=True)

        tech_context = self.build_tech_context(state)
        findings_context = self.build_findings_context(state)
        insights_context = self.build_insights_context(state)
        strategy_tech = ""
        if state.scan_strategy and state.scan_strategy.tech_hypotheses:
            strategy_tech = (
                f"\nTech Hypotheses (from planner): "
                f"{', '.join(state.scan_strategy.tech_hypotheses)}\n"
            )

        # pass 1: deterministic pre-classification
        llm_batch: list[tuple[int, str, Endpoint]] = []
        pre_classified = 0

        for idx, (key, ep) in enumerate(all_endpoints):
            category = _pre_classify(ep)
            if category is not None:
                ep.category = category
                if ep.requires_auth is None and ep.security_schemes:
                    ep.requires_auth = True
                elif ep.requires_auth is None and ep.status_code in (401, 403):
                    ep.requires_auth = True
                pre_classified += 1
            else:
                if ep.security_schemes and ep.requires_auth is None:
                    ep.requires_auth = True
                if ep.status_code in (401, 403) and ep.requires_auth is None:
                    ep.requires_auth = True
                llm_batch.append((idx, key, ep))

        # pass 2: LLM classification for non-obvious endpoints
        classified_count = pre_classified
        for i in range(0, len(llm_batch), BATCH_SIZE):
            batch = llm_batch[i : i + BATCH_SIZE]
            batch_data = self._build_batch_data(batch)

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
                        "OVERLAP GUIDANCE:\n"
                        "- Login/register forms → auth_endpoint (not form_action)\n"
                        "- Admin login → admin_endpoint\n"
                        "- POST /graphql → graphql (not rest_api)\n\n"
                        "CATEGORY DEFINITIONS:\n"
                        "- rest_api: RESTful data endpoints (CRUD on resources, JSON responses, parameterized paths)\n"
                        "- form_action: HTML form submission targets (POST with form-encoded data, contact forms)\n"
                        "- static_asset: CSS, JS, images, fonts, or other static files\n"
                        "- auth_endpoint: Login, logout, register, password reset, token refresh, OAuth callbacks\n"
                        "- admin_endpoint: Administrative panels, user management, system configuration, dashboards\n"
                        "- debug_endpoint: Debug tools, profilers, stack traces, actuator endpoints, phpinfo\n"
                        "- documentation: API docs, Swagger UI, ReDoc, developer guides\n"
                        "- health_check: Health, readiness, liveness probes\n"
                        "- graphql: GraphQL query endpoints\n"
                        "- unknown: Cannot determine from available information\n\n"
                        "AUTH ASSESSMENT CRITERIA:\n"
                        "- true: Returns 401/403 without token, has security schemes (Bearer, API key), "
                        "path contains /user/ /account/ /profile/ /admin/, operates on user-specific resources\n"
                        "- false: Returns 200 without credentials, serves public content, login/register, "
                        "health check, documentation\n"
                        "- null: Cannot determine\n\n"
                        "FORMAT:\n"
                        "Respond with JSON:\n"
                        "{\n"
                        '  "reasoning": "Brief observation about patterns in this batch",\n'
                        '  "classifications": [\n'
                        '    {"index": 1, "category": "...", "requires_auth": true|false|null}\n'
                        "  ]\n"
                        "}\n\n"
                        "IMPORTANT: Use the index number to identify each endpoint. Do not repeat URLs."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Target: {state.target}\n\n"
                        f"Prior Insights:\n{insights_context}\n\n"
                        f"Tech Stack:\n{tech_context}{strategy_tech}\n\n"
                        f"Prior Findings:\n{findings_context}\n\n"
                        f"Endpoints to classify:\n{self._format_batch(batch_data)}"
                    ),
                },
            ]

            try:
                response = await self.llm.chat_structured(
                    messages, response_model=ClassificationBatchResponse,
                    name=f"classify_batch_{i // BATCH_SIZE}",
                )

                for cls in response.classifications:
                    if cls.index is None:
                        # fallback to URL-based matching
                        key = state.endpoint_key(cls.method, cls.url)
                        if key not in state.endpoints:
                            continue
                        ep = state.endpoints[key]
                    else:
                        # index-based matching (1-indexed)
                        list_idx = cls.index - 1
                        if list_idx < 0 or list_idx >= len(batch):
                            continue
                        _, _, ep = batch[list_idx]

                    try:
                        ep.category = EndpointCategory(cls.category)
                    except ValueError:
                        ep.category = EndpointCategory.UNKNOWN

                    if cls.requires_auth is not None:
                        ep.requires_auth = cls.requires_auth
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

    @staticmethod
    def _build_batch_data(
        batch: list[tuple[int, str, Endpoint]]
    ) -> list[dict]:
        """Build batch data dicts with richer metadata."""
        result = []
        for idx, key, ep in batch:
            entry = {
                "index": len(result) + 1,
                "url": ep.url,
                "method": ep.method,
                "status_code": ep.status_code,
                "content_type": ep.content_type or "",
                "parameters": ep.parameters,
                "discovered_by": ep.discovered_by.value if ep.discovered_by else "",
            }
            if ep.notes:
                entry["notes"] = ep.notes[:_MAX_NOTES_LENGTH]
            if ep.security_schemes:
                entry["security"] = [
                    f"{s.scheme_type}:{s.scheme_name}" for s in ep.security_schemes
                ]
            if ep.response_body_snippet:
                entry["response_preview"] = ep.response_body_snippet[:_MAX_SNIPPET_IN_BATCH]
            if ep.request_body_content_type:
                entry["body_type"] = ep.request_body_content_type
            if ep.request_body_fields:
                entry["body_fields"] = ep.request_body_fields[:10]
            if ep.response_fields:
                entry["response_fields"] = ep.response_fields[:10]
            result.append(entry)
        return result

    @staticmethod
    def _category_summary(state: ScanState) -> str:
        cats: dict[str, int] = {}
        for ep in state.endpoints.values():
            cat = ep.category.value if ep.category else "unclassified"
            cats[cat] = cats.get(cat, 0) + 1
        return ", ".join(f"{v} {k}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))

    @staticmethod
    def _format_batch(batch: list[dict]) -> str:
        lines = []
        for ep in batch:
            parts = [
                f"  {ep['index']}. {ep['method']} {ep['url']} [{ep['status_code']}] "
                f"{ep['content_type']}"
            ]
            if ep.get("parameters"):
                parts.append(f"params={ep['parameters']}")
            if ep.get("security"):
                parts.append(f"security={ep['security']}")
            if ep.get("discovered_by"):
                parts.append(f"src={ep['discovered_by']}")
            if ep.get("response_preview"):
                parts.append(f"body={ep['response_preview'][:80]}...")
            lines.append(" | ".join(parts))
        return "\n".join(lines)
