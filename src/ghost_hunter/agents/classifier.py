"""Classifier agent — LLM categorizes endpoints, deterministic rules correct obvious cases."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.config import settings
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
_STATIC_CONTENT_TYPES = frozenset({
    "application/javascript", "text/css", "image/png",
    "image/jpeg", "image/svg+xml",
})


def _rule_override(ep: Endpoint) -> EndpointCategory | None:
    """Override LLM classification for static assets and health checks."""
    path = urlparse(ep.url).path

    if _STATIC_EXTENSIONS.search(path):
        return EndpointCategory.STATIC_ASSET

    ct = (ep.content_type or "").split(";")[0].strip()
    if ct in _STATIC_CONTENT_TYPES:
        return EndpointCategory.STATIC_ASSET

    if _HEALTH_PATHS.match(path):
        return EndpointCategory.HEALTH_CHECK

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

        # LLM classifies all endpoints
        llm_batch: list[tuple[int, str, Endpoint]] = [
            (idx, key, ep) for idx, (key, ep) in enumerate(all_endpoints)
        ]
        classified_count = 0

        for i in range(0, len(llm_batch), BATCH_SIZE):
            batch = llm_batch[i : i + BATCH_SIZE]
            batch_data = self._build_batch_data(batch)

            messages = [
                {
                    "role": "system",
                    "content": self.prompt_registry.get("classifier").system_prompt,
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
                    confidence_threshold=settings.active_confidence_threshold,
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

        # deterministic overrides for static assets and health checks
        override_count = 0
        for _key, ep in all_endpoints:
            override = _rule_override(ep)
            if override is not None and ep.category != override:
                ep.category = override
                override_count += 1

        # infer auth from status codes and security schemes
        for _key, ep in all_endpoints:
            if ep.requires_auth is None and ep.security_schemes:
                ep.requires_auth = True
            if ep.requires_auth is None and ep.status_code in (401, 403):
                ep.requires_auth = True

        category_summary = self._category_summary(state)
        logger.info("Classification: %s", category_summary)

        override_note = f" ({override_count} rule overrides)" if override_count else ""
        findings.append(
            Finding(
                agent_name=self.name,
                finding_type="classification_complete",
                title=f"Classified {classified_count}/{len(all_endpoints)} endpoints{override_note}",
                detail=category_summary,
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
