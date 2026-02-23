"""Tests for LLM response models and chat_structured()."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.ghost_hunter.models.llm_responses import (
    APIGuess,
    APIGuessResponse,
    AttackSurfaceItem,
    ClassificationBatchResponse,
    EndpointClassification,
    EndpointHypothesis,
    HypothesisResponse,
    JSAnalysisBatchResponse,
    JSEndpoint,
    PrioritizationBatchResponse,
    VerifierAction,
    VerifierBatchResponse,
    VulnAnalysisBatchResponse,
    VulnConfidenceAdjustment,
    VulnEndpointAnalysis,
    VulnNewIndicator,
    VulnSuppression,
)
from src.ghost_hunter.models.strategy import ScanStrategy
from src.ghost_hunter.models.insights import ScanInsight


# ---------------------------------------------------------------------------
# VulnAnalysisBatchResponse
# ---------------------------------------------------------------------------


class TestVulnAnalysisBatchResponse:
    def test_full_response(self):
        data = {
            "reasoning": "Found IDOR patterns",
            "endpoint_analyses": [
                {
                    "endpoint_key": "GET https://example.com/api/users",
                    "new_indicators": [
                        {
                            "pattern": "bola_idor",
                            "confidence": "high",
                            "evidence": "numeric ID",
                            "description": "IDOR via user ID",
                            "chain_with": None,
                        }
                    ],
                    "suppressions": [
                        {"original_pattern": "info_disclosure", "reason": "swagger doc"}
                    ],
                    "confidence_adjustments": [
                        {
                            "original_pattern": "ssrf",
                            "new_confidence": "critical",
                            "reason": "unauthenticated",
                        }
                    ],
                }
            ],
        }
        resp = VulnAnalysisBatchResponse.model_validate(data)
        assert resp.reasoning == "Found IDOR patterns"
        assert len(resp.endpoint_analyses) == 1
        analysis = resp.endpoint_analyses[0]
        assert analysis.endpoint_key == "GET https://example.com/api/users"
        assert len(analysis.new_indicators) == 1
        assert analysis.new_indicators[0].pattern == "bola_idor"
        assert len(analysis.suppressions) == 1
        assert len(analysis.confidence_adjustments) == 1

    def test_empty_response(self):
        resp = VulnAnalysisBatchResponse.model_validate({})
        assert resp.reasoning == ""
        assert resp.endpoint_analyses == []

    def test_missing_optional_fields(self):
        data = {
            "endpoint_analyses": [
                {"endpoint_key": "GET /api/test"}
            ]
        }
        resp = VulnAnalysisBatchResponse.model_validate(data)
        analysis = resp.endpoint_analyses[0]
        assert analysis.new_indicators == []
        assert analysis.suppressions == []
        assert analysis.confidence_adjustments == []


# ---------------------------------------------------------------------------
# ClassificationBatchResponse
# ---------------------------------------------------------------------------


class TestClassificationBatchResponse:
    def test_index_based(self):
        data = {
            "reasoning": "Standard REST API",
            "classifications": [
                {"index": 1, "category": "rest_api", "requires_auth": True}
            ],
        }
        resp = ClassificationBatchResponse.model_validate(data)
        assert resp.classifications[0].index == 1
        assert resp.classifications[0].category == "rest_api"
        assert resp.classifications[0].requires_auth is True

    def test_url_fallback(self):
        data = {
            "classifications": [
                {
                    "url": "https://example.com/api/login",
                    "method": "POST",
                    "category": "auth_endpoint",
                    "requires_auth": False,
                }
            ]
        }
        resp = ClassificationBatchResponse.model_validate(data)
        cls = resp.classifications[0]
        assert cls.index is None
        assert cls.url == "https://example.com/api/login"
        assert cls.method == "POST"

    def test_defaults(self):
        data = {"classifications": [{"index": 1}]}
        resp = ClassificationBatchResponse.model_validate(data)
        cls = resp.classifications[0]
        assert cls.category == "unknown"
        assert cls.requires_auth is None


# ---------------------------------------------------------------------------
# HypothesisResponse
# ---------------------------------------------------------------------------


class TestHypothesisResponse:
    def test_full_response(self):
        data = {
            "reasoning": "Banking app patterns",
            "hypotheses": [
                {
                    "path": "/api/v1/transfer",
                    "method": "POST",
                    "confidence": "high",
                    "reasoning": "transfer endpoint expected",
                }
            ],
        }
        resp = HypothesisResponse.model_validate(data)
        assert len(resp.hypotheses) == 1
        assert resp.hypotheses[0].path == "/api/v1/transfer"
        assert resp.hypotheses[0].confidence == "high"

    def test_empty(self):
        resp = HypothesisResponse.model_validate({})
        assert resp.hypotheses == []

    def test_minimal_hypothesis(self):
        data = {"hypotheses": [{"path": "/test"}]}
        resp = HypothesisResponse.model_validate(data)
        hyp = resp.hypotheses[0]
        assert hyp.method == "GET"
        assert hyp.confidence == "medium"
        assert hyp.reasoning == ""


# ---------------------------------------------------------------------------
# PrioritizationBatchResponse
# ---------------------------------------------------------------------------


class TestPrioritizationBatchResponse:
    def test_full_response(self):
        data = {
            "reasoning": "Critical IDOR on financial endpoint",
            "attack_surface": [
                {
                    "index": 1,
                    "category": "rest_api",
                    "risk_level": "critical",
                    "rationale": "Unauthenticated IDOR",
                    "suggested_tests": ["curl -X GET https://example.com/api/users/1"],
                }
            ],
        }
        resp = PrioritizationBatchResponse.model_validate(data)
        item = resp.attack_surface[0]
        assert item.index == 1
        assert item.risk_level == "critical"
        assert len(item.suggested_tests) == 1

    def test_url_fallback(self):
        data = {
            "attack_surface": [
                {"url": "https://example.com/api", "method": "GET", "risk_level": "low"}
            ]
        }
        resp = PrioritizationBatchResponse.model_validate(data)
        item = resp.attack_surface[0]
        assert item.index is None
        assert item.url == "https://example.com/api"


# ---------------------------------------------------------------------------
# VerifierBatchResponse
# ---------------------------------------------------------------------------


class TestVerifierBatchResponse:
    def test_full_response(self):
        data = {
            "reasoning": "Consistent findings",
            "actions": [
                {
                    "endpoint_key": "GET https://example.com/api",
                    "action": "suppress",
                    "pattern": "info_disclosure",
                    "reason": "swagger docs",
                }
            ],
            "cross_cutting_notes": ["All auth endpoints consistent"],
        }
        resp = VerifierBatchResponse.model_validate(data)
        assert len(resp.actions) == 1
        assert resp.actions[0].action == "suppress"
        assert len(resp.cross_cutting_notes) == 1

    def test_empty(self):
        resp = VerifierBatchResponse.model_validate({})
        assert resp.actions == []
        assert resp.cross_cutting_notes == []

    def test_action_requires_fields(self):
        with pytest.raises(ValidationError):
            VerifierAction.model_validate({"action": "suppress"})


# ---------------------------------------------------------------------------
# JSAnalysisBatchResponse
# ---------------------------------------------------------------------------


class TestJSAnalysisBatchResponse:
    def test_full_response(self):
        data = {
            "endpoints": [
                {"path": "/api/users", "method": "GET", "evidence": "fetch call"}
            ],
            "auth_patterns": ["Bearer token in localStorage"],
            "notes": "React SPA",
        }
        resp = JSAnalysisBatchResponse.model_validate(data)
        assert len(resp.endpoints) == 1
        assert resp.endpoints[0].path == "/api/users"
        assert len(resp.auth_patterns) == 1
        assert resp.notes == "React SPA"

    def test_empty(self):
        resp = JSAnalysisBatchResponse.model_validate({})
        assert resp.endpoints == []
        assert resp.auth_patterns == []
        assert resp.notes == ""


# ---------------------------------------------------------------------------
# APIGuessResponse
# ---------------------------------------------------------------------------


class TestAPIGuessResponse:
    def test_full_response(self):
        data = {
            "reasoning": "REST patterns observed",
            "guesses": [
                {"path": "/api/v1/orders", "method": "GET", "reason": "CRUD pattern"}
            ],
        }
        resp = APIGuessResponse.model_validate(data)
        assert len(resp.guesses) == 1
        assert resp.guesses[0].path == "/api/v1/orders"

    def test_minimal_guess(self):
        data = {"guesses": [{"path": "/test"}]}
        resp = APIGuessResponse.model_validate(data)
        guess = resp.guesses[0]
        assert guess.method == "GET"
        assert guess.reason == ""


# ---------------------------------------------------------------------------
# ScanStrategy (reused for planner)
# ---------------------------------------------------------------------------


class TestScanStrategyValidation:
    def test_from_llm_output(self):
        data = {
            "focus_areas": ["auth_flows"],
            "skip_agents": ["js_analyzer"],
            "extra_paths_to_try": ["/admin"],
            "tech_hypotheses": ["Django"],
            "scan_depth": "deep",
            "priority_patterns": ["admin"],
        }
        strategy = ScanStrategy.model_validate(data)
        assert strategy.focus_areas == ["auth_flows"]
        assert strategy.scan_depth == "deep"

    def test_empty_defaults(self):
        strategy = ScanStrategy.model_validate({})
        assert strategy.focus_areas == []
        assert strategy.scan_depth == "normal"


# ---------------------------------------------------------------------------
# ScanInsight (reused for orchestrator)
# ---------------------------------------------------------------------------


class TestScanInsightValidation:
    def test_from_llm_output(self):
        data = {
            "summary": "Discovered admin endpoints",
            "key_signals": ["admin panel"],
            "recommended_focus": ["auth testing"],
        }
        insight = ScanInsight.model_validate(data)
        assert insight.summary == "Discovered admin endpoints"
        assert len(insight.key_signals) == 1

    def test_phase_set_after(self):
        """Orchestrator sets phase after validation since LLM doesn't return it."""
        data = {"summary": "test", "key_signals": [], "recommended_focus": []}
        insight = ScanInsight.model_validate(data)
        insight.phase = "web_crawler"
        assert insight.phase == "web_crawler"


# ---------------------------------------------------------------------------
# Extra fields are ignored (Pydantic default)
# ---------------------------------------------------------------------------


class TestExtraFieldsIgnored:
    def test_extra_fields_ignored(self):
        data = {
            "reasoning": "test",
            "hypotheses": [{"path": "/x", "extra_field": "ignored"}],
            "unknown_key": "also ignored",
        }
        resp = HypothesisResponse.model_validate(data)
        assert len(resp.hypotheses) == 1
        assert resp.hypotheses[0].path == "/x"


# ---------------------------------------------------------------------------
# chat_structured() integration
# ---------------------------------------------------------------------------


class TestChatStructured:
    @pytest.mark.asyncio
    async def test_returns_validated_model(self):
        from src.ghost_hunter.clients.llm import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.chat_json = AsyncMock(
            return_value={
                "reasoning": "test",
                "hypotheses": [{"path": "/api/test", "method": "POST"}],
            }
        )

        result = await client.chat_structured(
            messages=[{"role": "user", "content": "test"}],
            response_model=HypothesisResponse,
            name="test_call",
        )

        assert isinstance(result, HypothesisResponse)
        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].path == "/api/test"
        assert result.hypotheses[0].method == "POST"

    @pytest.mark.asyncio
    async def test_raises_validation_error_on_bad_data(self):
        from src.ghost_hunter.clients.llm import LLMClient

        client = LLMClient.__new__(LLMClient)
        # VerifierAction requires endpoint_key, action, and pattern
        client.chat_json = AsyncMock(
            return_value={
                "actions": [{"bad_field": "test"}],
            }
        )

        with pytest.raises(ValidationError):
            await client.chat_structured(
                messages=[{"role": "user", "content": "test"}],
                response_model=VerifierBatchResponse,
            )

    @pytest.mark.asyncio
    async def test_passes_kwargs_through(self):
        from src.ghost_hunter.clients.llm import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.chat_json = AsyncMock(return_value={"hypotheses": []})

        await client.chat_structured(
            messages=[{"role": "user", "content": "test"}],
            response_model=HypothesisResponse,
            name="custom_name",
            temperature=0.5,
            max_tokens=1024,
        )

        client.chat_json.assert_called_once_with(
            messages=[{"role": "user", "content": "test"}],
            name="custom_name",
            temperature=0.5,
            max_tokens=1024,
        )
