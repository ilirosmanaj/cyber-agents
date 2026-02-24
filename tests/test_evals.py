"""Eval tests — run agent pipelines against golden datasets."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ghost_hunter.agents.classifier import ClassifierAgent
from src.ghost_hunter.agents.prioritizer import PrioritizerAgent
from src.ghost_hunter.agents.vuln_analyzer import VulnPatternAnalyzer
from src.ghost_hunter.models import (
    DiscoverySource,
    Endpoint,
    EndpointCategory,
    RiskLevel,
    ScanState,
    VulnIndicator,
    VulnPattern,
)
from tests.evals.golden_classifier import CLASSIFIER_CASES
from tests.evals.golden_vuln_analyzer import VULN_ANALYZER_CASES
from tests.evals.golden_prioritizer import PRIORITIZER_CASES
from tests.evals.scoring import (
    classification_accuracy,
    vuln_precision_recall,
    risk_rank_correlation,
)

# minimum acceptable score for eval tests to pass
_MIN_EVAL_SCORE = 0.8


def _make_endpoint(data: dict) -> Endpoint:
    category = data.get("category")
    if category:
        category = EndpointCategory(category)
    return Endpoint(
        url=data["url"],
        method=data["method"],
        status_code=data.get("status_code"),
        parameters=data.get("parameters", []),
        request_body_fields=data.get("request_body_fields", []),
        requires_auth=data.get("requires_auth"),
        category=category,
        discovered_by=DiscoverySource.CRAWL,
    )


def _build_state_with_endpoints(case: dict) -> ScanState:
    state = ScanState(target="vulnbank.org", base_url="https://vulnbank.org")
    for ep_data in case["endpoints"]:
        state.add_endpoint(_make_endpoint(ep_data))
    return state


def _mock_llm(response: object) -> MagicMock:
    llm = MagicMock()
    llm.chat_structured = AsyncMock(return_value=response)
    return llm


def _mock_prompt_registry() -> MagicMock:
    reg = MagicMock()
    reg.get.return_value = MagicMock(system_prompt="mock prompt")
    return reg


# -- Classifier evals -------------------------------------------------------

@pytest.mark.parametrize("case", CLASSIFIER_CASES, ids=lambda c: c["name"])
async def test_classifier_eval(case):
    """Mock LLM response should produce correct endpoint categories."""
    state = _build_state_with_endpoints(case)
    agent = ClassifierAgent(
        http_client=MagicMock(),
        llm_client=_mock_llm(case["llm_response"]),
        prompt_registry=_mock_prompt_registry(),
    )

    await agent.run(state)

    predicted: dict[str, tuple[str, bool | None]] = {}
    for key, ep in state.endpoints.items():
        cat = ep.category.value if ep.category else "unknown"
        predicted[key] = (cat, ep.requires_auth)

    accuracy = classification_accuracy(predicted, case["expected"])
    assert accuracy >= _MIN_EVAL_SCORE, (
        f"Classification accuracy {accuracy:.2f} below {_MIN_EVAL_SCORE} for '{case['name']}'\n"
        f"Predicted: {predicted}\nExpected: {case['expected']}"
    )


# -- Vuln analyzer evals ----------------------------------------------------

@pytest.mark.parametrize("case", VULN_ANALYZER_CASES, ids=lambda c: c["name"])
async def test_vuln_analyzer_eval(case):
    """Indicators should be merged/suppressed correctly given golden LLM output."""
    state = _build_state_with_endpoints(case)

    for ep_key, indicator_dicts in case.get("initial_indicators", {}).items():
        state.vuln_indicators[ep_key] = [
            VulnIndicator(
                pattern=VulnPattern(d["pattern"]),
                confidence=RiskLevel(d["confidence"]),
                evidence=d.get("evidence", ""),
                description=d.get("description", ""),
            )
            for d in indicator_dicts
        ]

    agent = VulnPatternAnalyzer(
        http_client=MagicMock(),
        llm_client=_mock_llm(case["llm_response"]),
        prompt_registry=_mock_prompt_registry(),
    )

    await agent.run(state)

    active_indicators = []
    suppressed_indicators = []
    for ep_key, indicators in state.vuln_indicators.items():
        for ind in indicators:
            tag = f"{ind.pattern.value} {ep_key}"
            (suppressed_indicators if ind.suppressed else active_indicators).append(tag)

    for expected in case["expected_indicators"]:
        assert expected in active_indicators, (
            f"Expected indicator '{expected}' not found.\n"
            f"Active: {active_indicators}"
        )

    for expected_sup in case["expected_suppressions"]:
        assert expected_sup in suppressed_indicators, (
            f"Expected suppression '{expected_sup}' not found.\n"
            f"Suppressed: {suppressed_indicators}"
        )

    _precision, recall, _f1 = vuln_precision_recall(
        active_indicators, case["expected_indicators"]
    )
    assert recall >= _MIN_EVAL_SCORE, (
        f"Vuln recall {recall:.2f} below {_MIN_EVAL_SCORE} for '{case['name']}'"
    )


# -- Prioritizer evals ------------------------------------------------------

@pytest.mark.parametrize("case", PRIORITIZER_CASES, ids=lambda c: c["name"])
async def test_prioritizer_eval(case):
    """Risk levels should match expectations (LLM path and fallback path)."""
    state = _build_state_with_endpoints(case)

    for ep_key, indicator_dicts in case.get("vuln_indicators", {}).items():
        state.vuln_indicators[ep_key] = [
            VulnIndicator(
                pattern=VulnPattern(d["pattern"]),
                confidence=RiskLevel(d["confidence"]),
                evidence="",
                description="",
            )
            for d in indicator_dicts
        ]

    if case["llm_response"] is not None:
        llm = _mock_llm(case["llm_response"])
    else:
        # LLM failure path — exercises deterministic fallback
        llm = MagicMock()
        llm.chat_structured = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    agent = PrioritizerAgent(
        http_client=MagicMock(),
        llm_client=llm,
        prompt_registry=_mock_prompt_registry(),
    )
    await agent.run(state)

    for ep_key, expected_risk in case["expected_risk_levels"].items():
        matching = [
            entry for entry in state.attack_surface
            if state.endpoint_key(entry.endpoint.method, entry.endpoint.url) == ep_key
        ]
        assert matching, (
            f"No attack surface entry for '{ep_key}'\n"
            f"Available: {[state.endpoint_key(e.endpoint.method, e.endpoint.url) for e in state.attack_surface]}"
        )
        actual_risk = matching[0].risk_level.value
        assert actual_risk == expected_risk, (
            f"Risk level mismatch for '{ep_key}': "
            f"expected '{expected_risk}', got '{actual_risk}'"
        )


# -- Scoring functions -------------------------------------------------------

class TestScoring:
    def test_classification_accuracy_perfect(self):
        """Perfect predictions → 1.0 accuracy."""
        expected = {"a": ("rest_api", True), "b": ("auth_endpoint", False)}
        predicted = {"a": ("rest_api", True), "b": ("auth_endpoint", False)}
        assert classification_accuracy(predicted, expected) == 1.0

    def test_classification_accuracy_half(self):
        """When one of two endpoints is wrong, accuracy should be 0.5."""
        expected = {"a": ("rest_api", True), "b": ("auth_endpoint", False)}
        predicted = {"a": ("rest_api", True), "b": ("rest_api", False)}
        assert classification_accuracy(predicted, expected) == 0.5

    def test_classification_accuracy_empty(self):
        """Empty sets should return perfect accuracy (nothing to get wrong)."""
        assert classification_accuracy({}, {}) == 1.0

    def test_vuln_precision_recall_perfect(self):
        """All vulns matched exactly → P/R/F1 all 1.0."""
        items = ["bola_idor GET /a", "ssrf POST /b"]
        p, r, f1 = vuln_precision_recall(items, items)
        assert p == 1.0 and r == 1.0 and f1 == 1.0

    def test_vuln_precision_recall_partial(self):
        """Extra false positive drops precision but recall stays at 1.0."""
        predicted = ["bola_idor GET /a", "ssrf POST /b", "extra GET /c"]
        expected = ["bola_idor GET /a", "ssrf POST /b"]
        p, r, f1 = vuln_precision_recall(predicted, expected)
        assert r == 1.0
        assert p == pytest.approx(2 / 3, abs=0.01)

    def test_vuln_precision_recall_empty(self):
        """Both empty → perfect score (nothing predicted, nothing expected)."""
        p, r, f1 = vuln_precision_recall([], [])
        assert p == 1.0 and r == 1.0 and f1 == 1.0

    def test_risk_rank_correlation_perfect(self):
        """Identical ordering → correlation 1.0."""
        items = ["a", "b", "c"]
        assert risk_rank_correlation(items, items) == 1.0

    def test_risk_rank_correlation_reversed(self):
        """Fully reversed ordering should produce correlation below 0.5."""
        predicted = ["c", "b", "a"]
        expected = ["a", "b", "c"]
        corr = risk_rank_correlation(predicted, expected)
        assert corr < 0.5
