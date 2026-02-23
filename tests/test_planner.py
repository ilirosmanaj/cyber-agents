"""Tests for ScanStrategy model and planner integration."""

from __future__ import annotations

import pytest

from src.ghost_hunter.agents.hypothesis import HypothesisAgent
from src.ghost_hunter.models.strategy import ScanStrategy
from src.ghost_hunter.models.scan import ScanState
from src.ghost_hunter.orchestrator import AGENT_DEPS, _resolve_waves


class TestScanStrategyModel:
    def test_create_strategy(self):
        """ScanStrategy should accept all fields."""
        strategy = ScanStrategy(
            focus_areas=["graphql", "auth_flows"],
            skip_agents=["js_analyzer"],
            extra_paths_to_try=["/api/internal/users", "/admin/config"],
            tech_hypotheses=["Spring Boot", "JWT auth"],
            scan_depth="deep",
            priority_patterns=["/api/v1/transfer", "/api/v1/payment"],
        )
        assert "graphql" in strategy.focus_areas
        assert "js_analyzer" in strategy.skip_agents
        assert len(strategy.extra_paths_to_try) == 2
        assert strategy.scan_depth == "deep"

    def test_strategy_defaults(self):
        """All list fields default to empty, scan_depth to 'normal'."""
        strategy = ScanStrategy()
        assert strategy.focus_areas == []
        assert strategy.skip_agents == []
        assert strategy.extra_paths_to_try == []
        assert strategy.tech_hypotheses == []
        assert strategy.scan_depth == "normal"
        assert strategy.priority_patterns == []


class TestScanStateWithStrategy:
    def test_scan_state_strategy_none_by_default(self, scan_state: ScanState):
        """ScanState.scan_strategy should be None by default."""
        assert scan_state.scan_strategy is None

    def test_scan_state_strategy_assignment(self, scan_state: ScanState):
        """Strategy can be assigned to scan state."""
        strategy = ScanStrategy(focus_areas=["auth_flows"])
        scan_state.scan_strategy = strategy
        assert scan_state.scan_strategy is not None
        assert "auth_flows" in scan_state.scan_strategy.focus_areas


class TestPlannerDAGIntegration:
    def test_planner_not_in_agent_deps(self):
        """Planner should NOT be in AGENT_DEPS — it's called explicitly."""
        assert "planner" not in AGENT_DEPS

    def test_dag_still_valid_with_verifier(self):
        """DAG should resolve cleanly with all agents including verifier."""
        waves = _resolve_waves(AGENT_DEPS)
        all_agents = [name for wave in waves for name in wave]
        assert "verifier" in all_agents
        assert "prioritizer" in all_agents

    def test_verifier_before_prioritizer(self):
        """Verifier must run before prioritizer in the DAG."""
        waves = _resolve_waves(AGENT_DEPS)
        order = {name: i for i, wave in enumerate(waves) for name in wave}
        assert order["verifier"] < order["prioritizer"]

    def test_verifier_after_vuln_analyzer(self):
        """Verifier must run after vuln_analyzer."""
        waves = _resolve_waves(AGENT_DEPS)
        order = {name: i for i, wave in enumerate(waves) for name in wave}
        assert order["vuln_analyzer"] < order["verifier"]


class TestStrategyContextBuilding:
    def test_hypothesis_strategy_context_empty(self, scan_state: ScanState):
        """When no strategy exists, strategy context should be empty."""
        result = HypothesisAgent._build_strategy_context(scan_state)
        assert result == ""

    def test_hypothesis_strategy_context_with_strategy(self, scan_state: ScanState):
        """With a strategy, context should include focus areas and tech hypotheses."""
        scan_state.scan_strategy = ScanStrategy(
            focus_areas=["financial_endpoints"],
            tech_hypotheses=["Django REST Framework"],
            priority_patterns=["/api/v1/transfer"],
        )
        result = HypothesisAgent._build_strategy_context(scan_state)
        assert "financial_endpoints" in result
        assert "Django REST Framework" in result
        assert "/api/v1/transfer" in result
        assert "Scan Strategy:" in result
