"""Tests for ScanInsight model and insights integration."""

from __future__ import annotations

import pytest

from src.ghost_hunter.models.insights import ScanInsight
from src.ghost_hunter.models.scan import ScanState


class TestScanInsightModel:
    def test_create_insight(self):
        """ScanInsight should accept all fields."""
        insight = ScanInsight(
            phase="crawl",
            summary="Discovered 15 endpoints, 3 with auth requirements.",
            key_signals=["JWT bearer auth", "REST API structure"],
            recommended_focus=["auth_flows", "financial_endpoints"],
        )
        assert insight.phase == "crawl"
        assert "15 endpoints" in insight.summary
        assert len(insight.key_signals) == 2
        assert "auth_flows" in insight.recommended_focus

    def test_insight_defaults(self):
        """Key signals and recommended focus default to empty lists."""
        insight = ScanInsight(phase="recon", summary="Basic fingerprint done.")
        assert insight.key_signals == []
        assert insight.recommended_focus == []


class TestInsightsContext:
    def test_no_insights_returns_default(self, scan_state: ScanState):
        """When no insights exist, returns placeholder text."""
        result = scan_state.insights_context()
        assert result == "No prior insights."

    def test_single_insight_formatted(self, scan_state: ScanState):
        """A single insight should format phase, summary, signals, and focus."""
        scan_state.scan_insights.append(
            ScanInsight(
                phase="crawl",
                summary="Found 10 pages and 5 forms.",
                key_signals=["form-based auth"],
                recommended_focus=["auth_endpoints"],
            )
        )
        result = scan_state.insights_context()
        assert "[crawl]" in result
        assert "Found 10 pages" in result
        assert "form-based auth" in result
        assert "auth_endpoints" in result

    def test_multiple_insights_accumulated(self, scan_state: ScanState):
        """Multiple insights should all appear in context."""
        scan_state.scan_insights.append(
            ScanInsight(phase="crawl", summary="Crawl complete.")
        )
        scan_state.scan_insights.append(
            ScanInsight(phase="discovery", summary="API spec found.")
        )
        result = scan_state.insights_context()
        assert "[crawl]" in result
        assert "[discovery]" in result

    def test_max_insights_truncation(self, scan_state: ScanState):
        """insights_context should respect max_insights parameter."""
        for i in range(10):
            scan_state.scan_insights.append(
                ScanInsight(phase=f"phase_{i}", summary=f"Insight {i}")
            )
        result = scan_state.insights_context(max_insights=3)
        # Should only contain the last 3 insights
        assert "[phase_7]" in result
        assert "[phase_8]" in result
        assert "[phase_9]" in result
        assert "[phase_0]" not in result

    def test_insight_without_signals(self, scan_state: ScanState):
        """Insight with no signals/focus should still format cleanly."""
        scan_state.scan_insights.append(
            ScanInsight(phase="analysis", summary="Analysis done.")
        )
        result = scan_state.insights_context()
        assert "[analysis]" in result
        assert "Signals:" not in result
        assert "Focus:" not in result


class TestScanStateWithInsights:
    def test_scan_state_has_insights_field(self, scan_state: ScanState):
        """ScanState should have scan_insights as an empty list by default."""
        assert scan_state.scan_insights == []

    def test_scan_state_summary_unchanged(self, scan_state: ScanState):
        """Adding insights shouldn't break the existing summary method."""
        summary = scan_state.summary()
        assert "Target: vulnbank.org" in summary
        assert "Endpoints discovered: 0" in summary
