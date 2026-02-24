"""Tests for passive_recon helper functions."""

from __future__ import annotations

from src.ghost_hunter.agents.passive_recon import _is_concrete_technology


class TestIsConcreteTechnology:
    def test_real_tech_names_pass(self):
        for name in ["Flask 2.x", "JWT authentication", "Cloudflare CDN/WAF",
                      "OAuth 2.0", "nginx reverse proxy"]:
            assert _is_concrete_technology(name) is True, name

    def test_rejects_unknown_origin_hedge(self):
        """This was the most common LLM hallucination we saw in VulneraBank scans."""
        assert _is_concrete_technology("Unknown origin server technology (could be any stack)") is False

    def test_rejects_risk_assessment_disguised_as_tech(self):
        assert _is_concrete_technology("Potential for outdated or misconfigured web application") is False

    def test_rejects_auth_flow_description(self):
        assert _is_concrete_technology("Snippet 1: Basic authorization with client id and client secret") is False

    def test_could_be_triggers_filter(self):
        assert _is_concrete_technology("Origin server could be any stack") is False

    def test_risk_keyword_alone_is_enough(self):
        assert _is_concrete_technology("High risk of data exposure") is False

    def test_too_long_is_clearly_a_description(self):
        assert _is_concrete_technology("A" * 81) is False
        assert _is_concrete_technology("A" * 80) is True  # boundary
