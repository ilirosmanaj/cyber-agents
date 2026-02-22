"""Tests for parsing helpers, the orchestrator DAG, and config resolution."""

from __future__ import annotations

import pytest

from src.ghost_hunter.agents.api_discovery import _extract_field_names
from src.ghost_hunter.agents.js_analyzer import JS_API_PATTERNS
from src.ghost_hunter.agents.web_crawler import _normalize_url
from src.ghost_hunter.config import Settings, _PROVIDER_DEFAULTS
from src.ghost_hunter.orchestrator import AGENT_DEPS, _resolve_waves


class TestExtractFieldNames:
    def test_flat_properties(self):
        """Flat properties come back as a list of names."""
        schema = {"properties": {"id": {"type": "integer"}, "name": {"type": "string"}}}
        assert _extract_field_names(schema) == ["id", "name"]

    def test_nested_object(self):
        """Nested objects should include both the parent and child field names."""
        schema = {
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                },
            },
        }
        result = _extract_field_names(schema)
        assert "user" in result
        assert "email" in result

    def test_ref_resolution(self):
        """$ref pointers should resolve against the definitions dict."""
        schema = {"$ref": "#/definitions/User"}
        definitions = {"User": {"properties": {"id": {"type": "integer"}}}}
        assert _extract_field_names(schema, definitions) == ["id"]

    def test_allof_composition(self):
        """allOf merges fields from all sub-schemas."""
        schema = {
            "allOf": [
                {"properties": {"id": {"type": "integer"}}},
                {"properties": {"name": {"type": "string"}}},
            ],
        }
        result = _extract_field_names(schema)
        assert "id" in result
        assert "name" in result

    def test_array_items(self):
        """Array schemas should dig into items for field names."""
        schema = {
            "type": "array",
            "items": {"properties": {"title": {"type": "string"}}},
        }
        assert _extract_field_names(schema) == ["title"]

    def test_max_depth_zero(self):
        """When max_depth is 0, return nothing regardless of schema content."""
        schema = {"properties": {"id": {"type": "integer"}}}
        assert _extract_field_names(schema, max_depth=0) == []

    def test_empty_schema(self):
        """None and {} both produce an empty list."""
        assert _extract_field_names(None) == []
        assert _extract_field_names({}) == []

    def test_circular_ref_stops(self):
        """Self-referencing $ref shouldn't infinite-loop; max_depth cuts it off."""
        definitions = {"A": {"$ref": "#/definitions/A"}}
        schema = {"$ref": "#/definitions/A"}
        result = _extract_field_names(schema, definitions, max_depth=5)
        assert result == []


class TestNormalizeUrl:
    def test_strips_fragment(self):
        """#fragment gets stripped."""
        assert _normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_strips_trailing_slash(self):
        """Trailing slash gets stripped for dedup."""
        assert _normalize_url("https://example.com/page/") == "https://example.com/page"

    def test_preserves_query(self):
        """Query parameters should survive normalization."""
        assert _normalize_url("https://example.com/page?q=1") == "https://example.com/page?q=1"

    def test_root_path(self):
        """Root path '/' should not be stripped to empty string."""
        assert _normalize_url("https://example.com/") == "https://example.com/"

    def test_already_normalized(self):
        """A clean URL should pass through unchanged."""
        url = "https://example.com/api/v1/users"
        assert _normalize_url(url) == url


class TestJsApiPatterns:
    @staticmethod
    def _find_all(text: str) -> list[str]:
        results = []
        for pattern in JS_API_PATTERNS:
            results.extend(m.group(1) for m in pattern.finditer(text))
        return results

    def test_fetch(self):
        """fetch('/api/users') should extract /api/users."""
        matches = self._find_all("fetch('/api/users')")
        assert "/api/users" in matches

    def test_axios_post(self):
        """axios.post('/api/login') should match too."""
        matches = self._find_all("axios.post('/api/login')")
        assert "/api/login" in matches

    def test_endpoint_assignment(self):
        """Variable assignments like endpoint = '/v2/...' get picked up."""
        matches = self._find_all("endpoint = '/v2/transactions'")
        assert "/v2/transactions" in matches

    def test_api_path_literal(self):
        """Bare '/api/...' string literals should match."""
        matches = self._find_all("const x = '/api/accounts/123'")
        assert "/api/accounts/123" in matches

    def test_template_literal(self):
        """Template literals like `/api/users/${id}/orders` should match."""
        matches = self._find_all("`/api/users/${id}/orders`")
        assert any("${id}" in m for m in matches)

    def test_no_match(self):
        """Plain JS with no API paths should yield nothing."""
        matches = self._find_all("const x = 42;")
        assert matches == []


class TestResolveWaves:
    def test_wave3_parallel(self):
        """api_discovery and js_analyzer should land in the same wave."""
        waves = _resolve_waves(AGENT_DEPS)
        wave3 = waves[2]
        assert "api_discovery" in wave3
        assert "js_analyzer" in wave3

    def test_all_agents_appear_once(self):
        """Every agent in AGENT_DEPS should appear in exactly one wave."""
        waves = _resolve_waves(AGENT_DEPS)
        all_agents = [name for wave in waves for name in wave]
        assert sorted(all_agents) == sorted(AGENT_DEPS.keys())

    def test_deps_before_dependents(self):
        """Each agent must be scheduled after all its dependencies."""
        waves = _resolve_waves(AGENT_DEPS)
        order = {name: i for i, wave in enumerate(waves) for name in wave}
        for name, deps in AGENT_DEPS.items():
            for dep in deps:
                assert order[dep] < order[name], f"{dep} must run before {name}"

    def test_circular_dependency_raises(self):
        """Circular deps should blow up with ValueError."""
        circular = {"a": ["b"], "b": ["a"]}
        with pytest.raises(ValueError, match="Circular dependency"):
            _resolve_waves(circular)


class TestResolveLlmDefaults:
    def test_known_provider_returns_defaults(self):
        """Each supported provider should resolve to its default URL and model."""
        for provider in _PROVIDER_DEFAULTS:
            s = Settings(llm_provider=provider, llm_api_key="test")
            base_url, model = s.resolve_llm_defaults()
            expected_url, expected_model = _PROVIDER_DEFAULTS[provider]
            assert base_url == expected_url
            assert model == expected_model

    def test_explicit_overrides_take_priority(self):
        """When llm_base_url and llm_model are set, they override provider defaults."""
        s = Settings(
            llm_provider="groq",
            llm_api_key="test",
            llm_base_url="http://custom:8000/v1",
            llm_model="my-model",
        )
        base_url, model = s.resolve_llm_defaults()
        assert base_url == "http://custom:8000/v1"
        assert model == "my-model"

    def test_unknown_provider_raises(self):
        """A typo in the provider name should raise, not silently fall back."""
        s = Settings(llm_provider="anthropic", llm_api_key="test")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            s.resolve_llm_defaults()
