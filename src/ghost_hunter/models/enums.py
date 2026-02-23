"""Enumerations used across all models."""

from __future__ import annotations

from enum import Enum


class DiscoverySource(str, Enum):
    ROBOTS_TXT = "robots_txt"
    SITEMAP = "sitemap"
    HEADER_PROBE = "header_probe"
    CRAWL = "crawl"
    OPENAPI_SPEC = "openapi_spec"
    COMMON_PATH = "common_path"
    JS_ANALYSIS = "js_analysis"
    LLM_HYPOTHESIS = "llm_hypothesis"
    LLM_API_GUESS = "llm_api_guess"
    VERSION_ENUM = "version_enum"
    JS_LLM_ANALYSIS = "js_llm_analysis"


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EndpointCategory(str, Enum):
    REST_API = "rest_api"
    FORM_ACTION = "form_action"
    STATIC_ASSET = "static_asset"
    AUTH_ENDPOINT = "auth_endpoint"
    ADMIN_ENDPOINT = "admin_endpoint"
    DEBUG_ENDPOINT = "debug_endpoint"
    DOCUMENTATION = "documentation"
    HEALTH_CHECK = "health_check"
    GRAPHQL = "graphql"
    UNKNOWN = "unknown"


class ParamLocation(str, Enum):
    PATH = "path"
    QUERY = "query"
    HEADER = "header"
    COOKIE = "cookie"
    BODY = "body"


class VulnPattern(str, Enum):
    BOLA_IDOR = "bola_idor"
    MASS_ASSIGNMENT = "mass_assignment"
    SSRF = "ssrf"
    FILE_UPLOAD = "file_upload"
    JWT_WEAKNESS = "jwt_weakness"
    API_VERSION_CONFUSION = "api_version_confusion"
    RACE_CONDITION = "race_condition"
    PROMPT_INJECTION = "prompt_injection"
    INFO_DISCLOSURE = "info_disclosure"
    AUTH_BOUNDARY_GAP = "auth_boundary_gap"
    EXCESSIVE_DATA_EXPOSURE = "excessive_data_exposure"
    BROKEN_FUNCTION_LEVEL_AUTH = "broken_function_level_auth"
    CHAINED_VULNERABILITY = "chained_vulnerability"
