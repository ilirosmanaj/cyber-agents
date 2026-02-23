"""Golden datasets for classifier eval."""

from __future__ import annotations

from src.ghost_hunter.models.llm_responses import (
    ClassificationBatchResponse,
    EndpointClassification,
)

CLASSIFIER_CASES = [
    {
        "name": "rest_api_with_auth",
        "description": "Banking REST endpoints, all need auth",
        "endpoints": [
            {
                "url": "https://vulnbank.org/api/v1/users",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
            },
            {
                "url": "https://vulnbank.org/api/v1/accounts/{id}",
                "method": "GET",
                "status_code": 200,
                "parameters": ["id"],
            },
            {
                "url": "https://vulnbank.org/api/v1/transfer",
                "method": "POST",
                "status_code": 200,
                "parameters": ["amount", "to_account"],
            },
        ],
        "llm_response": ClassificationBatchResponse(
            reasoning="REST API endpoints for a banking application",
            classifications=[
                EndpointClassification(index=1, category="rest_api", requires_auth=True),
                EndpointClassification(index=2, category="rest_api", requires_auth=True),
                EndpointClassification(index=3, category="rest_api", requires_auth=True),
            ],
        ),
        "expected": {
            "GET https://vulnbank.org/api/v1/users": ("rest_api", True),
            "GET https://vulnbank.org/api/v1/accounts/{id}": ("rest_api", True),
            "POST https://vulnbank.org/api/v1/transfer": ("rest_api", True),
        },
    },
    {
        "name": "mixed_categories",
        "description": "LLM receives all endpoints; auth/admin/docs classified by LLM",
        "endpoints": [
            {
                "url": "https://vulnbank.org/login",
                "method": "POST",
                "status_code": 200,
                "parameters": ["username", "password"],
            },
            {
                "url": "https://vulnbank.org/admin/dashboard",
                "method": "GET",
                "status_code": 403,
                "parameters": [],
            },
            {
                "url": "https://vulnbank.org/docs",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
            },
        ],
        "llm_response": ClassificationBatchResponse(
            reasoning="Login, admin, and docs endpoints — standard web app pattern",
            classifications=[
                EndpointClassification(index=1, category="auth_endpoint", requires_auth=None),
                EndpointClassification(index=2, category="admin_endpoint", requires_auth=True),
                EndpointClassification(index=3, category="documentation", requires_auth=None),
            ],
        ),
        # LLM classifies all; auth inference sets requires_auth=True for 403 status
        "expected": {
            "POST https://vulnbank.org/login": ("auth_endpoint", None),
            "GET https://vulnbank.org/admin/dashboard": ("admin_endpoint", True),
            "GET https://vulnbank.org/docs": ("documentation", None),
        },
    },
    {
        "name": "graphql_and_llm_rest",
        "description": "LLM receives all including graphql; LLM classifies both",
        "endpoints": [
            {
                "url": "https://vulnbank.org/graphql",
                "method": "POST",
                "status_code": 200,
                "parameters": ["query"],
            },
            {
                "url": "https://vulnbank.org/api/v1/products",
                "method": "GET",
                "status_code": 200,
                "parameters": ["page", "limit"],
            },
        ],
        "llm_response": ClassificationBatchResponse(
            reasoning="GraphQL endpoint and public REST endpoint",
            classifications=[
                EndpointClassification(index=1, category="graphql", requires_auth=None),
                EndpointClassification(index=2, category="rest_api", requires_auth=False),
            ],
        ),
        "expected": {
            "POST https://vulnbank.org/graphql": ("graphql", None),
            "GET https://vulnbank.org/api/v1/products": ("rest_api", False),
        },
    },
    {
        "name": "pre_classification_static_assets",
        "description": "LLM receives static assets and health check but rule overrides correct to STATIC_ASSET/HEALTH_CHECK",
        "endpoints": [
            {
                "url": "https://vulnbank.org/static/app.js",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
            },
            {
                "url": "https://vulnbank.org/health",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
            },
        ],
        # LLM might misclassify these, but rule overrides will correct them
        "llm_response": ClassificationBatchResponse(
            reasoning="Static JS file and health check endpoint",
            classifications=[
                EndpointClassification(index=1, category="rest_api", requires_auth=False),
                EndpointClassification(index=2, category="rest_api", requires_auth=False),
            ],
        ),
        # Rule overrides ensure correct category; LLM's auth assessment is kept
        "expected": {
            "GET https://vulnbank.org/static/app.js": ("static_asset", False),
            "GET https://vulnbank.org/health": ("health_check", False),
        },
    },
]
