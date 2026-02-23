"""Golden datasets for prioritizer eval."""

from __future__ import annotations

from src.ghost_hunter.models.llm_responses import (
    AttackSurfaceItem,
    PrioritizationBatchResponse,
)

PRIORITIZER_CASES = [
    {
        "name": "financial_endpoints_critical",
        "description": "Transfer with IDOR is critical, health check is info",
        "endpoints": [
            {
                "url": "https://vulnbank.org/api/v1/transfer/{account_number}",
                "method": "POST",
                "status_code": 200,
                "parameters": ["account_number", "amount"],
                "requires_auth": True,
                "category": "rest_api",
            },
            {
                "url": "https://vulnbank.org/api/v1/balance",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
                "requires_auth": True,
                "category": "rest_api",
            },
            {
                "url": "https://vulnbank.org/health",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
                "requires_auth": False,
                "category": "health_check",
            },
        ],
        "vuln_indicators": {
            "POST https://vulnbank.org/api/v1/transfer/{account_number}": [
                {"pattern": "bola_idor", "confidence": "critical"},
                {"pattern": "race_condition", "confidence": "high"},
            ],
        },
        "llm_response": PrioritizationBatchResponse(
            reasoning="Financial transfer endpoint with IDOR and race condition is the highest priority target",
            attack_surface=[
                AttackSurfaceItem(
                    index=1, category="rest_api", risk_level="critical",
                    rationale="IDOR on fund transfer — attacker can transfer from any account",
                    suggested_tests=["curl -X POST https://vulnbank.org/api/v1/transfer/12345 -d '{\"amount\": 1}' -H 'Authorization: Bearer TOKEN'"],
                ),
                AttackSurfaceItem(
                    index=2, category="rest_api", risk_level="low",
                    rationale="Balance check with auth — limited attack surface",
                    suggested_tests=[],
                ),
                AttackSurfaceItem(
                    index=3, category="health_check", risk_level="info",
                    rationale="Standard health check endpoint",
                    suggested_tests=[],
                ),
            ],
        ),
        "expected_risk_levels": {
            "POST https://vulnbank.org/api/v1/transfer/{account_number}": "critical",
            "GET https://vulnbank.org/api/v1/balance": "low",
            "GET https://vulnbank.org/health": "info",
        },
    },
    {
        "name": "admin_debug_high_priority",
        "description": "Open admin panel and debug console both critical",
        "endpoints": [
            {
                "url": "https://vulnbank.org/admin/users",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
                "requires_auth": False,
                "category": "admin_endpoint",
            },
            {
                "url": "https://vulnbank.org/debug/console",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
                "requires_auth": False,
                "category": "debug_endpoint",
            },
            {
                "url": "https://vulnbank.org/api/v1/products",
                "method": "GET",
                "status_code": 200,
                "parameters": ["page"],
                "requires_auth": False,
                "category": "rest_api",
            },
        ],
        "vuln_indicators": {
            "GET https://vulnbank.org/admin/users": [
                {"pattern": "broken_function_level_auth", "confidence": "critical"},
            ],
            "GET https://vulnbank.org/debug/console": [
                {"pattern": "info_disclosure", "confidence": "high"},
            ],
        },
        "llm_response": PrioritizationBatchResponse(
            reasoning="Unauthenticated admin panel is critical — debug console may expose RCE",
            attack_surface=[
                AttackSurfaceItem(
                    index=1, category="admin_endpoint", risk_level="critical",
                    rationale="Admin user listing without authentication — full user data exposure",
                    suggested_tests=["curl https://vulnbank.org/admin/users"],
                ),
                AttackSurfaceItem(
                    index=2, category="debug_endpoint", risk_level="critical",
                    rationale="Debug console may expose interactive shell — potential RCE",
                    suggested_tests=["curl https://vulnbank.org/debug/console"],
                ),
                AttackSurfaceItem(
                    index=3, category="rest_api", risk_level="info",
                    rationale="Public product listing — no sensitive data",
                    suggested_tests=[],
                ),
            ],
        ),
        "expected_risk_levels": {
            "GET https://vulnbank.org/admin/users": "critical",
            "GET https://vulnbank.org/debug/console": "critical",
            "GET https://vulnbank.org/api/v1/products": "info",
        },
    },
    {
        "name": "deterministic_fallback",
        "description": "LLM down — fallback uses vuln indicators and category",
        "endpoints": [
            {
                "url": "https://vulnbank.org/api/v1/accounts/{id}",
                "method": "GET",
                "status_code": 200,
                "parameters": ["id"],
                "requires_auth": True,
                "category": "rest_api",
            },
            {
                "url": "https://vulnbank.org/static/logo.png",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
                "requires_auth": False,
                "category": "static_asset",
            },
        ],
        "vuln_indicators": {
            "GET https://vulnbank.org/api/v1/accounts/{id}": [
                {"pattern": "bola_idor", "confidence": "high"},
            ],
        },
        "llm_response": None,  # Simulates LLM failure
        "expected_risk_levels": {
            "GET https://vulnbank.org/api/v1/accounts/{id}": "high",
            "GET https://vulnbank.org/static/logo.png": "info",
        },
    },
]
