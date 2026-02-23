"""Golden datasets for vuln analyzer eval."""

from __future__ import annotations

from src.ghost_hunter.models.llm_responses import (
    VulnAnalysisBatchResponse,
    VulnConfidenceAdjustment,
    VulnEndpointAnalysis,
    VulnNewIndicator,
    VulnSuppression,
)

VULN_ANALYZER_CASES = [
    {
        "name": "idor_upgrade_on_financial_endpoint",
        "description": "BOLA on /accounts should be bumped to critical",
        "endpoints": [
            {
                "url": "https://vulnbank.org/api/v1/accounts/{account_id}",
                "method": "GET",
                "status_code": 200,
                "parameters": ["account_id"],
                "requires_auth": True,
                "category": "rest_api",
            },
        ],
        "initial_indicators": {
            "GET https://vulnbank.org/api/v1/accounts/{account_id}": [
                {"pattern": "bola_idor", "confidence": "high", "evidence": "Path contains object ID template"},
            ],
        },
        "llm_response": VulnAnalysisBatchResponse(
            reasoning="Financial endpoint with account_id parameter — BOLA here gives access to another user's financial data",
            endpoint_analyses=[
                VulnEndpointAnalysis(
                    endpoint_key="GET https://vulnbank.org/api/v1/accounts/{account_id}",
                    confidence_adjustments=[
                        VulnConfidenceAdjustment(
                            original_pattern="bola_idor",
                            new_confidence="critical",
                            reason="Financial endpoint — accessing another user's account gives access to their balance and transaction history",
                        ),
                    ],
                ),
            ],
        ),
        "expected_indicators": [
            "bola_idor GET https://vulnbank.org/api/v1/accounts/{account_id}",
        ],
        "expected_suppressions": [],
    },
    {
        "name": "suppress_false_positive_swagger",
        "description": "Swagger docs are intentional, suppress the info_disclosure",
        "endpoints": [
            {
                "url": "https://vulnbank.org/swagger/v1/swagger.json",
                "method": "GET",
                "status_code": 200,
                "parameters": [],
                "requires_auth": False,
                "category": "documentation",
            },
        ],
        "initial_indicators": {
            "GET https://vulnbank.org/swagger/v1/swagger.json": [
                {"pattern": "info_disclosure", "confidence": "high", "evidence": "Accessible sensitive path: /swagger/v1/swagger.json"},
            ],
        },
        "llm_response": VulnAnalysisBatchResponse(
            reasoning="Swagger documentation is intentionally exposed — not a security issue",
            endpoint_analyses=[
                VulnEndpointAnalysis(
                    endpoint_key="GET https://vulnbank.org/swagger/v1/swagger.json",
                    suppressions=[
                        VulnSuppression(
                            original_pattern="info_disclosure",
                            reason="OpenAPI spec is intentionally public documentation, not accidental info disclosure",
                        ),
                    ],
                ),
            ],
        ),
        "expected_indicators": [],
        "expected_suppressions": [
            "info_disclosure GET https://vulnbank.org/swagger/v1/swagger.json",
        ],
    },
    {
        "name": "chained_vulnerability_detection",
        "description": "No auth + BOLA = chained_vulnerability at critical",
        "endpoints": [
            {
                "url": "https://vulnbank.org/api/v1/users/{user_id}",
                "method": "GET",
                "status_code": 200,
                "parameters": ["user_id"],
                "requires_auth": False,
                "category": "rest_api",
            },
        ],
        "initial_indicators": {
            "GET https://vulnbank.org/api/v1/users/{user_id}": [
                {"pattern": "bola_idor", "confidence": "high", "evidence": "Path contains object ID template"},
                {"pattern": "auth_boundary_gap", "confidence": "high", "evidence": "No auth required: GET /api/v1/users/{user_id}"},
            ],
        },
        "llm_response": VulnAnalysisBatchResponse(
            reasoning="Unauthenticated IDOR — critical chain: any caller can enumerate user records",
            endpoint_analyses=[
                VulnEndpointAnalysis(
                    endpoint_key="GET https://vulnbank.org/api/v1/users/{user_id}",
                    new_indicators=[
                        VulnNewIndicator(
                            pattern="chained_vulnerability",
                            confidence="critical",
                            evidence="auth_boundary_gap + bola_idor = unauthenticated user data access",
                            description="Unauthenticated IDOR on /api/v1/users/{user_id} allows any caller to enumerate all user records",
                            chain_with=None,
                        ),
                    ],
                ),
            ],
        ),
        "expected_indicators": [
            "bola_idor GET https://vulnbank.org/api/v1/users/{user_id}",
            "auth_boundary_gap GET https://vulnbank.org/api/v1/users/{user_id}",
            "chained_vulnerability GET https://vulnbank.org/api/v1/users/{user_id}",
        ],
        "expected_suppressions": [],
    },
    {
        "name": "semantic_variant_detection",
        "description": "subscription_level isn't in the regex set but it's a mass-assign field",
        "endpoints": [
            {
                "url": "https://vulnbank.org/api/v1/profile",
                "method": "PUT",
                "status_code": 200,
                "parameters": ["name", "email"],
                "request_body_fields": ["name", "email", "subscription_level"],
                "requires_auth": True,
                "category": "rest_api",
            },
        ],
        "initial_indicators": {},
        "llm_response": VulnAnalysisBatchResponse(
            reasoning="subscription_level is a privilege-escalation field — regex didn't catch it because it's not in the standard set",
            endpoint_analyses=[
                VulnEndpointAnalysis(
                    endpoint_key="PUT https://vulnbank.org/api/v1/profile",
                    new_indicators=[
                        VulnNewIndicator(
                            pattern="mass_assignment",
                            confidence="high",
                            evidence="Body field subscription_level can escalate account tier",
                            description="PUT /api/v1/profile accepts subscription_level in request body — user could upgrade their plan without payment",
                        ),
                    ],
                ),
            ],
        ),
        "expected_indicators": [
            "mass_assignment PUT https://vulnbank.org/api/v1/profile",
        ],
        "expected_suppressions": [],
    },
]
