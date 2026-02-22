"""Test fixtures and factories."""

from __future__ import annotations

import pytest

from src.ghost_hunter.models import (
    DiscoverySource,
    Endpoint,
    EndpointCategory,
    ParameterDetail,
    ScanState,
    SecuritySchemeInfo,
)


@pytest.fixture
def base_url() -> str:
    return "https://vulnbank.org"


@pytest.fixture
def scan_state(base_url: str) -> ScanState:
    return ScanState(target="vulnbank.org", base_url=base_url)


def make_endpoint(
    url: str = "https://vulnbank.org/api/v1/users",
    method: str = "GET",
    status_code: int | None = 200,
    requires_auth: bool | None = None,
    parameters: list[str] | None = None,
    parameter_details: list[ParameterDetail] | None = None,
    request_body_fields: list[str] | None = None,
    response_fields: list[str] | None = None,
    category: EndpointCategory | None = None,
    request_body_content_type: str | None = None,
    security_schemes: list[SecuritySchemeInfo] | None = None,
    response_body_snippet: str = "",
    discovered_by: DiscoverySource = DiscoverySource.CRAWL,
) -> Endpoint:
    """Factory for creating test endpoints with sensible defaults."""
    return Endpoint(
        url=url,
        method=method,
        status_code=status_code,
        discovered_by=discovered_by,
        parameters=parameters or [],
        parameter_details=parameter_details or [],
        request_body_fields=request_body_fields or [],
        response_fields=response_fields or [],
        requires_auth=requires_auth,
        category=category,
        request_body_content_type=request_body_content_type,
        security_schemes=security_schemes or [],
        response_body_snippet=response_body_snippet,
    )
