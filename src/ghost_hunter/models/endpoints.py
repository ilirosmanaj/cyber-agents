"""Endpoint, Finding, and TechFingerprint models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.ghost_hunter.models.enums import (
    DiscoverySource,
    EndpointCategory,
    ParamLocation,
    RiskLevel,
)


class ParameterDetail(BaseModel):
    name: str
    location: ParamLocation
    param_type: str = "string"
    required: bool = False


class SecuritySchemeInfo(BaseModel):
    scheme_type: str  # e.g. "http", "apiKey", "oauth2"
    scheme_name: str = ""
    location: str = ""  # e.g. "header", "query"
    bearer_format: str = ""


class Endpoint(BaseModel):
    url: str
    method: str = "GET"
    status_code: int | None = None
    content_type: str | None = None
    discovered_by: DiscoverySource
    parameters: list[str] = Field(default_factory=list)
    requires_auth: bool | None = None
    api_version: str | None = None
    notes: str = ""
    category: EndpointCategory | None = None
    # rich parameter info from OpenAPI
    parameter_details: list[ParameterDetail] = Field(default_factory=list)
    request_body_content_type: str | None = None
    request_body_fields: list[str] = Field(default_factory=list)
    response_fields: list[str] = Field(default_factory=list)
    security_schemes: list[SecuritySchemeInfo] = Field(default_factory=list)
    response_headers: dict[str, str] = Field(default_factory=dict)
    response_body_snippet: str = ""  # first 2000 chars of response (for analysis)


class Finding(BaseModel):
    agent_name: str
    finding_type: str
    title: str
    detail: str
    severity: RiskLevel = RiskLevel.INFO
    evidence: str = ""
    validated: bool | None = None  # None=untested, True=confirmed, False=refuted
    validation_evidence: str = ""
    verification_status: str = ""  # "", "consistent", "conflicting"
    verification_note: str = ""


class TechFingerprint(BaseModel):
    server: str | None = None
    frameworks: list[str] = Field(default_factory=list)
    security_headers: dict[str, str] = Field(default_factory=dict)
    missing_security_headers: list[str] = Field(default_factory=list)
    cookies: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
