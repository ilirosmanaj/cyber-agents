"""Attack surface entry model for prioritized results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.ghost_hunter.models.endpoints import Endpoint
from src.ghost_hunter.models.enums import EndpointCategory, RiskLevel, VulnPattern


class VulnIndicator(BaseModel):
    pattern: VulnPattern
    confidence: RiskLevel
    evidence: str
    description: str
    llm_enhanced: bool = False
    suppressed: bool = False
    chain_id: str | None = None


class AttackSurfaceEntry(BaseModel):
    endpoint: Endpoint
    category: EndpointCategory
    risk_level: RiskLevel
    priority_rank: int
    rationale: str
    suggested_tests: list[str] = Field(default_factory=list)
    vuln_indicators: list[VulnIndicator] = Field(default_factory=list)
