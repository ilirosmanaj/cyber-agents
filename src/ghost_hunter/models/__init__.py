"""Models package — re-exports all model classes for convenient access."""

from src.ghost_hunter.models.attack_surface import AttackSurfaceEntry, VulnIndicator
from src.ghost_hunter.models.endpoints import (
    Endpoint,
    Finding,
    ParameterDetail,
    SecuritySchemeInfo,
    TechFingerprint,
)
from src.ghost_hunter.models.insights import ScanInsight
from src.ghost_hunter.models.strategy import ScanStrategy
from src.ghost_hunter.models.enums import (
    DiscoverySource,
    EndpointCategory,
    ParamLocation,
    RiskLevel,
    VulnPattern,
)
from src.ghost_hunter.models.scan import AgentResult, ScanState

__all__ = [
    "AgentResult",
    "AttackSurfaceEntry",
    "DiscoverySource",
    "Endpoint",
    "EndpointCategory",
    "Finding",
    "ParamLocation",
    "ParameterDetail",
    "RiskLevel",
    "ScanInsight",
    "ScanState",
    "ScanStrategy",
    "SecuritySchemeInfo",
    "TechFingerprint",
    "VulnIndicator",
    "VulnPattern",
]
