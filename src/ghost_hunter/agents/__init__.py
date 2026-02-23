"""Agent registry — re-exports registry helpers and triggers agent registration."""

from src.ghost_hunter.agents.registry import get_agent_registry, register_agent

from src.ghost_hunter.agents.api_discovery import APIDiscoveryAgent  # noqa: F401
from src.ghost_hunter.agents.classifier import ClassifierAgent  # noqa: F401
from src.ghost_hunter.agents.hypothesis import HypothesisAgent  # noqa: F401
from src.ghost_hunter.agents.js_analyzer import JSAnalyzerAgent  # noqa: F401
from src.ghost_hunter.agents.passive_recon import PassiveReconAgent  # noqa: F401
from src.ghost_hunter.agents.prioritizer import PrioritizerAgent  # noqa: F401
from src.ghost_hunter.agents.vuln_analyzer import VulnPatternAnalyzer  # noqa: F401
from src.ghost_hunter.agents.verifier import VerifierAgent  # noqa: F401
from src.ghost_hunter.agents.web_crawler import WebCrawlerAgent  # noqa: F401

__all__ = [
    "get_agent_registry",
    "register_agent",
]
