"""Scan strategy model — planner agent output that guides downstream agents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScanStrategy(BaseModel):
    focus_areas: list[str] = Field(default_factory=list)  # e.g., ["graphql", "auth_flows"]
    skip_agents: list[str] = Field(default_factory=list)  # agents to skip entirely
    extra_paths_to_try: list[str] = Field(default_factory=list)  # additional paths for api_discovery
    tech_hypotheses: list[str] = Field(default_factory=list)  # e.g., ["Spring Boot", "JWT auth"]
    scan_depth: str = "normal"  # "shallow" | "normal" | "deep"
    priority_patterns: list[str] = Field(default_factory=list)  # endpoint patterns to prioritize
