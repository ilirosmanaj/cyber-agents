"""Scan insight model — accumulated learning across phases."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScanInsight(BaseModel):
    phase: str = ""  # "recon", "crawl", "discovery", "classification", "analysis"
    summary: str = ""  # 2-3 sentence LLM summary
    key_signals: list[str] = Field(default_factory=list)
    recommended_focus: list[str] = Field(default_factory=list)
