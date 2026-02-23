"""Prompt registry — loads and caches YAML prompt files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptEntry:
    name: str
    version: str
    system_prompt: str
    description: str


class PromptRegistry:
    """Loads and caches YAML prompt files from the prompts/ directory."""

    def __init__(self, prompts_dir: Path | None = None):
        self._dir = prompts_dir or Path(__file__).parent / "prompts"
        self._cache: dict[str, PromptEntry] = {}

    def get(self, name: str) -> PromptEntry:
        """Return the cached prompt entry, loading from disk on first access."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _load(self, name: str) -> PromptEntry:
        """Load a single YAML prompt file and return a PromptEntry."""
        path = self._dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return PromptEntry(
            name=data.get("name", name),
            version=str(data.get("version", "1.0")),
            system_prompt=data.get("system_prompt", ""),
            description=data.get("description", ""),
        )
