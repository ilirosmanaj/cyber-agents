"""Agent registry — maps agent names to classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ghost_hunter.agents.base import BaseAgent

_AGENT_CLASSES: dict[str, type[BaseAgent]] = {}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Class decorator that registers an agent by its `name` attribute."""
    _AGENT_CLASSES[cls.name] = cls
    return cls


def get_agent_registry() -> dict[str, type[BaseAgent]]:
    """Return a copy of the agent registry."""
    return dict(_AGENT_CLASSES)
