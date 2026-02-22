"""Langfuse tracing initialization and helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from langfuse import Langfuse

from src.ghost_hunter.config import settings

logger = logging.getLogger(__name__)

_langfuse: Langfuse | None = None
_trace_id: str | None = None


def init_langfuse() -> Langfuse | None:
    """Initialize Langfuse client if keys are configured."""
    global _langfuse
    if not settings.langfuse_enabled:
        logger.info("Langfuse not configured — tracing disabled")
        return None
    try:
        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing initialized")
        return _langfuse
    except Exception as e:
        logger.warning("Failed to initialize Langfuse: %s", e)
        return None


def create_trace(name: str, session_id: str, metadata: dict | None = None) -> str | None:
    """Create a trace ID for a scan run and store it for child spans."""
    global _trace_id
    if _langfuse is None:
        return None
    _trace_id = Langfuse.create_trace_id(seed=session_id)
    return _trace_id


@contextmanager
def trace_span(name: str, metadata: dict | None = None) -> Generator:
    """Context manager for creating a traced span."""
    if _langfuse is None or _trace_id is None:
        yield None
        return

    trace_context = {"trace_id": _trace_id}
    with _langfuse.start_as_current_span(
        name=name,
        metadata=metadata or {},
        trace_context=trace_context,
    ) as span:
        try:
            yield span
        except Exception as e:
            span.update(metadata={"error": str(e)})
            raise


def trace_generation(
    name: str,
    model: str,
    input_data: Any,
    output_data: Any,
    usage: dict | None = None,
    metadata: dict | None = None,
) -> None:
    """Record an LLM generation in the current trace."""
    if _langfuse is None or _trace_id is None:
        return

    usage_details = {}
    if usage:
        usage_details = {
            "input_tokens": usage.get("input", 0),
            "output_tokens": usage.get("output", 0),
            "total_tokens": usage.get("total", 0),
        }

    trace_context = {"trace_id": _trace_id}
    gen = _langfuse.start_generation(
        name=name,
        model=model,
        input=input_data,
        output=output_data,
        usage_details=usage_details,
        metadata=metadata or {},
        trace_context=trace_context,
    )
    gen.end()


def flush_langfuse() -> None:
    """Flush any pending Langfuse events."""
    if _langfuse is not None:
        try:
            _langfuse.flush()
        except Exception as e:
            logger.warning("Failed to flush Langfuse: %s", e)
