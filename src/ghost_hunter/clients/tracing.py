"""Langfuse tracing initialization and helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from langfuse import Langfuse
from opentelemetry import trace as otel_trace_api

from src.ghost_hunter.config import settings

logger = logging.getLogger(__name__)

_langfuse: Langfuse | None = None
_trace_id: str | None = None
_trace_name: str | None = None
_trace_session_id: str | None = None
_trace_metadata: dict | None = None


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
    """Create a trace ID and store name/session for the first span to register."""
    global _trace_id, _trace_name, _trace_session_id, _trace_metadata
    if _langfuse is None:
        return None
    _trace_id = Langfuse.create_trace_id(seed=session_id)
    _trace_name = name
    _trace_session_id = session_id
    _trace_metadata = metadata
    return _trace_id


def _register_trace_if_needed() -> None:
    """Set the trace name/session on the first span, then clear the pending state."""
    global _trace_name, _trace_session_id, _trace_metadata
    if _langfuse is None or _trace_name is None:
        return
    _langfuse.update_current_trace(
        name=_trace_name,
        session_id=_trace_session_id,
        metadata=_trace_metadata,
    )
    _trace_name = None
    _trace_session_id = None
    _trace_metadata = None


@contextmanager
def trace_span(name: str, metadata: dict | None = None) -> Generator:
    """Context manager for creating a traced span."""
    if _langfuse is None or _trace_id is None:
        yield None
        return

    # only pass trace_context for root spans; nested spans inherit from OTel context
    kwargs: dict[str, Any] = {"name": name, "metadata": metadata or {}}
    if not otel_trace_api.get_current_span().is_recording():
        kwargs["trace_context"] = {"trace_id": _trace_id}

    with _langfuse.start_as_current_span(**kwargs) as span:
        _register_trace_if_needed()
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

    # nest under current span if inside one, otherwise link to trace root
    kwargs: dict[str, Any] = {
        "name": name,
        "as_type": "generation",
        "model": model,
        "input": input_data,
        "output": output_data,
        "usage_details": usage_details,
        "metadata": metadata or {},
    }
    if not otel_trace_api.get_current_span().is_recording():
        kwargs["trace_context"] = {"trace_id": _trace_id}

    gen = _langfuse.start_observation(**kwargs)
    gen.end()


def flush_langfuse() -> None:
    """Flush any pending Langfuse events."""
    if _langfuse is not None:
        try:
            _langfuse.flush()
        except Exception as e:
            logger.warning("Failed to flush Langfuse: %s", e)
