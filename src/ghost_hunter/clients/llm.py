"""Groq LLM wrapper with retry logic and Langfuse tracing."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from groq import AsyncGroq, RateLimitError

from src.ghost_hunter.clients.tracing import trace_generation
from src.ghost_hunter.config import settings

logger = logging.getLogger(__name__)

MAX_CONCURRENT_LLM_CALLS = 5
BACKOFF_BASE = 2
_groq_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)


class LLMClient:
    """Async Groq client with retry, rate-limit handling, and tracing."""

    def __init__(self):
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        name: str = "llm_call",
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> Any:
        """Send a chat completion request with retry and tracing."""
        async with _groq_semaphore:
            return await self._chat_with_retry(
                messages=messages,
                name=name,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

    async def _chat_with_retry(
        self,
        messages: list[dict[str, str]],
        name: str,
        tools: list[dict] | None,
        tool_choice: str | dict | None,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.groq_temperature,
            "max_tokens": max_tokens or settings.groq_max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format

        for attempt in range(settings.max_retries):
            try:
                response = await self._client.chat.completions.create(**kwargs)

                usage = {}
                if response.usage:
                    usage = {
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total": response.usage.total_tokens,
                    }
                trace_generation(
                    name=name,
                    model=self.model,
                    input_data=messages,
                    output_data=(
                        response.choices[0].message.content
                        if response.choices
                        else None
                    ),
                    usage=usage,
                )

                return response

            except RateLimitError as e:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Groq rate limit hit (attempt %d/%d): %s — waiting %ds",
                    attempt + 1,
                    settings.max_retries,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(f"Groq API call failed after {settings.max_retries} retries")

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        name: str = "llm_call",
        temperature: float | None = None,
    ) -> dict:
        """Chat completion that returns parsed JSON."""
        response = await self.chat(
            messages=messages,
            name=name,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)

