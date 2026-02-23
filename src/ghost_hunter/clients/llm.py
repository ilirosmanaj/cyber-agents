"""OpenAI-compatible LLM client with retry and Langfuse tracing."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TypeVar

from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from src.ghost_hunter.clients.tracing import trace_generation
from src.ghost_hunter.config import settings

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

MAX_CONCURRENT_LLM_CALLS = 5
BACKOFF_BASE = 2
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)


@dataclass
class LLMResponse:
    content: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient:
    """Async wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        resolved_key = api_key or settings.llm_api_key or settings.groq_api_key
        resolved_base_url, resolved_model = settings.resolve_llm_defaults()
        self.model = model or resolved_model

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=base_url or resolved_base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        name: str = "llm_call",
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Send a chat completion request with retry and tracing."""
        async with _llm_semaphore:
            return await self._chat_with_retry(
                messages=messages,
                name=name,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

    async def _chat_with_retry(
        self,
        messages: list[dict[str, str]],
        name: str,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict | None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }
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

                content = response.choices[0].message.content if response.choices else ""

                trace_generation(
                    name=name,
                    model=self.model,
                    input_data=messages,
                    output_data=content,
                    usage=usage,
                )

                return LLMResponse(content=content or "", usage=usage)

            except RateLimitError as e:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Rate limit hit (attempt %d/%d): %s — waiting %ds",
                    attempt + 1,
                    settings.max_retries,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(f"LLM API call failed after {settings.max_retries} retries")

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        name: str = "llm_call",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Chat completion that returns parsed JSON."""
        response = await self.chat(
            messages=messages,
            name=name,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(response.content)

    async def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        name: str = "llm_call",
        temperature: float | None = None,
        max_tokens: int | None = None,
        confidence_threshold: float | None = None,
        retry_context: str = "",
    ) -> T:
        """Chat completion that returns a validated Pydantic model.

        If ``confidence_threshold`` is set and the model has a ``confidence``
        field, a single retry is attempted when confidence falls below the
        threshold.  The retry prompt includes the first response as context.
        """
        data = await self.chat_json(
            messages=messages,
            name=name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = response_model.model_validate(data)

        if (
            confidence_threshold is not None
            and hasattr(result, "confidence")
            and result.confidence is not None
            and result.confidence < confidence_threshold
        ):
            logger.info(
                "Low confidence %.2f (threshold %.2f), retrying %s",
                result.confidence,
                confidence_threshold,
                name,
            )
            enhanced = self._build_retry_messages(
                original_messages=messages,
                first_result=result,
                retry_context=retry_context,
            )
            data = await self.chat_json(
                messages=enhanced,
                name=f"{name}_retry",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response_model.model_validate(data)

        return result

    @staticmethod
    def _build_retry_messages(
        original_messages: list[dict[str, str]],
        first_result: BaseModel,
        retry_context: str,
    ) -> list[dict[str, str]]:
        """Build enhanced messages for a confidence-gated retry."""
        enhanced = list(original_messages)
        first_json = first_result.model_dump_json(indent=2)
        retry_note = (
            "Your previous response had low confidence. "
            "Here it is for reference:\n\n"
            f"{first_json}\n\n"
            "Try again — be more specific and precise this time."
        )
        if retry_context:
            retry_note += f"\n\nAdditional context:\n{retry_context}"
        enhanced.append({"role": "assistant", "content": first_json})
        enhanced.append({"role": "user", "content": retry_note})
        return enhanced
