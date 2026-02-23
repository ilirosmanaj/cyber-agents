"""Configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings

_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "openai": ("https://api.openai.com/v1", "gpt-4o"),
    "together": ("https://api.together.xyz/v1", "meta-llama/Llama-3-70b-chat-hf"),
    "vllm": ("http://localhost:8000/v1", "default"),
    "ollama": ("http://localhost:11434/v1", "llama3"),
}


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # llm provider
    llm_provider: str = "groq"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""

    # llm defaults
    llm_temperature: float = 0.0
    llm_max_tokens: int = 16384

    # groq (backward compat — used as fallback if llm_api_key is empty)
    groq_api_key: str = ""

    # langfuse (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # http
    default_rate_limit: float = 5.0  # requests per second
    request_timeout: float = 15.0
    max_retries: int = 3
    proxy: str | None = None

    # crawler
    max_crawl_depth: int = 5
    max_pages: int = 200
    max_js_files: int = 50

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def resolve_llm_defaults(self) -> tuple[str, str]:
        """Return (base_url, model) using explicit settings or provider defaults."""
        if self.llm_provider not in _PROVIDER_DEFAULTS:
            supported = ", ".join(sorted(_PROVIDER_DEFAULTS))
            raise ValueError(
                f"Unknown LLM provider '{self.llm_provider}'. "
                f"Supported: {supported}"
            )
        default_url, default_model = _PROVIDER_DEFAULTS[self.llm_provider]
        base_url = self.llm_base_url or default_url
        model = self.llm_model or default_model
        return base_url, model


settings = Settings()
