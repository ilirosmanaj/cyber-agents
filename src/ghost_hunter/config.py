"""Configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 4096
    groq_temperature: float = 0.2

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
    max_crawl_depth: int = 3
    max_pages: int = 100

    # orchestrator
    max_orchestrator_iterations: int = 15

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
