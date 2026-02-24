"""Adaptive rate-limited async HTTP client."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urljoin, urlparse

import httpx
from aiolimiter import AsyncLimiter

from src.ghost_hunter.config import settings

logger = logging.getLogger(__name__)

MIN_RATE_LIMIT = 0.5
SUCCESSES_BEFORE_RATE_RESTORE = 20
RATE_RESTORE_FACTOR = 1.25
BACKOFF_BASE = 2


class AdaptiveHttpClient:
    """Async HTTP client with adaptive rate limiting and retry logic."""

    def __init__(
        self,
        base_url: str,
        rate_limit: float | None = None,
        proxy: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._rate = rate_limit or settings.default_rate_limit
        self._limiter = AsyncLimiter(max_rate=max(1, int(self._rate)), time_period=1)
        self._consecutive_ok = 0
        self._original_rate = self._rate
        self._proxy = proxy or settings.proxy

        transport = httpx.AsyncHTTPTransport(retries=0, proxy=self._proxy)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            follow_redirects=True,
            transport=transport,
            headers={
                "User-Agent": "GhostHunter/0.1 (security-research)",
                "Accept": "text/html,application/json,application/xml,*/*",
            },
        )

    def resolve_url(self, path: str) -> str:
        """Resolve a path against the base URL."""
        if path.startswith(("http://", "https://")):
            return path
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def is_same_origin(self, url: str) -> bool:
        """Check if URL belongs to the same origin as base_url."""
        base_parsed = urlparse(self.base_url)
        url_parsed = urlparse(url)
        return url_parsed.netloc == base_parsed.netloc or not url_parsed.netloc

    async def request(
        self,
        method: str,
        url: str,
        *,
        follow_redirects: bool = True,
        **kwargs,
    ) -> httpx.Response | None:
        """Make an HTTP request with rate limiting, retry, and backoff."""
        full_url = self.resolve_url(url)

        for attempt in range(settings.max_retries):
            await self._limiter.acquire()

            try:
                resp = await self._client.request(
                    method,
                    full_url,
                    follow_redirects=follow_redirects,
                    **kwargs,
                )

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "2"))
                    self._rate = max(MIN_RATE_LIMIT, self._rate / 2)
                    self._limiter = AsyncLimiter(
                        max_rate=max(1, int(self._rate)), time_period=1
                    )
                    self._consecutive_ok = 0
                    logger.warning(
                        "Rate limited on %s — backing off %.1fs, new rate %.1f req/s",
                        full_url,
                        retry_after,
                        self._rate,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if 200 <= resp.status_code < 500:
                    self._consecutive_ok += 1
                    if self._consecutive_ok >= SUCCESSES_BEFORE_RATE_RESTORE and self._rate < self._original_rate:
                        self._rate = min(self._original_rate, self._rate * RATE_RESTORE_FACTOR)
                        self._limiter = AsyncLimiter(
                            max_rate=max(1, int(self._rate)), time_period=1
                        )
                        self._consecutive_ok = 0

                return resp

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s — retrying in %ds",
                    full_url,
                    attempt + 1,
                    settings.max_retries,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)

        logger.error("All %d retries exhausted for %s", settings.max_retries, full_url)
        return None

    async def get(self, url: str, **kwargs) -> httpx.Response | None:
        return await self.request("GET", url, **kwargs)

    async def head(self, url: str, **kwargs) -> httpx.Response | None:
        return await self.request("HEAD", url, **kwargs)

    async def close(self) -> None:
        await self._client.aclose()
