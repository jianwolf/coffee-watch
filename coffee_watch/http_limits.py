"""Per-host concurrency limiter.

The original implementation used a single global ``asyncio.Semaphore`` which
serialized requests across *all* domains — polite to the union, rude to each
individual host. ``PerHostLimiter`` keeps per-netloc semaphores so multiple
roasters can run in parallel without hammering any single site.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit


class PerHostLimiter:
    def __init__(self, per_host: int, global_cap: int | None = None) -> None:
        if per_host < 1:
            raise ValueError("per_host must be >= 1")
        self._per_host = per_host
        self._global = asyncio.Semaphore(global_cap) if global_cap and global_cap > 0 else None
        self._locks: dict[str, asyncio.Semaphore] = {}

    def _key(self, url: str) -> str:
        return urlsplit(url).netloc.lower()

    def _semaphore_for(self, key: str) -> asyncio.Semaphore:
        sem = self._locks.get(key)
        if sem is None:
            # setdefault closes the TOCTOU window: if two coroutines both saw
            # the key as missing, the second one's freshly-built Semaphore is
            # discarded and the first instance is shared.
            sem = self._locks.setdefault(key, asyncio.Semaphore(self._per_host))
        return sem

    @asynccontextmanager
    async def host_slot(self, url: str) -> AsyncIterator[None]:
        """Hold only the per-host slot.

        Politeness sleeps belong inside this context: they keep same-host
        requests spaced out without tying up a global slot while sleeping.
        """
        key = self._key(url) or "_default"
        async with self._semaphore_for(key):
            yield

    @asynccontextmanager
    async def global_slot(self) -> AsyncIterator[None]:
        """Hold a global slot; should wrap only the request itself."""
        if self._global is None:
            yield
            return
        async with self._global:
            yield

    @asynccontextmanager
    async def acquire(self, url: str) -> AsyncIterator[None]:
        # Host first, then global: waiting for a busy host must not pin one of
        # the global slots.
        async with self.host_slot(url), self.global_slot():
            yield


class Host429Gate:
    """Fast-skip tracker for hosts that 429 every page request.

    Some Shopify storefronts classify the bot User-Agent and return 429 for
    every product HTML page while leaving the catalog API open (see
    docs/shopify-bot-gating.md). Retrying page after page against such a host
    wastes hundreds of requests per run. Callers record the status of each
    page-fetch attempt; once a host has answered ``threshold`` consecutive
    429s, ``is_gated`` turns true and remaining page fetches for that host
    should be skipped for the rest of the run.

    Only page fetches should record into the gate: catalog API and robots.txt
    requests are unaffected by the gating and must not reset the streak.
    """

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._threshold = threshold
        self._consecutive_429s: dict[str, int] = {}

    @staticmethod
    def _key(url: str) -> str:
        return urlsplit(url).netloc.lower()

    def is_gated(self, url: str) -> bool:
        return self._consecutive_429s.get(self._key(url), 0) >= self._threshold

    def record(self, url: str, status_code: int) -> None:
        key = self._key(url)
        if status_code == 429:
            self._consecutive_429s[key] = self._consecutive_429s.get(key, 0) + 1
        else:
            self._consecutive_429s.pop(key, None)
