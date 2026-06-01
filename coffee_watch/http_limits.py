"""Per-host concurrency limiter.

The original implementation used a single global ``asyncio.Semaphore`` which
serialized requests across *all* domains — polite to the union, rude to each
individual host. ``PerHostLimiter`` keeps per-netloc semaphores so multiple
roasters can run in parallel without hammering any single site.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from urllib.parse import urlsplit


class PerHostLimiter:
    def __init__(self, per_host: int, global_cap: Optional[int] = None) -> None:
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
    async def acquire(self, url: str) -> AsyncIterator[None]:
        key = self._key(url) or "_default"
        host_sem = self._semaphore_for(key)
        if self._global is None:
            async with host_sem:
                yield
            return
        async with self._global:
            async with host_sem:
                yield
