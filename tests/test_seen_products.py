from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from coffee_watch.seen_products import SeenProducts


def test_async_seen_products_helpers_work(tmp_path: Path):
    seen = SeenProducts(tmp_path / "seen.db", logging.getLogger("test"))
    try:
        product_hash = seen.compute_hash(
            "https://example.com/products/x",
            "Coffee X",
            "Bright mango",
        )

        async def scenario() -> None:
            assert await seen.aget(product_hash) is None
            await seen.arecord(
                product_hash,
                "https://example.com/products/x",
                "Coffee X",
                "Bright mango",
                "2026-04-18T00:00:00+00:00",
                roaster="R",
                platform="shopify",
            )
            row = await seen.aget(product_hash)
            assert row is not None
            assert row.url == "https://example.com/products/x"
            assert (
                await seen.afirst_seen_for_url("https://example.com/products/x")
                == "2026-04-18T00:00:00+00:00"
            )

        asyncio.run(scenario())
    finally:
        seen.close()
