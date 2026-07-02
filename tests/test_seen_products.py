from __future__ import annotations

import logging
from pathlib import Path

from coffee_watch.seen_products import SeenProducts


def _row(
    seen: SeenProducts,
    url: str,
    title: str,
    description: str,
    first_seen_at: str,
) -> tuple:
    return (
        seen.compute_hash(url, title, description),
        url,
        title,
        description,
        first_seen_at,
        "",
        "R",
        "shopify",
    )


def test_record_many_and_bulk_lookups(tmp_path: Path):
    seen = SeenProducts(tmp_path / "seen.db", logging.getLogger("test"))
    try:
        url = "https://example.com/products/x"
        row = _row(seen, url, "Coffee X", "Bright mango", "2026-04-18T00:00:00+00:00")
        seen.record_many([row])

        assert seen.first_seen_for_urls([url]) == {url: "2026-04-18T00:00:00+00:00"}
        assert seen.first_seen_for_urls(["https://example.com/other"]) == {}
        assert seen.first_seen_for_hashes([row[0]]) == {
            row[0]: "2026-04-18T00:00:00+00:00"
        }
        assert seen.first_seen_for_hashes(["missing"]) == {}
    finally:
        seen.close()


def test_upsert_preserves_original_first_seen_at(tmp_path: Path):
    seen = SeenProducts(tmp_path / "seen.db", logging.getLogger("test"))
    try:
        url = "https://example.com/products/x"
        original = _row(seen, url, "Coffee X", "Bright mango", "2026-04-18T00:00:00+00:00")
        seen.record_many([original])
        # Same hash re-recorded on a later run must not move first_seen_at.
        rerecorded = (*original[:4], "2026-05-01T00:00:00+00:00", *original[5:])
        seen.record_many([rerecorded])

        assert seen.first_seen_for_urls([url]) == {url: "2026-04-18T00:00:00+00:00"}
    finally:
        seen.close()


def test_url_lookup_returns_earliest_across_description_edits(tmp_path: Path):
    seen = SeenProducts(tmp_path / "seen.db", logging.getLogger("test"))
    try:
        url = "https://example.com/products/x"
        seen.record_many(
            [
                _row(seen, url, "Coffee X", "original", "2026-04-18T00:00:00+00:00"),
                _row(seen, url, "Coffee X", "reworded", "2026-05-01T00:00:00+00:00"),
            ]
        )

        # A description edit creates a second hash row for the same URL; the
        # URL lookup must keep reporting the earliest sighting.
        assert seen.first_seen_for_urls([url]) == {url: "2026-04-18T00:00:00+00:00"}
    finally:
        seen.close()
