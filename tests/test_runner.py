from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from coffee_watch import runner
from coffee_watch.config import Settings
from coffee_watch.models import RoasterRunStatus, RoasterSource
from coffee_watch.report_status import STATUS_SUCCESS
from coffee_watch.roaster_pipeline import RoasterScrapeResult


def _status(roaster: str, status: str) -> RoasterRunStatus:
    return RoasterRunStatus(
        roaster=roaster,
        run_id="20260418",
        status=status,
        attempts=1,
        catalog_path=f"20260418-{roaster.lower()}.catalog.json",
        products_found=1 if status == STATUS_SUCCESS else 0,
        new_products=0,
        fetched_pages=0,
        platform="shopify",
    )


def test_full_run_writes_catalog_with_roaster_failures(monkeypatch, tmp_path: Path):
    settings = replace(Settings.defaults(), reports_dir=tmp_path)
    ctx = SimpleNamespace(
        settings=settings,
        logger=logging.getLogger("test"),
        run_id="20260418",
    )

    async def fake_run_roasters(ctx_arg, roasters):
        return [
            RoasterScrapeResult(
                tmp_path / "20260418-success.catalog.json",
                {
                    "roaster": {"name": "Success Roaster"},
                    "products": [{"title": "Coffee", "is_new": True}],
                },
                _status("Success", STATUS_SUCCESS),
            )
        ], ["Failed Roaster"]

    monkeypatch.setattr(runner, "_run_roasters", fake_run_roasters)

    exit_code = asyncio.run(runner._run_full(ctx, [RoasterSource("R", "https://x")]))

    assert exit_code == 0
    combined = tmp_path / "20260418-catalog.json"
    new_products = tmp_path / "20260418-new-products.json"
    assert combined.exists()
    assert new_products.exists()
    assert "Failed Roaster" in combined.read_text(encoding="utf-8")
