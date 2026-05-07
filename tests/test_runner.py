from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from coffee_watch import runner
from coffee_watch.config import Settings
from coffee_watch.models import RoasterSource


def test_full_run_passes_roaster_task_failures_to_digest(monkeypatch, tmp_path: Path):
    settings = replace(Settings.defaults(), reports_dir=tmp_path)
    ctx = SimpleNamespace(
        settings=settings,
        logger=logging.getLogger("test"),
        run_id="20260418",
    )
    captured: dict[str, object] = {}

    async def fake_run_roasters(ctx_arg, roasters):
        return [tmp_path / "20260418-success.md"], ["No Report Roaster"]

    def fake_load_reports(paths, logger):
        return [("20260418-success.md", "Roaster: Success Roaster\n")]

    def fake_extract_failed(reports, reports_dir, run_id):
        return ["Failed Markdown Roaster"]

    async def fake_finalize_digests(ctx_arg, reports, failed_roasters):
        captured["reports"] = reports
        captured["failed_roasters"] = failed_roasters

    monkeypatch.setattr(runner, "_run_roasters", fake_run_roasters)
    monkeypatch.setattr(runner, "load_reports_for_digest", fake_load_reports)
    monkeypatch.setattr(
        runner, "extract_failed_roasters_from_reports", fake_extract_failed
    )
    monkeypatch.setattr(runner, "_finalize_digests", fake_finalize_digests)

    exit_code = asyncio.run(runner._run_full(ctx, [RoasterSource("R", "https://x")]))

    assert exit_code == 0
    assert captured["failed_roasters"] == [
        "No Report Roaster",
        "Failed Markdown Roaster",
    ]
