from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from coffee_watch.reporting import (
    load_roaster_catalogs,
    prune_assets_dir,
    save_json,
)


def test_save_json_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "nested" / "catalog.json"

    saved = save_json(path, {"ok": True})

    assert saved == path
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_save_json_leaves_no_temp_file_behind(tmp_path: Path):
    path = tmp_path / "catalog.json"

    save_json(path, {"ok": True})

    assert [p.name for p in tmp_path.iterdir()] == ["catalog.json"]


def test_load_roaster_catalogs_skips_non_object_payloads(tmp_path: Path, caplog):
    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        catalogs = load_roaster_catalogs([catalog], logging.getLogger("test"))

    assert catalogs == []
    assert "Skipping non-object roaster catalog" in caplog.text


def test_prune_assets_dir_removes_only_stale_files(tmp_path: Path):
    stale = tmp_path / "20260101-old.products.raw.page1.json"
    fresh = tmp_path / "20260630-new.products.raw.page1.json"
    stale.write_text("{}", encoding="utf-8")
    fresh.write_text("{}", encoding="utf-8")
    forty_days_ago = time.time() - 40 * 86400
    os.utime(stale, (forty_days_ago, forty_days_ago))

    removed = prune_assets_dir(tmp_path, 30, logging.getLogger("test"))

    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()


def test_prune_assets_dir_disabled_with_zero_retention(tmp_path: Path):
    stale = tmp_path / "old.json"
    stale.write_text("{}", encoding="utf-8")
    forty_days_ago = time.time() - 40 * 86400
    os.utime(stale, (forty_days_ago, forty_days_ago))

    assert prune_assets_dir(tmp_path, 0, logging.getLogger("test")) == 0
    assert stale.exists()


def test_prune_assets_dir_handles_missing_directory(tmp_path: Path):
    assert prune_assets_dir(tmp_path / "absent", 30, logging.getLogger("test")) == 0
