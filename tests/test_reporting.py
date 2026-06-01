from __future__ import annotations

import json
import logging
from pathlib import Path

from coffee_watch.reporting import load_roaster_catalogs, save_json


def test_save_json_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "nested" / "catalog.json"

    saved = save_json(path, {"ok": True})

    assert saved == path
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_load_roaster_catalogs_skips_non_object_payloads(tmp_path: Path, caplog):
    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        catalogs = load_roaster_catalogs([catalog], logging.getLogger("test"))

    assert catalogs == []
    assert "Skipping non-object roaster catalog" in caplog.text
