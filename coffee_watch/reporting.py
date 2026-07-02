from __future__ import annotations

import functools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .models import RoasterSource
from .url_utils import safe_slug


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temp file, then atomically replace ``path``.

    Catalog and status JSON is read back by resume mode and the analysis
    skill; an interrupted run must not leave a torn, half-written file.
    """
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def log_products_json_snippet(
    logger: logging.Logger,
    roaster: RoasterSource,
    url: str,
    json_text: str,
    max_chars: int,
) -> None:
    if max_chars <= 0:
        return
    if len(json_text) > max_chars:
        snippet = json_text[:max_chars]
        logger.warning(
            "Products JSON snippet for %s (%s) [truncated %d chars]: %s",
            roaster.name,
            url,
            len(json_text) - max_chars,
            snippet,
        )
        return
    logger.warning(
        "Products JSON snippet for %s (%s): %s", roaster.name, url, json_text
    )


@functools.lru_cache(maxsize=64)
def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def report_file_path(
    reports_dir: Path,
    roaster_name: str,
    run_id: str,
    kind: str | None,
    ext: str,
) -> Path:
    _ensure_directory(reports_dir)
    slug = safe_slug(roaster_name)
    base = f"{run_id}-{slug}"
    filename = f"{base}.{ext}" if not kind else f"{base}.{kind}.{ext}"
    return reports_dir / filename


def make_roaster_catalog_path(
    reports_dir: Path, roaster_name: str, run_id: str
) -> Path:
    return report_file_path(reports_dir, roaster_name, run_id, "catalog", "json")


def combined_catalog_path(reports_dir: Path, run_id: str) -> Path:
    _ensure_directory(reports_dir)
    return reports_dir / f"{run_id}-catalog.json"


def new_products_catalog_path(reports_dir: Path, run_id: str) -> Path:
    _ensure_directory(reports_dir)
    return reports_dir / f"{run_id}-new-products.json"


def save_json(path: Path, payload: Any) -> Path:
    _ensure_directory(path.parent)
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    return path


def save_products_json(
    output_dir: Path,
    run_id: str,
    roaster: RoasterSource,
    page_index: int,
    json_text: str,
) -> Path:
    path = report_file_path(
        output_dir,
        roaster.name,
        run_id,
        f"products.raw.page{page_index}",
        "json",
    )
    write_text_atomic(path, json_text)
    return path


def save_products_json_pretty(
    output_dir: Path,
    run_id: str,
    roaster: RoasterSource,
    page_index: int,
    data: Any,
) -> Path | None:
    try:
        pretty = json.dumps(data, indent=2, ensure_ascii=True)
    except (TypeError, ValueError):
        return None
    path = report_file_path(
        output_dir,
        roaster.name,
        run_id,
        f"products.pretty.page{page_index}",
        "json",
    )
    write_text_atomic(path, pretty)
    return path


def prune_assets_dir(
    assets_dir: Path, retention_days: int, logger: logging.Logger
) -> int:
    """Delete raw payload files in ``assets_dir`` older than ``retention_days``.

    Assets are a per-run debugging cache, re-created on every scrape; without
    pruning the directory grows without bound. ``retention_days <= 0``
    disables pruning. Only regular files directly inside ``assets_dir`` are
    considered.
    """
    if retention_days <= 0 or not assets_dir.is_dir():
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for path in assets_dir.iterdir():
        try:
            if not path.is_file() or path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("Failed to prune asset %s: %s", path, exc)
    if removed:
        logger.info(
            "Pruned %d asset file(s) older than %d days from %s",
            removed,
            retention_days,
            assets_dir,
        )
    return removed


def today_roaster_catalog_paths(reports_dir: Path, run_id: str) -> list[Path]:
    return sorted(
        reports_dir.glob(f"{run_id}-*.catalog.json"),
        key=lambda path: path.name,
    )


def load_roaster_catalogs(
    catalog_paths: list[Path], logger: logging.Logger
) -> list[dict[str, Any]]:
    catalogs: list[dict[str, Any]] = []
    for path in catalog_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read roaster catalog %s: %s", path, exc)
            continue
        if isinstance(payload, dict):
            catalogs.append(payload)
        else:
            logger.warning("Skipping non-object roaster catalog %s", path)
    return catalogs
