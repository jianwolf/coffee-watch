from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any, Optional

from .models import RoasterSource
from .url_utils import safe_slug


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
    kind: Optional[str],
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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
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
    path.write_text(json_text, encoding="utf-8")
    return path


def save_products_json_pretty(
    output_dir: Path,
    run_id: str,
    roaster: RoasterSource,
    page_index: int,
    data: Any,
) -> Optional[Path]:
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
    path.write_text(pretty, encoding="utf-8")
    return path


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
    return catalogs
