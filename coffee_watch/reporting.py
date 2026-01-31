from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import RoasterSource
from .url_utils import safe_slug


def log_products_json(
    logger: logging.Logger,
    roaster: RoasterSource,
    url: str,
    json_text: str,
    max_chars: int,
) -> None:
    if max_chars > 0 and len(json_text) > max_chars:
        snippet = json_text[:max_chars]
        logger.info(
            "Products JSON for %s (%s) [truncated %d chars]: %s",
            roaster.name,
            url,
            len(json_text) - max_chars,
            snippet,
        )
        return
    logger.info("Products JSON for %s (%s): %s", roaster.name, url, json_text)


def report_file_path(
    reports_dir: Path,
    roaster_name: str,
    run_id: str,
    kind: Optional[str],
    ext: str,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_slug(roaster_name)
    base = f"{run_id}-{slug}"
    filename = f"{base}.{ext}" if not kind else f"{base}.{kind}.{ext}"
    return reports_dir / filename

def save_products_json(
    report_dir: Path,
    run_id: str,
    roaster: RoasterSource,
    page_index: int,
    json_text: str,
) -> Path:
    path = report_file_path(
        report_dir,
        roaster.name,
        run_id,
        f"products.raw.page{page_index}",
        "json",
    )
    path.write_text(json_text, encoding="utf-8")
    return path


def save_products_json_pretty(
    report_dir: Path,
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
        report_dir,
        roaster.name,
        run_id,
        f"products.pretty.page{page_index}",
        "json",
    )
    path.write_text(pretty, encoding="utf-8")
    return path


def make_report_path(reports_dir: Path, roaster_name: str, run_id: str) -> Path:
    return report_file_path(reports_dir, roaster_name, run_id, None, "md")


def save_prompt_text(
    reports_dir: Path, run_id: str, roaster_name: str, prompt: str
) -> Path:
    path = report_file_path(reports_dir, roaster_name, run_id, "prompt", "txt")
    path.write_text(prompt, encoding="utf-8")
    return path


def init_report(path: Path, roaster_name: str, run_id: str) -> None:
    header = (
        "# Coffee Watch Report\n\n"
        f"Roaster: {roaster_name}\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
        f"Run: {run_id}\n\n"
        "## Recommendations\n\n"
    )
    path.write_text(header, encoding="utf-8")


def select_reports_for_digest(report_paths: list[Path], limit: int) -> list[Path]:
    if len(report_paths) <= limit:
        return report_paths
    return report_paths[:limit]


def load_reports_for_digest(
    report_paths: list[Path], logger: logging.Logger
) -> list[tuple[str, str]]:
    reports: list[tuple[str, str]] = []
    for path in report_paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Failed to read report %s: %s", path, exc)
            continue
        if text:
            reports.append((path.name, text))
    return reports
