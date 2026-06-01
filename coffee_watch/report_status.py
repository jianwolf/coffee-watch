"""Roaster-level scrape status sidecars."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import RoasterRunStatus, RoasterSource
from .reporting import make_roaster_catalog_path, report_file_path

STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"
STATUS_EMPTY = "empty"
STATUS_SKIPPED = "skipped"


def status_sidecar_path(reports_dir: Path, roaster_name: str, run_id: str) -> Path:
    return report_file_path(reports_dir, roaster_name, run_id, "status", "json")


def write_status_sidecar(
    reports_dir: Path,
    status: RoasterRunStatus,
    logger: logging.Logger,
) -> Optional[Path]:
    path = status_sidecar_path(reports_dir, status.roaster, status.run_id)
    try:
        path.write_text(
            json.dumps(status.to_dict(), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to write status sidecar %s: %s", path, exc)
        return None
    return path


def read_status_sidecar(
    reports_dir: Path,
    roaster_name: str,
    run_id: str,
) -> Optional[dict[str, Any]]:
    path = status_sidecar_path(reports_dir, roaster_name, run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def status_is_retryable(status_dict: Optional[dict[str, Any]]) -> bool:
    if not status_dict:
        return False
    return status_dict.get("status") in {STATUS_FAILURE, STATUS_EMPTY}


def collect_resume_targets(
    roasters: list[RoasterSource],
    reports_dir: Path,
    run_id: str,
    logger: logging.Logger,
) -> list[RoasterSource]:
    targets: list[RoasterSource] = []
    for roaster in roasters:
        catalog_path = make_roaster_catalog_path(reports_dir, roaster.name, run_id)
        status = read_status_sidecar(reports_dir, roaster.name, run_id)
        if status is not None:
            if not catalog_path.exists() and status.get("status") != STATUS_SKIPPED:
                logger.info(
                    "Resume mode: catalog missing for %s (%s) despite status "
                    "sidecar; scheduling retry.",
                    roaster.name,
                    catalog_path,
                )
                targets.append(roaster)
                continue
            if status_is_retryable(status):
                logger.info(
                    "Resume mode: sidecar marks %s as %s; scheduling retry.",
                    roaster.name,
                    status.get("status"),
                )
                targets.append(roaster)
            continue
        if not catalog_path.exists():
            logger.info(
                "Resume mode: catalog missing for %s (%s); scheduling retry.",
                roaster.name,
                catalog_path,
            )
            targets.append(roaster)
    return targets


def collect_missing_roaster_names(
    roasters: list[RoasterSource],
    reports_dir: Path,
    run_id: str,
) -> list[str]:
    missing: list[str] = []
    for roaster in roasters:
        catalog_path = make_roaster_catalog_path(reports_dir, roaster.name, run_id)
        if not catalog_path.exists():
            missing.append(roaster.name)
    return missing


def merge_failed_roaster_names(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in group:
            cleaned = name.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                merged.append(cleaned)
    return merged
