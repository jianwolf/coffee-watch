"""Roaster-level run status: sidecar JSON + markdown fallback.

Replaces the fragile `grep the generated markdown for failure lines` flow
used by resume mode with structured ``*.status.json`` sidecars. Falls back
to markdown grep for reports produced before sidecars existed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import RoasterRunStatus, RoasterSource
from .reporting import make_report_path, report_file_path
from .url_utils import safe_slug

ROASTER_REPORT_FAILURE_SUFFIX = "roaster report generation has failed"
LEGACY_EMPTY_REPORT_LINE = "Gemini returned no text for this roaster."
EMPTY_REPORT_LINE = "LLM returned no text for this roaster."

STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"
STATUS_EMPTY = "empty"
STATUS_SKIPPED = "skipped"


def build_roaster_failure_line(roaster_name: str) -> str:
    return f"{roaster_name} {ROASTER_REPORT_FAILURE_SUFFIX}"


def status_sidecar_path(reports_dir: Path, roaster_name: str, run_id: str) -> Path:
    return report_file_path(reports_dir, roaster_name, run_id, "status", "json")


def items_sidecar_path(reports_dir: Path, roaster_name: str, run_id: str) -> Path:
    return report_file_path(reports_dir, roaster_name, run_id, "items", "json")


def write_status_sidecar(
    reports_dir: Path,
    status: RoasterRunStatus,
    logger: logging.Logger,
) -> Optional[Path]:
    path = status_sidecar_path(reports_dir, status.roaster, status.run_id)
    try:
        path.write_text(
            json.dumps(status.to_dict(), indent=2, ensure_ascii=True),
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


def write_items_sidecar(
    reports_dir: Path,
    roaster_name: str,
    run_id: str,
    items: list[dict[str, Any]],
    logger: logging.Logger,
) -> Optional[Path]:
    path = items_sidecar_path(reports_dir, roaster_name, run_id)
    try:
        path.write_text(
            json.dumps({"roaster": roaster_name, "run_id": run_id, "items": items},
                       indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to write items sidecar %s: %s", path, exc)
        return None
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def digest_report_name(job_name: str) -> str:
    names = {
        "digest": "z-digest",
        "new-digest": "z-new-digest",
        "roaster-digest": "z-roaster-digest",
    }
    return names.get(job_name, f"z-{job_name}")


def digest_report_filenames(run_id: str) -> set[str]:
    current_names = {
        digest_report_name("digest"),
        digest_report_name("new-digest"),
        digest_report_name("roaster-digest"),
    }
    legacy_names = {"digest", "new-digest", "roaster-digest"}
    all_names = current_names | legacy_names
    return {f"{run_id}-{name}.md" for name in all_names}


def today_roaster_report_paths(reports_dir: Path, run_id: str) -> list[Path]:
    digest_files = digest_report_filenames(run_id)
    paths = [
        path
        for path in reports_dir.glob(f"{run_id}-*.md")
        if path.name not in digest_files
    ]
    return sorted(paths, key=lambda path: path.name)


def _report_indicates_roaster_failure_from_text(text: str) -> bool:
    if LEGACY_EMPTY_REPORT_LINE in text or EMPTY_REPORT_LINE in text:
        return True
    for raw_line in text.splitlines():
        if raw_line.strip().endswith(ROASTER_REPORT_FAILURE_SUFFIX):
            return True
    return False


def status_is_failure(status_dict: Optional[dict[str, Any]]) -> bool:
    if not status_dict:
        return False
    return status_dict.get("status") in {STATUS_FAILURE, STATUS_EMPTY}


def collect_resume_targets(
    roasters: list[RoasterSource],
    reports_dir: Path,
    run_id: str,
    logger: logging.Logger,
) -> list[RoasterSource]:
    """Decide which roasters need a retry this run.

    Prefers the structured status sidecar; falls back to inspecting the
    markdown report for legacy failure lines.
    """
    targets: list[RoasterSource] = []
    for roaster in roasters:
        report_path = make_report_path(reports_dir, roaster.name, run_id)
        status = read_status_sidecar(reports_dir, roaster.name, run_id)
        if status is not None:
            if not report_path.exists():
                logger.info(
                    "Resume mode: report missing for %s (%s) despite status sidecar; scheduling retry.",
                    roaster.name,
                    report_path,
                )
                targets.append(roaster)
                continue
            if status.get("status") in {STATUS_FAILURE, STATUS_EMPTY}:
                logger.info(
                    "Resume mode: sidecar marks %s as %s; scheduling retry.",
                    roaster.name,
                    status.get("status"),
                )
                targets.append(roaster)
            continue
        if not report_path.exists():
            logger.info(
                "Resume mode: report missing for %s (%s); scheduling retry.",
                roaster.name,
                report_path,
            )
            targets.append(roaster)
            continue
        try:
            text = report_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Resume mode: failed to read report for %s (%s): %s; scheduling retry.",
                roaster.name,
                report_path,
                exc,
            )
            targets.append(roaster)
            continue
        if _report_indicates_roaster_failure_from_text(text):
            logger.info(
                "Resume mode: failed report detected for %s (%s); scheduling retry.",
                roaster.name,
                report_path,
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
        report_path = make_report_path(reports_dir, roaster.name, run_id)
        if not report_path.exists():
            missing.append(roaster.name)
    return missing


def extract_failed_roasters_from_reports(
    reports: list[tuple[str, str]],
    reports_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> list[str]:
    """Return roaster names whose report failed.

    Uses the status sidecar when available (keyed by report filename), else
    falls back to text-pattern matching in the markdown itself.
    """
    failed: list[str] = []
    seen: set[str] = set()

    def _append(name: str) -> None:
        cleaned = name.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            failed.append(cleaned)

    for report_name, text in reports:
        roaster_name = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("Roaster: "):
                roaster_name = line.split("Roaster: ", 1)[1].strip()
                break

        sidecar = None
        if reports_dir is not None and run_id and roaster_name:
            sidecar = read_status_sidecar(reports_dir, roaster_name, run_id)
        if sidecar is not None:
            if status_is_failure(sidecar):
                _append(roaster_name or report_name)
            continue

        current_roaster = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("Roaster: "):
                current_roaster = line.split("Roaster: ", 1)[1].strip()
                continue
            if line in {LEGACY_EMPTY_REPORT_LINE, EMPTY_REPORT_LINE}:
                _append(current_roaster)
                continue
            if line.endswith(ROASTER_REPORT_FAILURE_SUFFIX):
                candidate = line[: -len(ROASTER_REPORT_FAILURE_SUFFIX)].strip()
                _append(candidate or current_roaster)
    return failed


def append_failed_roaster_lines_to_digest(
    digest_text: str,
    failed_roasters: list[str],
) -> str:
    if not failed_roasters:
        return digest_text
    failure_lines = [build_roaster_failure_line(name) for name in failed_roasters]
    footer = "## Report Generation Failures\n\n" + "\n".join(failure_lines)
    base = digest_text.strip()
    if base:
        return f"{base}\n\n{footer}\n"
    return f"{footer}\n"


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


def _safe_slug(name: str) -> str:
    # Local alias so the report_status module doesn't need to reach into
    # url_utils directly from callers.
    return safe_slug(name)
