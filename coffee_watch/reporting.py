from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import RoasterSource
from .url_utils import safe_slug

UNTRUSTED_OPEN = "<UNTRUSTED_SCRAPED_TEXT>"
UNTRUSTED_CLOSE = "</UNTRUSTED_SCRAPED_TEXT>"


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


def make_report_path(reports_dir: Path, roaster_name: str, run_id: str) -> Path:
    return report_file_path(reports_dir, roaster_name, run_id, None, "md")


def save_prompt_text(
    output_dir: Path, run_id: str, roaster_name: str, prompt: str
) -> Path:
    path = report_file_path(output_dir, roaster_name, run_id, "prompt", "txt")
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


def _items_sidecar_path(reports_dir: Path, roaster_name: str, run_id: str) -> Path:
    return report_file_path(reports_dir, roaster_name, run_id, "items", "json")


def load_items_for_reports(
    reports: list[tuple[str, str]],
    reports_dir: Path,
    run_id: str,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Read structured items from the JSON sidecar when available; fall back
    to grepping the markdown ``## Coffee list`` block when it's not."""
    items: list[dict[str, Any]] = []
    for report_name, text in reports:
        roaster_name = ""
        for line in text.splitlines():
            if line.startswith("Roaster: "):
                roaster_name = line.split("Roaster: ", 1)[1].strip()
                break
        if roaster_name:
            sidecar = _items_sidecar_path(reports_dir, roaster_name, run_id)
            if sidecar.exists():
                try:
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Failed to read items sidecar %s: %s", sidecar, exc
                    )
                else:
                    sidecar_items = payload.get("items") or []
                    if isinstance(sidecar_items, list):
                        items.extend(
                            item for item in sidecar_items if isinstance(item, dict)
                        )
                        continue
        items.extend(_extract_coffee_list_items_from_markdown(report_name, text))
    if not items:
        logger.warning(
            "No coffee list items found while building digest-only new-digest."
        )
    return items


def _extract_coffee_list_items_from_markdown(
    report_name: str, text: str
) -> list[dict[str, Any]]:
    """Fallback legacy parser for reports generated before items sidecars."""
    from .parsing import product_id_from_url  # local import to avoid cycle

    items: list[dict[str, Any]] = []
    roaster_name = ""
    for line in text.splitlines():
        if line.startswith("Roaster: "):
            roaster_name = line.split("Roaster: ", 1)[1].strip()
            break
    if not roaster_name:
        roaster_name = report_name

    def finalize_description(item: Optional[dict[str, Any]], lines: list[str]) -> None:
        if item is None:
            return
        cleaned = "\n".join(lines).strip()
        if cleaned:
            cleaned = cleaned.replace(UNTRUSTED_OPEN, "").replace(UNTRUSTED_CLOSE, "")
        item["description"] = cleaned.strip()

    def finalize_item(item: Optional[dict[str, Any]]) -> None:
        if not item:
            return
        if not item.get("product_id") and item.get("url"):
            item["product_id"] = product_id_from_url(str(item["url"]))
        if "variant_lines" not in item:
            item["variant_lines"] = []
        items.append(item)

    in_list = False
    current: Optional[dict[str, Any]] = None
    in_variants = False
    expecting_description = False
    description_lines: list[str] = []
    for line in text.splitlines():
        if line.strip() == "## Coffee list":
            in_list = True
            continue
        if not in_list:
            continue
        if line.startswith("## "):
            if expecting_description:
                finalize_description(current, description_lines)
                expecting_description = False
                description_lines = []
            break
        if line.startswith("- "):
            if expecting_description:
                finalize_description(current, description_lines)
                expecting_description = False
                description_lines = []
            finalize_item(current)
            current = None
            in_variants = False
            entry = line[2:].strip()
            if not entry:
                continue
            if entry.startswith("product_id:"):
                product_id = entry.split(":", 1)[1].strip()
                current = {
                    "roaster": roaster_name,
                    "product_id": product_id,
                    "name": "",
                    "url": "",
                    "list_price": "",
                    "badge": "",
                    "description": "",
                    "variant_lines": [],
                }
                continue
            name = entry
            url = ""
            if entry.endswith(")") and " (" in entry:
                maybe_name, maybe_url = entry.rsplit(" (", 1)
                if maybe_url.startswith("http"):
                    name = maybe_name.strip()
                    url = maybe_url[:-1]
            finalize_item(
                {
                    "roaster": roaster_name,
                    "product_id": "",
                    "name": name,
                    "url": url,
                    "list_price": "",
                    "badge": "",
                    "description": "",
                    "variant_lines": [],
                }
            )
            continue
        if current is None:
            continue
        stripped = line.strip()
        if not stripped:
            if expecting_description:
                finalize_description(current, description_lines)
                expecting_description = False
                description_lines = []
            finalize_item(current)
            current = None
            in_variants = False
            continue
        if stripped.startswith("name:"):
            current["name"] = stripped.split(":", 1)[1].strip()
            in_variants = False
            expecting_description = False
            description_lines = []
            continue
        if stripped.startswith("url:"):
            current["url"] = stripped.split(":", 1)[1].strip()
            in_variants = False
            expecting_description = False
            description_lines = []
            continue
        if stripped.startswith("list price:"):
            current["list_price"] = stripped.split(":", 1)[1].strip()
            in_variants = False
            expecting_description = False
            description_lines = []
            continue
        if stripped.startswith("badge:"):
            current["badge"] = stripped.split(":", 1)[1].strip()
            in_variants = False
            expecting_description = False
            description_lines = []
            continue
        if stripped == "variants:":
            in_variants = True
            expecting_description = False
            description_lines = []
            variant_lines = current.get("variant_lines")
            if isinstance(variant_lines, list):
                variant_lines.append("  variants:")
            continue
        if in_variants and stripped.startswith("- "):
            variant = stripped[2:].strip()
            if variant:
                variant_lines = current.get("variant_lines")
                if isinstance(variant_lines, list):
                    variant_lines.append(f"    - {variant}")
            continue
        if stripped.startswith("description:"):
            in_variants = False
            desc = stripped.split("description:", 1)[1].strip()
            description_lines = [desc] if desc else []
            expecting_description = True
            continue
        if expecting_description:
            description_lines.append(stripped)
    if expecting_description:
        finalize_description(current, description_lines)
    finalize_item(current)
    return items


# Backwards-compatible alias; prefer ``load_items_for_reports``.
def extract_coffee_list_items(
    reports: list[tuple[str, str]], logger: logging.Logger
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for report_name, text in reports:
        items.extend(_extract_coffee_list_items_from_markdown(report_name, text))
    if not items:
        logger.warning(
            "No coffee list items found while building digest-only new-digest."
        )
    return items
