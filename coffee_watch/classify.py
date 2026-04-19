"""New-product classification and date resolution.

Extracted from ``runner.py`` so the classification logic is testable in
isolation (no asyncio, no httpx, no LLM).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

from .config import Settings
from .models import ProductCandidate
from .seen_products import SeenProducts

NEW_PRODUCTS_WINDOW_DAYS = 7

DATE_SOURCES = (
    "shopify_published_at",
    "http_last_modified",
    "wix_lastmod",
    "seen_at",
)


def _parse_iso_date(value: str) -> Optional[date]:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date()


def _parse_http_date(value: str) -> Optional[date]:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date()


def _parse_wix_lastmod(value: str) -> Optional[date]:
    return _parse_http_date(value) or _parse_iso_date(value)


def resolve_update_date(
    product: ProductCandidate,
    http_last_modified: str,
    wix_lastmod: str,
    seen_at: str,
) -> tuple[Optional[date], str]:
    published_date = _parse_iso_date(product.shopify_published_at)
    if published_date:
        return published_date, "shopify_published_at"
    http_date = _parse_http_date(http_last_modified)
    if http_date:
        return http_date, "http_last_modified"
    wix_date = _parse_wix_lastmod(wix_lastmod)
    if wix_date:
        return wix_date, "wix_lastmod"
    seen_date = _parse_iso_date(seen_at)
    if seen_date:
        return seen_date, "seen_at"
    return None, "unknown"


def classify_new_products(
    products: list[ProductCandidate],
    run_day: date,
    seen_products: SeenProducts,
    descriptions_by_url: dict[str, str],
    http_last_modified_by_url: dict[str, str],
    wix_lastmod_by_url: dict[str, str],
    platform: str,
    window_days: int = NEW_PRODUCTS_WINDOW_DAYS,
    persist_seen: bool = True,
) -> tuple[set[str], dict[str, int], int, int]:
    new_urls: set[str] = set()
    by_source: dict[str, int] = {source: 0 for source in DATE_SOURCES}
    undated = 0
    outside_window = 0
    now_iso = datetime.now(timezone.utc).isoformat() if persist_seen else ""
    days = max(1, window_days)
    window_start = run_day - timedelta(days=days - 1)
    for product in products:
        description = descriptions_by_url.get(product.url, "")
        # Prefer URL-based lookup: description edits should not reset seen_at.
        seen_at = seen_products.first_seen_for_url(product.url)
        if not seen_at:
            seen_hash = seen_products.compute_hash(product.url, product.name, description)
            seen_entry = seen_products.get(seen_hash)
            seen_at = seen_entry.first_seen_at if seen_entry else ""
        http_last_modified = http_last_modified_by_url.get(product.url, "")
        wix_lastmod = wix_lastmod_by_url.get(product.url, "")

        updated_date, source = resolve_update_date(
            product, http_last_modified, wix_lastmod, seen_at
        )
        if updated_date is None:
            undated += 1
            if persist_seen:
                source = "undated"
                updated_date = run_day
            else:
                outside_window += 1
                continue

        if persist_seen:
            seen_hash = seen_products.compute_hash(product.url, product.name, description)
            seen_products.record(
                seen_hash,
                product.url,
                product.name,
                description,
                now_iso,
                shopify_updated_at=product.shopify_updated_at,
                roaster=product.source,
                platform=platform,
            )

        if updated_date < window_start or updated_date > run_day:
            outside_window += 1
            continue
        new_urls.add(product.url)
        if source in by_source:
            by_source[source] += 1
    return new_urls, by_source, undated, outside_window


def build_filtered_new_items_for_digest(
    reports: list[tuple[str, str]],
    run_id: str,
    settings: Settings,
    logger: logging.Logger,
    mode_label: str,
    items_loader,
    *,
    window_days: int = NEW_PRODUCTS_WINDOW_DAYS,
    persist_seen: bool = False,
) -> list[dict[str, Any]]:
    """Re-run classification over previously captured coffee-list items.

    ``items_loader(reports)`` returns the structured coffee-list items
    (preferably from JSON sidecars; falls back to markdown-grep).
    """
    new_items = items_loader(reports)
    if not new_items:
        return new_items

    seen_products = SeenProducts(settings.seen_db_path, logger)
    try:
        digest_products: list[ProductCandidate] = []
        descriptions_by_url: dict[str, str] = {}
        http_last_modified_by_url: dict[str, str] = {}
        wix_lastmod_by_url: dict[str, str] = {}
        for item in new_items:
            url = str(item.get("url", "") or "").strip()
            product = ProductCandidate(
                product_id=str(item.get("product_id", "") or ""),
                name=str(item.get("name", "") or ""),
                url=url,
                source=str(item.get("roaster", "") or ""),
                list_price=str(item.get("list_price", "") or ""),
                list_badge=str(item.get("badge", "") or ""),
                body_html="",
                variants=(),
                shopify_updated_at=str(item.get("shopify_updated_at", "") or ""),
                shopify_published_at=str(item.get("shopify_published_at", "") or ""),
            )
            digest_products.append(product)
            descriptions_by_url[url] = str(item.get("description", "") or "")
        run_day = datetime.strptime(run_id, "%Y%m%d").date()
        new_urls, by_source, undated, outside_window = classify_new_products(
            digest_products,
            run_day,
            seen_products,
            descriptions_by_url,
            http_last_modified_by_url,
            wix_lastmod_by_url,
            "unknown",
            window_days=window_days,
            persist_seen=persist_seen,
        )
        filtered_new_items = [
            item
            for item, product in zip(new_items, digest_products)
            if product.url and product.url in new_urls
        ]
        logger.info(
            "%s new products (last %d days ending %s UTC): %d "
            "[shopify_published_at=%d, http_last_modified=%d, "
            "wix_lastmod=%d, seen_at=%d, undated=%d, outside_window=%d].",
            mode_label,
            max(1, window_days),
            run_id,
            len(filtered_new_items),
            by_source["shopify_published_at"],
            by_source["http_last_modified"],
            by_source["wix_lastmod"],
            by_source["seen_at"],
            undated,
            outside_window,
        )
    finally:
        seen_products.close()
    return filtered_new_items


def log_new_products_summary(
    logger: logging.Logger,
    roaster_name: str,
    run_id: str,
    new_urls: set[str],
    by_source: dict[str, int],
    undated: int,
    outside_window: int,
    page_fetch_count: int,
    window_days: int = NEW_PRODUCTS_WINDOW_DAYS,
) -> None:
    logger.info(
        "New products for %s (last %d days ending %s UTC): %d "
        "[shopify_published_at=%d, http_last_modified=%d, wix_lastmod=%d, "
        "seen_at=%d, undated=%d, outside_window=%d]. "
        "Page text fetched for %d products.",
        roaster_name,
        window_days,
        run_id,
        len(new_urls),
        by_source.get("shopify_published_at", 0),
        by_source.get("http_last_modified", 0),
        by_source.get("wix_lastmod", 0),
        by_source.get("seen_at", 0),
        undated,
        outside_window,
        page_fetch_count,
    )
