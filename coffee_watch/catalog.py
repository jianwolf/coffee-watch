from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from .constants import USER_AGENT
from .models import ProductCandidate, RoasterRunStatus, RoasterSource, VariantInfo

SCHEMA_VERSION = 1


def _clean(value: str) -> str:
    return " ".join(str(value or "").split())


def _first_present(*values: str) -> str:
    for value in values:
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return ""


def _extract_labeled_value(text: str, labels: tuple[str, ...]) -> str:
    if not text:
        return ""
    for label in labels:
        pattern = re.compile(
            rf"\b{re.escape(label)}\b\s*[:\-]\s*([^|.;\n]+)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return _clean(match.group(1))
    return ""


def _split_notes(value: str) -> list[str]:
    cleaned = _clean(value)
    if not cleaned:
        return []
    parts = re.split(r",|;|\|| / ", cleaned)
    return [part.strip(" .") for part in parts if part.strip(" .")]


def _extract_tasting_notes(text: str) -> list[str]:
    value = _extract_labeled_value(
        text,
        (
            "tasting notes",
            "taste notes",
            "flavor notes",
            "flavour notes",
            "notes",
            "flavors",
            "flavours",
            "tastes like",
        ),
    )
    return _split_notes(value)


def _variant_to_dict(variant: VariantInfo) -> dict[str, Any]:
    return {
        "title": variant.title,
        "price": variant.price,
        "grams": variant.grams,
        "available": variant.available,
    }


def _availability(product: ProductCandidate) -> str:
    if not product.variants:
        return "unknown"
    if any(variant.available for variant in product.variants):
        return "available"
    return "unavailable"


def catalog_product_from_candidate(
    *,
    product: ProductCandidate,
    roaster: RoasterSource,
    raw_product_text: str,
    first_seen_at: str,
    is_new: bool,
    date_source: str,
    update_date: Optional[date],
    http_last_modified: str,
    wix_lastmod: str,
    errors: list[str],
) -> dict[str, Any]:
    process = _extract_labeled_value(
        raw_product_text,
        ("process", "processing", "processing method", "method"),
    )
    origin = _extract_labeled_value(
        raw_product_text,
        ("origin", "country", "region", "farm", "producer"),
    )
    return {
        "roaster": roaster.name,
        "platform": roaster.platform,
        "product_id": product.product_id,
        "product_url": product.url,
        "title": product.name,
        "price": _first_present(
            product.list_price,
            *(variant.price for variant in product.variants),
        ),
        "badge": product.list_badge,
        "origin": origin,
        "process": process,
        "tasting_notes": _extract_tasting_notes(raw_product_text),
        "availability": _availability(product),
        "variants": [_variant_to_dict(variant) for variant in product.variants],
        "first_seen_at": first_seen_at,
        "is_new": is_new,
        "date_source": date_source,
        "update_date": update_date.isoformat() if update_date else "",
        "shopify_updated_at": product.shopify_updated_at,
        "shopify_published_at": product.shopify_published_at,
        "http_last_modified": http_last_modified,
        "wix_lastmod": wix_lastmod,
        "source": product.source,
        "raw_product_text": raw_product_text,
        "errors": errors,
    }


def build_roaster_catalog(
    *,
    roaster: RoasterSource,
    run_id: str,
    generated_at: str,
    status: RoasterRunStatus,
    products: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "roaster": {
            "name": roaster.name,
            "base_url": roaster.base_url,
            "platform": roaster.platform,
        },
        "status": status.to_dict(),
        "summary": {
            "products_found": status.products_found,
            "new_products": status.new_products,
            "fetched_pages": status.fetched_pages,
        },
        "products": products,
        "errors": errors,
    }


def build_combined_catalog(
    *,
    run_id: str,
    generated_at: str,
    roasters_path: str,
    denylist_path: str,
    roaster_catalogs: list[dict[str, Any]],
    failed_roasters: list[str],
) -> dict[str, Any]:
    products = [
        product
        for catalog in roaster_catalogs
        for product in catalog.get("products", [])
        if isinstance(product, dict)
    ]
    roasters = []
    for catalog in roaster_catalogs:
        roaster = catalog.get("roaster")
        if not isinstance(roaster, dict):
            continue
        roaster_entry = dict(roaster)
        roaster_entry["status"] = catalog.get("status", {})
        roaster_entry["summary"] = catalog.get("summary", {})
        roaster_entry["errors"] = catalog.get("errors", [])
        roasters.append(roaster_entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "source": {
            "roasters_path": roasters_path,
            "denylist_path": denylist_path,
            "user_agent": USER_AGENT,
        },
        "summary": {
            "roasters": len(roaster_catalogs),
            "products": len(products),
            "new_products": sum(1 for product in products if product.get("is_new")),
            "failed_roasters": failed_roasters,
        },
        "roasters": roasters,
        "products": products,
    }


def build_new_products_catalog(combined_catalog: dict[str, Any]) -> dict[str, Any]:
    products = [
        product
        for product in combined_catalog.get("products", [])
        if isinstance(product, dict) and product.get("is_new")
    ]
    return {
        "schema_version": combined_catalog.get("schema_version", SCHEMA_VERSION),
        "run_id": combined_catalog.get("run_id", ""),
        "generated_at": combined_catalog.get("generated_at", ""),
        "source": combined_catalog.get("source", {}),
        "summary": {
            "products": len(products),
        },
        "products": products,
    }
