from __future__ import annotations

import re
from datetime import date
from typing import Any

from .constants import USER_AGENT
from .models import ProductCandidate, RoasterRunStatus, RoasterSource, VariantInfo

SCHEMA_VERSION = 1
STOREFRONT_UNAVAILABLE_STATUS = "storefront_unavailable"
# Sanity bounds for trusting Shopify API grams on placeholder-titled variants.
_API_GRAMS_MIN = 20
_API_GRAMS_MAX = 10000


def _clean(value: str) -> str:
    return " ".join(str(value or "").split())


# Spec-sheet product pages often run one labeled field straight into the next
# without punctuation ("...Peach, Key Lime Origin: El Paraiso Variety: ...");
# a captured value must stop at the next field label or it swallows the rest
# of the spec sheet.
_FIELD_BOUNDARY = re.compile(
    r"\s+(?:origin|region|country|variety|varietal|varieties|producer|producers|"
    r"farm|co-op|coop|elevation|altitude|process|processing|harvest|method|roast|"
    r"importer|sourced\s+by|cup\s+profile|current\s+components|components|"
    r"farm\s+gate\s+price|fob\s+price|notes?)\s*[:|]",
    re.IGNORECASE,
)


def _extract_labeled_value(text: str, labels: tuple[str, ...]) -> str:
    if not text:
        return ""
    for label in labels:
        pattern = re.compile(
            rf"\b{re.escape(label)}\b\s*[:|\-]\s*([^|.;\n]+)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            value = match.group(1)
            boundary = _FIELD_BOUNDARY.search(value)
            if boundary:
                value = value[: boundary.start()]
            return _clean(value.strip(" .,"))
    return ""


def _split_notes(value: str) -> list[str]:
    cleaned = _trim_notes_value(_clean(value))
    if not cleaned:
        return []
    parts = re.split(r"[,;|]|\s*/\s*", cleaned)
    notes: list[str] = []
    for part in parts:
        note = _trim_note(part.strip(" ."))
        # Real flavor notes are short noun phrases; an overlong item means
        # prose or a spec-sheet field leaked past the trimmers.
        if note and len(note) <= 40:
            notes.append(note)
    return notes


def _trim_notes_value(value: str) -> str:
    return _trim_at_markers(
        value,
        (
            " This ",
            " On the ",
            " is ",
            " It ",
            " The ",
            " A ",
            " As ",
            " Coming ",
            " We ",
            " Our ",
        ),
    )


def _trim_note(value: str) -> str:
    trimmed = _trim_at_markers(value, (" is ",))
    words = trimmed.split()
    if len(words) > 2 and words[0].lower() == "candy":
        return words[0]
    return trimmed


def _trim_at_markers(value: str, markers: tuple[str, ...]) -> str:
    for marker in markers:
        index = value.find(marker)
        if index > 0:
            return value[:index].strip(" .")
    return value


_NOTES_OF_PATTERN = re.compile(
    r"\b(?:notes|flavors|flavours|profile)\s+of\s+([^.|;\n]{3,120})",
    re.IGNORECASE,
)
# Trailing process labels that pages append right after an inline notes list.
_NOTES_TAIL_PATTERN = re.compile(
    r"\s+(?:(?:washed|natural|honey|wet)\s+process|washed\s+anaerobic|ethyl\s+acetate)\b.*$",
    re.IGNORECASE,
)
# A "notes of" match containing these words is page boilerplate, not flavor.
_NOTES_BLOCKLIST = ("variety", "processing", "origin", "coffee")


def _extract_tasting_notes(text: str) -> list[str]:
    value = _extract_labeled_value(
        text,
        (
            "tasting notes",
            "tasting note",
            "taste notes",
            "taste note",
            "flavor notes",
            "flavor note",
            "flavour notes",
            "flavour note",
            "notes",
            "flavors",
            "flavours",
            "tastes like",
        ),
    )
    notes = _split_notes(value)
    if notes:
        return notes
    match = _NOTES_OF_PATTERN.search(text or "")
    if match:
        candidate = _NOTES_TAIL_PATTERN.sub("", match.group(1))
        notes = []
        for note in _split_notes(candidate):
            if note.lower().startswith("and "):
                note = note[4:]
            # Real flavor notes are short noun phrases; stop at prose clauses.
            if len(note) > 32 or re.search(
                r"\b(?:that|which|we|this|it|its)\b", note, re.IGNORECASE
            ):
                break
            notes.append(note)
        if 1 < len(notes) <= 6 and not any(
            blocked in note.lower() for note in notes for blocked in _NOTES_BLOCKLIST
        ):
            return notes
    return []


def _variant_to_dict(variant: VariantInfo) -> dict[str, Any]:
    return {
        "title": variant.title,
        "price": variant.price,
        "grams": variant.grams,
        "available": variant.available,
    }


def _normalize_variant_title(title: str) -> str:
    return re.sub(r"\s+", "", title.strip().lower())


def _is_default_variant_title(title: str) -> bool:
    return _normalize_variant_title(title) in {"defaulttitle", "defaultvariant"}


def _variant_sort_key(variant: VariantInfo) -> tuple[int, int, int, str]:
    grind_rank = _variant_grind_rank(variant.title)
    if variant.grams > 0:
        return (0, variant.grams, grind_rank, variant.title)
    return (1, 0, grind_rank, variant.title)


def _variant_grind_rank(title: str) -> int:
    normalized = title.lower()
    if "whole bean" in normalized or "wholebean" in normalized:
        return 0
    grind_terms = (
        "ground",
        "grind",
        "drip",
        "espresso",
        "aeropress",
        "aero press",
        "pour over",
        "pourover",
        "french press",
        "stove top",
        "stovetop",
        "mocha pot",
        "moka pot",
        "brewed coffee",
        "cold brew",
    )
    if any(term in normalized for term in grind_terms):
        return 2
    return 1


def _preferred_price_variant(
    product: ProductCandidate,
) -> tuple[VariantInfo | None, str]:
    if not product.variants:
        return None, ""
    variants = [variant for variant in product.variants if variant.available]
    if not variants:
        variants = list(product.variants)

    visible_titles = {
        _normalize_variant_title(title)
        for title in product.visible_variant_titles
        if title
    }
    if visible_titles:
        visible_variants = [
            variant
            for variant in variants
            if _normalize_variant_title(variant.title) in visible_titles
        ]
        if visible_variants:
            return (
                sorted(visible_variants, key=_variant_sort_key)[0],
                "visible_variant",
            )

    return sorted(variants, key=_variant_sort_key)[0], "variant"


def _money_label(price: str) -> str:
    cleaned = price.strip()
    if not cleaned:
        return ""
    # Bare numbers ("22.00") come from API payloads whose storefronts are
    # priced in USD; anything already carrying a currency marker ("$24.00",
    # "€18.50", "kr 145") must not get a second symbol stacked on top.
    if cleaned[0].isdigit() or cleaned[0] == ".":
        return f"${cleaned}"
    return cleaned


def _price_label(price: str, price_variant: str) -> str:
    if price and price_variant:
        return f"{price_variant} = {_money_label(price)}"
    if price:
        return _money_label(price)
    return ""


def _api_grams_size(variant: VariantInfo) -> str:
    if _API_GRAMS_MIN <= variant.grams <= _API_GRAMS_MAX:
        return f"{variant.grams} g"
    return ""


def _display_price(product: ProductCandidate) -> tuple[str, str, str, str]:
    if product.storefront_status == STOREFRONT_UNAVAILABLE_STATUS:
        return "", "", "", STOREFRONT_UNAVAILABLE_STATUS
    variant, source = _preferred_price_variant(product)
    if variant is not None:
        price = _clean(variant.price)
        if _is_default_variant_title(variant.title):
            price_variant = _api_grams_size(variant)
            if price_variant:
                source = "variant_api_grams"
        else:
            price_variant = _clean(variant.title)
        return price, price_variant, _price_label(price, price_variant), source
    list_price = _clean(product.list_price)
    if list_price:
        return list_price, "", _price_label(list_price, ""), "list_price"
    return "", "", "", ""


def _badge_indicates_unavailable(badge: str) -> bool:
    normalized = badge.strip().lower()
    return any(
        phrase in normalized
        for phrase in ("sold out", "out of stock", "unavailable")
    )


def _availability(product: ProductCandidate, roaster: RoasterSource) -> str:
    if product.storefront_status == STOREFRONT_UNAVAILABLE_STATUS:
        return "unavailable"
    if _badge_indicates_unavailable(product.list_badge):
        return "unavailable"
    if not product.variants:
        if roaster.platform != "shopify" and product.url:
            return "available"
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
    update_date: date | None,
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
    price, price_variant, price_label, price_source = _display_price(product)
    return {
        "roaster": roaster.name,
        "platform": roaster.platform,
        "product_id": product.product_id,
        "product_url": product.url,
        "title": product.name,
        "price": price,
        "price_variant": price_variant,
        "price_label": price_label,
        "price_source": price_source,
        "badge": product.list_badge,
        "origin": origin,
        "process": process,
        "tasting_notes": _extract_tasting_notes(raw_product_text),
        "availability": _availability(product, roaster),
        "variants": [_variant_to_dict(variant) for variant in product.variants],
        "visible_variant_titles": list(product.visible_variant_titles),
        "storefront_status": product.storefront_status,
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
