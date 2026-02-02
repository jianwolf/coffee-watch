from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

from .config import Settings
from .models import (
    PaginationConfig,
    ProductCandidate,
    ProductFieldConfig,
    RoasterSource,
    VariantInfo,
)
from .text_utils import LinkParser, guess_name_from_url
from .url_utils import canonicalize_url, matches_patterns, normalize_base_url


def load_denylist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line.lower())
    return entries


def to_str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, val in value.items():
        if key is None or val is None:
            continue
        result[str(key)] = str(val)
    return result


def to_str_tuple(
    value: Any, default: tuple[str, ...], fallback_on_empty: bool = True
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        if not value:
            return default if fallback_on_empty else ()
        items = [str(item) for item in value if str(item)]
        if items:
            return tuple(items)
        return default if fallback_on_empty else ()
    return default


def parse_items_path(value: Any) -> Optional[tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(item) for item in value if str(item)]
        return tuple(parts) if parts else None
    if isinstance(value, str):
        parts = [part for part in value.replace("/", ".").split(".") if part]
        return tuple(parts) if parts else None
    return None


def normalize_tokens(values: tuple[str, ...]) -> set[str]:
    return {value.strip().lower() for value in values if value.strip()}


def normalize_tags(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return {part.lower() for part in parts if part}
    return set()


def product_matches_filters(
    item: dict[str, Any],
    include_tags: set[str],
    exclude_tags: set[str],
    include_types: set[str],
    exclude_types: set[str],
    exclude_title_keywords: tuple[str, ...],
) -> bool:
    tags = normalize_tags(item.get("tags"))
    product_type = str(item.get("product_type") or "").strip().lower()
    title = str(item.get("title") or item.get("name") or "").strip().lower()

    if include_tags and not (tags & include_tags):
        return False
    if exclude_tags and (tags & exclude_tags):
        return False
    if include_types and product_type not in include_types:
        return False
    if exclude_types and product_type in exclude_types:
        return False
    for keyword in exclude_title_keywords:
        if keyword and keyword in title:
            return False
    return True


def parse_product_fields(value: Any) -> Optional[ProductFieldConfig]:
    if not isinstance(value, dict):
        return None
    return ProductFieldConfig(
        name_fields=to_str_tuple(value.get("name"), ProductFieldConfig().name_fields),
        url_fields=to_str_tuple(value.get("url"), ProductFieldConfig().url_fields),
        handle_fields=to_str_tuple(
            value.get("handle"), ProductFieldConfig().handle_fields
        ),
        id_fields=to_str_tuple(value.get("id"), ProductFieldConfig().id_fields),
    )


def parse_pagination(value: Any) -> Optional[PaginationConfig]:
    if not isinstance(value, dict):
        return None
    return PaginationConfig(
        page_param=str(value.get("page_param", "page")),
        start=int(value.get("start", 1)),
        max_pages=int(value.get("max_pages", 1)),
        page_size_param=value.get("page_size_param"),
        page_size=int(value["page_size"]) if value.get("page_size") is not None else None,
        stop_on_empty=bool(value.get("stop_on_empty", True)),
    )


def load_roasters(settings: Settings, logger: logging.Logger) -> list[RoasterSource]:
    data: list[dict[str, Any]] = []
    if settings.roasters_path.exists():
        try:
            data = json.loads(settings.roasters_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Invalid roasters file %s: %s", settings.roasters_path, exc)
    else:
        logger.warning("Roasters file not found: %s", settings.roasters_path)

    if not data:
        return [
            RoasterSource(
                name="Example Roaster",
                base_url="https://example.com",
                products_path="/products.json",
            )
        ]

    roasters: list[RoasterSource] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        base_url = normalize_base_url(str(entry.get("base_url", "")).strip())
        if not base_url:
            continue
        products_type = str(entry.get("products_type", "auto")).lower()
        if products_type not in {"auto", "json", "html"}:
            logger.warning(
                "Invalid products_type %s for %s; defaulting to auto.",
                products_type,
                base_url,
            )
            products_type = "auto"
        roaster = RoasterSource(
            name=str(entry.get("name", base_url)),
            base_url=base_url,
            platform=str(entry.get("platform", "unknown")).strip().lower() or "unknown",
            products_path=str(entry.get("products_path", "/products.json")),
            enabled=bool(entry.get("enabled", True)),
            products_type=products_type,
            products_parser=(
                str(entry.get("products_parser")).strip()
                if entry.get("products_parser") is not None
                else None
            ),
            jitter_multiplier=float(entry.get("jitter_multiplier", 1.0)),
            products_headers=to_str_dict(entry.get("products_headers")),
            products_params=to_str_dict(entry.get("products_params")),
            product_page_headers=to_str_dict(entry.get("product_page_headers")),
            product_link_patterns=to_str_tuple(
                entry.get("product_link_patterns"),
                ("/products/", "/product/"),
                fallback_on_empty=False,
            ),
            product_link_exclude_patterns=to_str_tuple(
                entry.get("product_link_exclude_patterns"),
                (),
                fallback_on_empty=False,
            ),
            product_url_template=entry.get("product_url_template"),
            json_items_path=parse_items_path(entry.get("json_items_path")),
            product_fields=parse_product_fields(entry.get("product_fields")),
            pagination=parse_pagination(entry.get("pagination")),
            max_products=(
                int(entry["max_products"])
                if entry.get("max_products") is not None
                else None
            ),
            page_text_stop_phrases=to_str_tuple(
                entry.get("page_text_stop_phrases"), (), fallback_on_empty=False
            ),
            include_tags=to_str_tuple(
                entry.get("include_tags"), (), fallback_on_empty=False
            ),
            exclude_tags=to_str_tuple(
                entry.get("exclude_tags"), (), fallback_on_empty=False
            ),
            include_product_types=to_str_tuple(
                entry.get("include_product_types"), (), fallback_on_empty=False
            ),
            exclude_product_types=to_str_tuple(
                entry.get("exclude_product_types"), (), fallback_on_empty=False
            ),
            exclude_title_keywords=to_str_tuple(
                entry.get("exclude_title_keywords"), (), fallback_on_empty=False
            ),
        )
        if not roaster.enabled:
            logger.info("Roaster disabled in config: %s", roaster.name)
            continue
        roasters.append(roaster)
    return roasters


def extract_items_by_path(data: Any, path: Optional[tuple[str, ...]]) -> list[Any]:
    if not path:
        return []
    current: Any = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return []
    return current if isinstance(current, list) else []


def extract_items_list(data: Any, items_path: Optional[tuple[str, ...]]) -> list[Any]:
    if items_path:
        items = extract_items_by_path(data, items_path)
        if items:
            return items
    if isinstance(data, dict):
        for key in ("products", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    return []


def first_field(item: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field_name in fields:
        value = item.get(field_name)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return ""


def product_id_from_url(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return digest


def parse_variants(value: Any) -> tuple[VariantInfo, ...]:
    variants_raw = value if isinstance(value, list) else []
    variants: list[VariantInfo] = []
    for variant in variants_raw:
        if not isinstance(variant, dict):
            continue
        title = str(variant.get("title") or "").strip()
        price = str(variant.get("price") or "").strip()
        grams_value = variant.get("grams")
        grams = int(grams_value) if isinstance(grams_value, (int, float)) else 0
        available = bool(variant.get("available", False))
        variants.append(
            VariantInfo(
                title=title,
                price=price,
                grams=grams,
                available=available,
            )
        )
    return tuple(variants)


def resolve_product_url(
    base_url: str,
    url: str,
    handle: str,
    item_id: str,
    url_template: Optional[str],
) -> str:
    resolved = url
    if not resolved and (handle or item_id) and url_template:
        try:
            resolved = url_template.format(handle=handle, id=item_id, slug=handle)
        except KeyError:
            resolved = ""
    if not resolved and handle:
        resolved = urljoin(base_url, f"/products/{handle}")
    if not resolved:
        return ""
    resolved = urljoin(base_url, resolved) if resolved.startswith("/") else resolved
    return canonicalize_url(resolved)


def parse_products_json(
    data: Any,
    base_url: str,
    roaster: RoasterSource,
    max_count: int,
    product_fields: ProductFieldConfig,
    url_template: Optional[str],
    items_path: Optional[tuple[str, ...]],
) -> list[ProductCandidate]:
    items = extract_items_list(data, items_path)

    products: list[ProductCandidate] = []
    include_tags = normalize_tokens(roaster.include_tags)
    exclude_tags = normalize_tokens(roaster.exclude_tags)
    include_types = normalize_tokens(roaster.include_product_types)
    exclude_types = normalize_tokens(roaster.exclude_product_types)
    for item in items:
        if not isinstance(item, dict):
            continue
        if not product_matches_filters(
            item,
            include_tags,
            exclude_tags,
            include_types,
            exclude_types,
            roaster.exclude_title_keywords,
        ):
            continue
        name = first_field(item, product_fields.name_fields)
        url = first_field(item, product_fields.url_fields)
        handle = first_field(item, product_fields.handle_fields)
        item_id = first_field(item, product_fields.id_fields)
        body_html = str(item.get("body_html") or "").strip()
        updated_at = str(item.get("updated_at") or "").strip()
        published_at = str(item.get("published_at") or "").strip()
        variants = parse_variants(item.get("variants"))

        url = resolve_product_url(base_url, url, handle, item_id, url_template)
        if not url:
            continue
        if not name:
            name = guess_name_from_url(url)
        product_id = (
            f"{roaster.name}:{item_id}" if item_id else product_id_from_url(url)
        )
        products.append(
            ProductCandidate(
                product_id=product_id,
                name=name,
                url=url,
                source=roaster.name,
                body_html=body_html,
                variants=tuple(variants),
                shopify_updated_at=updated_at,
                shopify_published_at=published_at,
            )
        )
        if len(products) >= max_count:
            break
    return products


def parse_products_html(
    html: str,
    base_url: str,
    source: str,
    max_count: int,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
    exclude_title_keywords: tuple[str, ...],
) -> list[ProductCandidate]:
    generic_link_tokens = {
        "quick view",
        "view",
        "view product",
        "view item",
        "details",
        "learn more",
        "read more",
        "shop",
        "add to cart",
        "buy now",
        "new harvest",
    }
    excluded_keywords = normalize_tokens(exclude_title_keywords)
    parser = LinkParser()
    parser.feed(html)
    products: list[ProductCandidate] = []
    seen: set[str] = set()
    for href, text in parser.links:
        resolved = urljoin(base_url, href)
        if not matches_patterns(resolved, include_patterns, exclude_patterns):
            continue
        url = canonicalize_url(resolved)
        if url in seen:
            continue
        seen.add(url)
        link_text = " ".join(text.split()).strip()
        link_text_lower = link_text.lower()
        if not link_text or link_text_lower in generic_link_tokens or "quick view" in link_text_lower:
            name = guess_name_from_url(url)
        else:
            name = link_text
        if excluded_keywords:
            name_lower = name.lower()
            url_lower = url.lower()
            if any(
                keyword in name_lower
                or keyword in url_lower
                or keyword.replace(" ", "-") in url_lower
                for keyword in excluded_keywords
            ):
                continue
        products.append(
            ProductCandidate(
                product_id=product_id_from_url(url),
                name=name,
                url=url,
                source=source,
                list_price="",
                list_badge="",
                body_html="",
                variants=(),
            )
        )
        if len(products) >= max_count:
            break
    return products


def parse_products_response(
    content: bytes,
    content_type: str,
    base_url: str,
    roaster: RoasterSource,
    max_count: int,
    logger: logging.Logger,
) -> list[ProductCandidate]:
    product_fields = roaster.product_fields or ProductFieldConfig()
    force_type = roaster.products_type
    if force_type == "json" or (force_type == "auto" and "json" in content_type.lower()):
        try:
            data = json.loads(content.decode("utf-8"))
            return parse_products_json(
                data,
                base_url,
                roaster,
                max_count,
                product_fields,
                roaster.product_url_template,
                roaster.json_items_path,
            )
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse JSON for %s; falling back to HTML.", roaster.name
            )
    text = content.decode("utf-8", errors="ignore")
    if roaster.products_parser:
        from .catalog_parsers import parse_catalog_html

        parsed = parse_catalog_html(
            roaster.products_parser,
            text,
            base_url,
            roaster,
            max_count,
            logger,
        )
        if parsed:
            return parsed
    return parse_products_html(
        text,
        base_url,
        roaster.name,
        max_count,
        roaster.product_link_patterns,
        roaster.product_link_exclude_patterns,
        roaster.exclude_title_keywords,
    )
