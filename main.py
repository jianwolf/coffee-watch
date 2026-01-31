from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import logging.handlers
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, confloat

USER_AGENT = "CoffeeWatch-Bot/1.0 (+https://github.com/jianwolf/coffee-watch; contact: GitHub Issues)"


@dataclass(frozen=True)
class Settings:
    model: str
    http_timeout_s: float
    jitter_min_s: float
    jitter_max_s: float
    http_concurrency: int
    max_products_per_source: int
    page_text_max_chars: int
    batch_page_text_max_chars: int
    log_json_max_chars: int
    fetch_only: bool
    skip_gemini: bool
    save_prompt: bool
    save_pretty_products_json: bool
    save_raw_products_json: bool
    save_report: bool
    roasters_path: Path
    denylist_path: Path
    reports_dir: Path
    log_path: Path
    log_level: str
    language: str

    @staticmethod
    def from_env() -> "Settings":
        def env_bool(key: str, default: bool) -> bool:
            value = os.getenv(key)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}

        def env_float(key: str, default: float) -> float:
            value = os.getenv(key)
            return float(value) if value else default

        def env_int(key: str, default: int) -> int:
            value = os.getenv(key)
            return int(value) if value else default

        return Settings(
            model=os.getenv("COFFEEWATCH_MODEL", "gemini-3-pro-preview"),
            http_timeout_s=env_float("COFFEEWATCH_HTTP_TIMEOUT_S", 20.0),
            jitter_min_s=env_float("COFFEEWATCH_JITTER_MIN_S", 0.7),
            jitter_max_s=env_float("COFFEEWATCH_JITTER_MAX_S", 2.0),
            http_concurrency=env_int("COFFEEWATCH_HTTP_CONCURRENCY", 4),
            max_products_per_source=env_int("COFFEEWATCH_MAX_PRODUCTS", 200),
            page_text_max_chars=env_int("COFFEEWATCH_PAGE_TEXT_MAX_CHARS", 0),
            batch_page_text_max_chars=env_int(
                "COFFEEWATCH_BATCH_PAGE_TEXT_MAX_CHARS", 0
            ),
            log_json_max_chars=env_int("COFFEEWATCH_LOG_JSON_MAX_CHARS", 0),
            fetch_only=env_bool("COFFEEWATCH_FETCH_ONLY", False),
            skip_gemini=env_bool("COFFEEWATCH_SKIP_GEMINI", False),
            save_prompt=env_bool("COFFEEWATCH_SAVE_PROMPT", True),
            save_pretty_products_json=env_bool(
                "COFFEEWATCH_SAVE_PRETTY_PRODUCTS_JSON", True
            ),
            save_raw_products_json=env_bool(
                "COFFEEWATCH_SAVE_RAW_PRODUCTS_JSON", False
            ),
            save_report=env_bool("COFFEEWATCH_SAVE_REPORT", True),
            roasters_path=Path(os.getenv("COFFEEWATCH_ROASTERS_PATH", "config/roasters.json")),
            denylist_path=Path(os.getenv("COFFEEWATCH_DENYLIST_PATH", "config/denylist.txt")),
            reports_dir=Path(os.getenv("COFFEEWATCH_REPORTS_DIR", "reports")),
            log_path=Path(os.getenv("COFFEEWATCH_LOG_PATH", "logs/coffee_watch.log")),
            log_level=os.getenv("COFFEEWATCH_LOG_LEVEL", "INFO"),
            language=os.getenv("COFFEEWATCH_LANGUAGE", "en"),
        )


@dataclass(frozen=True)
class RoasterSource:
    name: str
    base_url: str
    products_path: str = "/products.json"
    enabled: bool = True
    products_type: str = "auto"  # auto|json|html
    products_headers: dict[str, str] = field(default_factory=dict)
    products_params: dict[str, str] = field(default_factory=dict)
    product_page_headers: dict[str, str] = field(default_factory=dict)
    product_link_patterns: tuple[str, ...] = ("/products/", "/product/")
    product_link_exclude_patterns: tuple[str, ...] = ()
    product_url_template: Optional[str] = None
    json_items_path: Optional[tuple[str, ...]] = None
    product_fields: Optional["ProductFieldConfig"] = None
    pagination: Optional["PaginationConfig"] = None
    max_products: Optional[int] = None
    include_tags: tuple[str, ...] = ()
    exclude_tags: tuple[str, ...] = ()
    include_product_types: tuple[str, ...] = ()
    exclude_product_types: tuple[str, ...] = ()
    exclude_title_keywords: tuple[str, ...] = ()

    @property
    def products_url(self) -> str:
        base = self.base_url if self.base_url.endswith("/") else f"{self.base_url}/"
        return urljoin(base, self.products_path.lstrip("/"))


@dataclass(frozen=True)
class ProductCandidate:
    product_id: str
    name: str
    url: str
    source: str
    body_html: str = ""
    variants: tuple["VariantInfo", ...] = ()


@dataclass(frozen=True)
class VariantInfo:
    title: str
    price: str
    grams: int
    available: bool


class CoffeeScore(BaseModel):
    score: confloat(ge=0, le=10)
    worth_trying: bool
    summary: str = Field(..., description="Short, neutral summary of the coffee.")
    rationale: str = Field(..., description="Brief, critical explanation for the score.")
    rarity_signals: list[str] = Field(default_factory=list)
    reputation_signals: list[str] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1) = 0.5


class CoffeeEvaluation(BaseModel):
    product_id: str = Field(..., description="Use the provided product_id exactly.")
    name: str
    url: str
    score: confloat(ge=0, le=10)
    worth_trying: bool
    summary: str
    rationale: str
    rarity_signals: list[str] = Field(default_factory=list)
    reputation_signals: list[str] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1) = 0.5


class CoffeeBatchResult(BaseModel):
    evaluations: list[CoffeeEvaluation]


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: Optional[str] = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self._current_href = href
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href is not None:
            text = " ".join(part.strip() for part in self._text_parts if part.strip())
            self.links.append((self._current_href, text.strip()))
            self._current_href = None
            self._text_parts = []


class VisibleTextExtractor(HTMLParser):
    _skip_tags = {"script", "style", "head", "noscript", "svg", "meta", "link"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        elif tag in {"br", "p", "li", "div", "section"} and self._skip_depth == 0:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = " ".join(data.split())
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        parts: list[str] = []
        for chunk in self._chunks:
            if chunk == "\n":
                parts.append("\n")
            else:
                parts.append(f"{chunk} ")
        return "".join(parts)


@dataclass(frozen=True)
class ProductFieldConfig:
    name_fields: tuple[str, ...] = ("title", "name")
    url_fields: tuple[str, ...] = ("url", "product_url")
    handle_fields: tuple[str, ...] = ("handle", "slug")
    id_fields: tuple[str, ...] = ("id", "product_id")


@dataclass(frozen=True)
class PaginationConfig:
    page_param: str = "page"
    start: int = 1
    max_pages: int = 1
    page_size_param: Optional[str] = None
    page_size: Optional[int] = None
    stop_on_empty: bool = True


def setup_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    while root.handlers:
        root.handlers.pop()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)


def normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    parsed = urlsplit(url)
    if not parsed.scheme:
        url = f"https://{url.lstrip('/')}"
        parsed = urlsplit(url)
    normalized = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    return normalized or url


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def build_url_with_params(base_url: str, path: str, params: dict[str, str]) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        url = path
    else:
        base = base_url if base_url.endswith("/") else f"{base_url}/"
        url = urljoin(base, path.lstrip("/"))
    if not params:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    for key, value in params.items():
        query[str(key)] = str(value)
    new_query = urlencode(query)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, ""))


def merge_headers(
    base_headers: dict[str, str],
    extra_headers: dict[str, str],
    logger: logging.Logger,
    context: str,
) -> dict[str, str]:
    headers = dict(base_headers)
    for key, value in extra_headers.items():
        if key.lower() == "user-agent":
            logger.warning("Ignoring custom User-Agent for %s", context)
            continue
        headers[key] = value
    return headers


def matches_patterns(url: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    def match(pattern: str) -> bool:
        if pattern.startswith("re:"):
            try:
                return re.search(pattern[3:], url) is not None
            except re.error:
                return False
        return pattern in url

    if exclude and any(match(pattern) for pattern in exclude):
        return False
    if not include:
        return True
    return any(match(pattern) for pattern in include)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "roaster"


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
    kind: str,
    ext: str,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_slug(roaster_name)
    filename = f"{slug}.{kind}.{run_id}.{ext}"
    return reports_dir / filename


def save_products_json(
    report_dir: Path,
    run_id: str,
    roaster: RoasterSource,
    page_index: int,
    json_text: str,
) -> Path:
    path = report_file_path(
        report_dir, roaster.name, f"{run_id}-page{page_index}", "products.raw", "json"
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
        report_dir, roaster.name, f"{run_id}-page{page_index}", "products.pretty", "json"
    )
    path.write_text(pretty, encoding="utf-8")
    return path


def product_id_from_url(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return digest


def guess_name_from_url(url: str) -> str:
    path = urlsplit(url).path.rstrip("/")
    if not path:
        return "Unknown Coffee"
    slug = path.split("/")[-1]
    return slug.replace("-", " ").replace("_", " ").strip().title() or "Unknown Coffee"


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
    env_json = os.getenv("COFFEEWATCH_ROASTERS_JSON", "").strip()
    data: list[dict[str, Any]] = []
    if env_json:
        try:
            data = json.loads(env_json)
        except json.JSONDecodeError as exc:
            logger.error("Invalid COFFEEWATCH_ROASTERS_JSON: %s", exc)
    elif settings.roasters_path.exists():
        try:
            data = json.loads(settings.roasters_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Invalid roasters file %s: %s", settings.roasters_path, exc)

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
            products_path=str(entry.get("products_path", "/products.json")),
            enabled=bool(entry.get("enabled", True)),
            products_type=products_type,
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


def first_field(item: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field_name in fields:
        value = item.get(field_name)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return ""


def parse_products_json(
    data: Any,
    base_url: str,
    roaster: RoasterSource,
    max_count: int,
    product_fields: ProductFieldConfig,
    url_template: Optional[str],
    items_path: Optional[tuple[str, ...]],
) -> list[ProductCandidate]:
    items: list[Any] = []
    if items_path:
        items = extract_items_by_path(data, items_path)
    if not items:
        if isinstance(data, dict):
            if isinstance(data.get("products"), list):
                items = data["products"]
            elif isinstance(data.get("items"), list):
                items = data["items"]
            elif isinstance(data.get("data"), list):
                items = data["data"]
            else:
                items = [data]
        elif isinstance(data, list):
            items = data

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
        variants_raw = item.get("variants") if isinstance(item.get("variants"), list) else []
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

        if not url and (handle or item_id) and url_template:
            try:
                url = url_template.format(handle=handle, id=item_id, slug=handle)
            except KeyError:
                url = ""
        if not url and handle:
            url = urljoin(base_url, f"/products/{handle}")
        if not url:
            continue
        url = canonicalize_url(urljoin(base_url, url) if url.startswith("/") else url)
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
) -> list[ProductCandidate]:
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
        name = text.strip() or guess_name_from_url(url)
        products.append(
            ProductCandidate(
                product_id=product_id_from_url(url),
                name=name,
                url=url,
                source=source,
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
    try:
        text = content.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        text = content.decode(errors="ignore")
    return parse_products_html(
        text,
        base_url,
        roaster.name,
        max_count,
        roaster.product_link_patterns,
        roaster.product_link_exclude_patterns,
    )


async def fetch_products_for_roaster(
    http_client: httpx.AsyncClient,
    roaster: RoasterSource,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    report_dir: Path,
    run_id: str,
    semaphore: Optional[asyncio.Semaphore],
    logger: logging.Logger,
) -> list[ProductCandidate]:
    max_products = roaster.max_products or settings.max_products_per_source
    pagination = roaster.pagination or PaginationConfig(max_pages=1)
    products: list[ProductCandidate] = []
    headers = merge_headers(
        {"User-Agent": USER_AGENT},
        roaster.products_headers,
        logger,
        f"{roaster.name} products",
    )

    for page_index in range(pagination.start, pagination.start + pagination.max_pages):
        params = dict(roaster.products_params)
        if pagination.max_pages > 1 or "page" in params:
            params[pagination.page_param] = str(page_index)
        if pagination.page_size_param and pagination.page_size:
            params[pagination.page_size_param] = str(pagination.page_size)

        products_url = build_url_with_params(roaster.base_url, roaster.products_path, params)
        allowed = await robots_allows(
            http_client, products_url, settings, robots_cache, logger
        )
        if not allowed:
            logger.warning(
                "Robots.txt disallows %s for %s; skipping.",
                products_url,
                roaster.name,
            )
            if pagination.stop_on_empty:
                break
            continue

        response = await fetch_text_with_jitter(
            http_client, products_url, settings, logger, headers=headers, semaphore=semaphore
        )
        if response is None:
            logger.warning("Request failed for %s", products_url)
            if pagination.stop_on_empty:
                break
            continue
        if response.status_code >= 400:
            logger.warning(
                "Non-200 response %s for %s", response.status_code, products_url
            )
            if pagination.stop_on_empty:
                break
            continue

        content_type = response.headers.get("content-type", "")
        page_products: list[ProductCandidate] = []
        if roaster.products_type == "json" or "json" in content_type.lower():
            json_text = response.text
            log_products_json(
                logger,
                roaster,
                products_url,
                json_text,
                settings.log_json_max_chars,
            )
            if settings.save_raw_products_json:
                path = save_products_json(
                    report_dir, run_id, roaster, page_index, json_text
                )
                logger.info("Saved raw products JSON for %s to %s", roaster.name, path)
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON for %s: %s", roaster.name, exc)
            else:
                if settings.save_pretty_products_json:
                    pretty_path = save_products_json_pretty(
                        report_dir, run_id, roaster, page_index, data
                    )
                    if pretty_path:
                        logger.info(
                            "Saved pretty products JSON for %s to %s",
                            roaster.name,
                            pretty_path,
                        )
                product_fields = roaster.product_fields or ProductFieldConfig()
                page_products = parse_products_json(
                    data,
                    roaster.base_url,
                    roaster,
                    max_products - len(products),
                    product_fields,
                    roaster.product_url_template,
                    roaster.json_items_path,
                )
        if not page_products:
            page_products = parse_products_response(
                response.content,
                content_type,
                roaster.base_url,
                roaster,
                max_products - len(products),
                logger,
            )
        if not page_products and pagination.stop_on_empty:
            break
        products.extend(page_products)
        if len(products) >= max_products:
            break

    return products


async def fetch_product_page_text(
    http_client: httpx.AsyncClient,
    product: ProductCandidate,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    logger: logging.Logger,
    headers: dict[str, str],
    semaphore: Optional[asyncio.Semaphore],
) -> str:
    product_allowed = await robots_allows(
        http_client, product.url, settings, robots_cache, logger
    )
    if not product_allowed:
        logger.warning(
            "Robots.txt disallows product page %s; skipping page fetch.",
            product.url,
        )
        return ""

    page_response = await fetch_text_with_jitter(
        http_client,
        product.url,
        settings,
        logger,
        headers=headers,
        semaphore=semaphore,
    )
    if page_response is None:
        logger.warning("Request failed for product page %s", product.url)
        return ""
    if page_response.status_code >= 400:
        logger.warning(
            "Non-200 response %s for product page %s",
            page_response.status_code,
            product.url,
        )
        return ""
    html = page_response.text
    page_text = sanitize_html_to_text(html, settings.page_text_max_chars)
    logger.info(
        "Sanitized %s chars of page text for %s", len(page_text), product.url
    )
    return page_text


async def jitter_sleep(min_s: float, max_s: float) -> None:
    if max_s <= 0:
        return
    low = min(min_s, max_s)
    high = max(min_s, max_s)
    await asyncio.sleep(random.uniform(low, high))


async def fetch_text_with_jitter(
    client: httpx.AsyncClient,
    url: str,
    settings: Settings,
    logger: logging.Logger,
    headers: Optional[dict[str, str]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Optional[httpx.Response]:
    async def _run() -> Optional[httpx.Response]:
        await jitter_sleep(settings.jitter_min_s, settings.jitter_max_s)
        logger.info("HTTP GET %s", url)
        try:
            response = await client.get(url, headers=headers)
            logger.info("HTTP %s %s", response.status_code, url)
            return response
        except httpx.RequestError as exc:
            logger.warning("HTTP request failed for %s: %s", url, exc)
            return None

    if semaphore is None:
        return await _run()
    async with semaphore:
        return await _run()


async def robots_allows(
    client: httpx.AsyncClient,
    products_url: str,
    settings: Settings,
    cache: dict[str, RobotFileParser],
    logger: logging.Logger,
) -> bool:
    parsed = urlsplit(products_url)
    if not parsed.scheme or not parsed.netloc:
        return False
    cache_key = f"{parsed.scheme}://{parsed.netloc}"
    if cache_key in cache:
        return cache[cache_key].can_fetch(USER_AGENT, products_url)

    robots_url = urljoin(cache_key, "/robots.txt")
    response = await fetch_text_with_jitter(client, robots_url, settings, logger)
    parser = RobotFileParser()
    if response is None:
        logger.warning("Robots.txt fetch failed for %s; proceeding cautiously.", cache_key)
        parser.parse([])
        cache[cache_key] = parser
        return parser.can_fetch(USER_AGENT, products_url)
    if response.status_code >= 400:
        logger.info("Robots.txt not found for %s; proceeding with allowed default.", cache_key)
        parser.parse([])
        cache[cache_key] = parser
        return parser.can_fetch(USER_AGENT, products_url)
    parser.parse(response.text.splitlines())
    cache[cache_key] = parser
    return parser.can_fetch(USER_AGENT, products_url)


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?[\d][\d\s().-]{7,}\d)")
BOILERPLATE_PHRASES = {
    "cookie",
    "privacy policy",
    "terms of service",
    "newsletter",
    "subscribe",
    "sign up",
    "log in",
    "login",
    "add to cart",
    "cart",
    "checkout",
    "shipping",
    "search",
    "filter",
}


def sanitize_html_to_text(
    html: str, max_chars: int, remove_boilerplate: bool = True
) -> str:
    extractor = VisibleTextExtractor()
    extractor.feed(html)
    raw = extractor.text()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    filtered: list[str] = []
    for line in lines:
        if remove_boilerplate:
            lowered = line.lower()
            if any(phrase in lowered for phrase in BOILERPLATE_PHRASES):
                continue
        filtered.append(line)
    text = " ".join(filtered)
    text = " ".join(text.split())
    text = "".join(ch for ch in text if ch.isprintable())
    text = EMAIL_RE.sub("[redacted email]", text)

    def redact_phone(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) >= 10:
            return "[redacted phone]"
        return match.group(0)

    text = PHONE_RE.sub(redact_phone, text)
    if max_chars <= 0:
        return text
    return text[:max_chars]

def sanitize_prompt_field(value: str, max_chars: int) -> str:
    cleaned = " ".join(value.split())
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    if max_chars <= 0:
        return cleaned
    return cleaned[:max_chars]


def format_variant_lines(variants: tuple[VariantInfo, ...]) -> list[str]:
    if not variants:
        return ["  variants: (none)"]
    lines = ["  variants:"]
    for variant in variants:
        parts: list[str] = []
        if variant.title:
            parts.append(variant.title)
        if variant.price:
            parts.append(f"${variant.price}")
        if variant.grams:
            parts.append(f"{variant.grams}g")
        parts.append("available" if variant.available else "unavailable")
        lines.append(f"    - {' | '.join(parts)}")
    return lines


def normalize_language(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"zh", "zh-cn", "zh-hans", "cn", "simplified", "simplified-chinese"}:
        return "zh"
    return "en"


def language_instruction(language: str) -> str:
    if normalize_language(language) == "zh":
        return "Use 简体中文 for the entire report."
    return "Use English for the entire report."


def format_coffee_list(products: list[ProductCandidate]) -> str:
    if not products:
        return ""
    lines = ["## Coffee list", ""]
    for product in products:
        name = product.name.strip() or "Unknown"
        url = product.url.strip()
        if url:
            lines.append(f"- {name} ({url})")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines) + "\n"


def build_prompt(
    product: ProductCandidate, page_text: str, max_chars: int, language: str
) -> str:
    excerpt = page_text.strip()
    if excerpt:
        excerpt = f"Sanitized page text:\n{excerpt}"
    else:
        excerpt = "Sanitized page text: (none available)"
    coffee_name = sanitize_prompt_field(product.name, 200)
    roaster_name = sanitize_prompt_field(product.source, 200)
    url = sanitize_prompt_field(product.url, 400)
    return (
        "Research this coffee. Decide if it's worth trying. "
        "Score it (0-10) based on rarity (Geisha, etc.) and reputation. Be critical.\n"
        "These roasters are already high-end; only recommend coffees that are exceptional even by "
        "specialty standards. Treat routine offerings as not worth trying unless there is clear "
        "evidence of standout quality.\n"
        "High-end signals: rare varieties (e.g., Geisha, Sudan Rume, SL28), exceptional producers "
        "or farms, competition lots, Cup of Excellence winners, limited microlots, experimental "
        "processing (e.g., anaerobic, thermal shock), unusually high cupping scores, or strong "
        "peer reputation. Use clear evidence from the page text or grounded sources.\n"
        "If evidence is thin or ambiguous, set worth_trying=false and explain the uncertainty.\n"
        "Use the sanitized product page text and Google Search grounding.\n"
        f"{language_instruction(language)}\n\n"
        f"Coffee: {coffee_name}\n"
        f"Roaster: {roaster_name}\n"
        f"URL: {url}\n"
        f"{excerpt}\n"
    )


def build_batch_prompt(
    roaster_name: str,
    products: list[ProductCandidate],
    page_text_by_id: dict[str, str],
    max_chars: int,
    language: str,
) -> str:
    header = (
        f"You are evaluating coffees from {sanitize_prompt_field(roaster_name, 200)}. "
        "Decide if each is worth trying.\n"
        "These roasters are already high-end; only recommend coffees that are exceptional "
        "even by specialty standards. Treat routine offerings as not worth trying unless "
        "there is clear evidence of standout quality.\n"
        "High-end signals: rare varieties (e.g., Geisha, Sudan Rume, SL28), exceptional "
        "producers or farms, competition lots, Cup of Excellence winners, limited microlots, "
        "experimental processing (e.g., anaerobic, thermal shock), unusually high cupping "
        "scores, or strong peer reputation. Use clear evidence from the page text or "
        "grounded sources.\n"
        "Think carefully and generate a markdown recommendation report.\n"
        f"{language_instruction(language)}\n\n"
        "Products:\n"
    )
    sections: list[str] = [header]
    for product in products:
        body_text = ""
        if product.body_html:
            body_text = sanitize_html_to_text(
                product.body_html, max_chars, remove_boilerplate=False
            )
        body_text = sanitize_prompt_field(body_text, max_chars)
        page_text = sanitize_prompt_field(
            page_text_by_id.get(product.product_id, ""), max_chars
        )
        description_block = (
            f"  description (from body_html):\n  {body_text}"
            if body_text
            else "  description (from body_html): (none available)"
        )
        page_text_block = ""
        if page_text and page_text != body_text:
            page_text_block = (
                f"  page text (sanitized):\n  {page_text}"
            )
        sections.append(
            "\n".join(
                [
                    f"- product_id: {sanitize_prompt_field(product.product_id, 200)}",
                    f"  name: {sanitize_prompt_field(product.name, 200)}",
                    f"  url: {sanitize_prompt_field(product.url, 400)}",
                    *format_variant_lines(product.variants),
                    description_block,
                    page_text_block,
                    "",
                ]
            )
        )
    return "\n".join(sections)


def model_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()  # type: ignore[attr-defined]
    return model_cls.schema()  # type: ignore[no-any-return]


def validate_model(model_cls: type[BaseModel], data: Any) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)  # type: ignore[attr-defined]
    return model_cls.parse_obj(data)  # type: ignore[call-arg]


def validate_model_json(model_cls: type[BaseModel], text: str) -> BaseModel:
    if hasattr(model_cls, "model_validate_json"):
        return model_cls.model_validate_json(text)  # type: ignore[attr-defined]
    return model_cls.parse_raw(text)  # type: ignore[call-arg]


async def generate_content_async(client: genai.Client, **kwargs: Any) -> Any:
    aio = getattr(client, "aio", None)
    method = getattr(getattr(aio, "models", None), "generate_content", None)
    if method and inspect.iscoroutinefunction(method):
        return await method(**kwargs)
    return await asyncio.to_thread(client.models.generate_content, **kwargs)


async def count_tokens_async(client: genai.Client, **kwargs: Any) -> Any:
    aio = getattr(client, "aio", None)
    method = getattr(getattr(aio, "models", None), "count_tokens", None)
    if method and inspect.iscoroutinefunction(method):
        return await method(**kwargs)
    return await asyncio.to_thread(client.models.count_tokens, **kwargs)


def extract_total_tokens(count_response: Any) -> Optional[int]:
    if count_response is None:
        return None
    if isinstance(count_response, int):
        return count_response
    if isinstance(count_response, dict):
        for key in (
            "total_tokens",
            "totalTokens",
            "total_token_count",
            "totalTokenCount",
        ):
            value = count_response.get(key)
            if isinstance(value, int):
                return value
    for attr in (
        "total_tokens",
        "totalTokens",
        "total_token_count",
        "totalTokenCount",
    ):
        value = getattr(count_response, attr, None)
        if isinstance(value, int):
            return value
    return None


def extract_usage_metadata(response: Any) -> Optional[dict[str, int]]:
    usage = getattr(response, "usage_metadata", None) or getattr(
        response, "usageMetadata", None
    )
    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_token_count") or usage.get(
            "promptTokenCount"
        )
        output_tokens = usage.get("candidates_token_count") or usage.get(
            "candidatesTokenCount"
        )
        total_tokens = usage.get("total_token_count") or usage.get("totalTokenCount")
        data = {
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": output_tokens,
            "total_token_count": total_tokens,
        }
        return {k: v for k, v in data.items() if isinstance(v, int)} or None
    prompt_tokens = getattr(usage, "prompt_token_count", None) or getattr(
        usage, "promptTokenCount", None
    )
    output_tokens = getattr(usage, "candidates_token_count", None) or getattr(
        usage, "candidatesTokenCount", None
    )
    total_tokens = getattr(usage, "total_token_count", None) or getattr(
        usage, "totalTokenCount", None
    )
    data = {
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": output_tokens,
        "total_token_count": total_tokens,
    }
    return {k: v for k, v in data.items() if isinstance(v, int)} or None


def extract_grounding_metadata(response: Any) -> Optional[dict[str, Any]]:
    try:
        candidate = response.candidates[0]
        metadata = getattr(candidate, "grounding_metadata", None) or getattr(
            candidate, "groundingMetadata", None
        )
        if metadata is None:
            return None

        def get_value(obj: Any, *names: str) -> Any:
            for name in names:
                if isinstance(obj, dict) and name in obj:
                    return obj[name]
                value = getattr(obj, name, None)
                if value is not None:
                    return value
            return None

        raw_queries = get_value(metadata, "web_search_queries", "webSearchQueries") or []
        queries = [str(item) for item in raw_queries if str(item)]

        raw_chunks = get_value(metadata, "grounding_chunks", "groundingChunks") or []
        chunks: list[dict[str, Any]] = []
        for chunk in raw_chunks:
            web = get_value(chunk, "web")
            if web is None:
                continue
            uri = get_value(web, "uri")
            title = get_value(web, "title") or ""
            if uri or title:
                chunks.append({"web": {"uri": uri, "title": title}})

        raw_supports = get_value(metadata, "grounding_supports", "groundingSupports") or []
        supports: list[dict[str, Any]] = []
        for support in raw_supports:
            segment = get_value(support, "segment")
            segment_obj: dict[str, Any] = {}
            if segment is not None:
                start_index = get_value(segment, "start_index", "startIndex")
                end_index = get_value(segment, "end_index", "endIndex")
                text = get_value(segment, "text")
                if isinstance(start_index, int):
                    segment_obj["startIndex"] = start_index
                if isinstance(end_index, int):
                    segment_obj["endIndex"] = end_index
                if isinstance(text, str) and text:
                    segment_obj["text"] = text
            indices = get_value(
                support, "grounding_chunk_indices", "groundingChunkIndices"
            )
            chunk_indices = (
                [int(idx) for idx in indices if isinstance(idx, (int, float))]
                if isinstance(indices, list)
                else []
            )
            if segment_obj or chunk_indices:
                payload: dict[str, Any] = {}
                if segment_obj:
                    payload["segment"] = segment_obj
                if chunk_indices:
                    payload["groundingChunkIndices"] = chunk_indices
                supports.append(payload)

        result: dict[str, Any] = {}
        if queries:
            result["webSearchQueries"] = queries
        if chunks:
            result["groundingChunks"] = chunks
        if supports:
            result["groundingSupports"] = supports
        return result or None
    except Exception:
        return None


def format_grounding_metadata(grounding: dict[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for key in ("webSearchQueries", "groundingChunks", "groundingSupports"):
        value = grounding.get(key)
        if value:
            payload[key] = value
    if not payload:
        return ""
    return (
        "## Grounding metadata\n\n```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=True)
        + "\n```\n"
    )


async def evaluate_roaster_markdown(
    client: genai.Client,
    model: str,
    roaster_name: str,
    prompt: str,
    logger: logging.Logger,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        response_mime_type="text/plain",
        temperature=0.2,
        max_output_tokens=2048,
    )

    input_tokens: Optional[int] = None
    try:
        count_response = await count_tokens_async(
            client,
            model=model,
            contents=prompt,
        )
        input_tokens = extract_total_tokens(count_response)
        if input_tokens is not None:
            logger.info("Gemini input tokens for %s: %d", roaster_name, input_tokens)
    except Exception as exc:
        logger.warning("Gemini input token count failed for %s: %s", roaster_name, exc)

    try:
        response = await generate_content_async(
            client,
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        logger.exception("Gemini request failed for %s: %s", roaster_name, exc)
        return None, None

    text = (getattr(response, "text", None) or "").strip()
    usage = extract_usage_metadata(response)
    if usage:
        logger.info(
            "Gemini usage for %s: prompt_tokens=%s output_tokens=%s total_tokens=%s",
            roaster_name,
            usage.get("prompt_token_count"),
            usage.get("candidates_token_count"),
            usage.get("total_token_count"),
        )
    elif text:
        try:
            count_response = await count_tokens_async(
                client,
                model=model,
                contents=text,
            )
            output_tokens = extract_total_tokens(count_response)
            if output_tokens is not None:
                logger.info(
                    "Gemini output tokens for %s: %d", roaster_name, output_tokens
                )
        except Exception as exc:
            logger.warning(
                "Gemini output token count failed for %s: %s", roaster_name, exc
            )
    if text:
        logger.info("Gemini raw response for %s: %s", roaster_name, text)
        return text, extract_grounding_metadata(response)
    return None, extract_grounding_metadata(response)


def record_evaluation(
    path: Path,
    product: ProductCandidate,
    score: CoffeeEvaluation,
    grounding: Optional[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "product_id": product.product_id,
        "name": product.name,
        "url": product.url,
        "source": product.source,
        "score": score.score,
        "worth_trying": score.worth_trying,
        "summary": score.summary,
        "rationale": score.rationale,
        "rarity_signals": score.rarity_signals,
        "reputation_signals": score.reputation_signals,
        "confidence": score.confidence,
    }
    if grounding:
        record["grounding"] = grounding
    logging.getLogger("coffee_watch").info(
        "Evaluation record for %s: %s", product.url, json.dumps(record, ensure_ascii=True)
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def make_report_path(reports_dir: Path, roaster_name: str, run_id: str) -> Path:
    return report_file_path(reports_dir, roaster_name, run_id, "report", "md")


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


def append_report_entry(
    path: Path,
    product: ProductCandidate,
    score: CoffeeEvaluation,
    grounding: Optional[dict[str, Any]],
) -> None:
    sources = grounding.get("sources") if grounding else []
    lines = [
        f"### {product.name}",
        "",
        f"- Roaster: {product.source}",
        f"- URL: {product.url}",
        f"- Worth trying: {'Yes' if score.worth_trying else 'No'}",
        f"- Score: {score.score}/10",
        f"- Confidence: {score.confidence}",
        "",
        f"**Summary**: {score.summary}",
        "",
        f"**Rationale**: {score.rationale}",
    ]
    if score.rarity_signals:
        lines.append("")
        lines.append(f"**Rarity signals**: {', '.join(score.rarity_signals)}")
    if score.reputation_signals:
        lines.append("")
        lines.append(f"**Reputation signals**: {', '.join(score.reputation_signals)}")
    if sources:
        lines.append("")
        lines.append("**Grounding sources**:")
        for source in sources:
            title = source.get("title") or source.get("uri") or "Source"
            uri = source.get("uri") or ""
            if uri:
                lines.append(f"- {title} ({uri})")
    lines.append("\n---\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


async def run() -> int:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger = logging.getLogger("coffee_watch")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(
            "No GEMINI_API_KEY/GOOGLE_API_KEY found; relying on default SDK auth."
        )

    roasters = load_roasters(settings, logger)
    if not roasters:
        logger.error("No roasters configured; exiting.")
        return 1

    raw_language = settings.language
    language = normalize_language(raw_language)
    if raw_language.strip().lower() != language:
        logger.info("Normalized language setting from %s to %s", raw_language, language)

    denylist = load_denylist(settings.denylist_path)
    if denylist:
        logger.info("Loaded %d denylisted domains.", len(denylist))

    robots_cache: dict[str, RobotFileParser] = {}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    timeout = httpx.Timeout(settings.http_timeout_s)
    http_semaphore = asyncio.Semaphore(max(1, settings.http_concurrency))
    async with httpx.AsyncClient(
        http2=True,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=timeout,
    ) as http_client:
        genai_client = genai.Client(api_key=api_key) if api_key else genai.Client()
        for roaster in roasters:
            base_url = normalize_base_url(roaster.base_url)
            domain = urlsplit(base_url).netloc.lower()
            if domain and domain in denylist:
                logger.info("Skipping denylisted domain %s", domain)
                continue

            products = await fetch_products_for_roaster(
                http_client,
                roaster,
                settings,
                robots_cache,
                settings.reports_dir,
                run_id,
                http_semaphore,
                logger,
            )
            if not products:
                logger.info("No products parsed for %s", roaster.name)
                continue

            new_products = list(products)
            logger.info(
                "Found %d products for %s (%d to evaluate).",
                len(products),
                roaster.name,
                len(new_products),
            )

            if settings.fetch_only:
                logger.info(
                    "Fetch-only mode enabled; skipping product page fetches and Gemini."
                )
                if settings.save_report:
                    report_path = make_report_path(
                        settings.reports_dir, roaster.name, run_id
                    )
                    init_report(report_path, roaster.name, run_id)
                    with report_path.open("a", encoding="utf-8") as handle:
                        handle.write("\nFetch-only mode enabled; no LLM output.\n")
                        coffee_list = format_coffee_list(new_products)
                        if coffee_list:
                            handle.write("\n")
                            handle.write(coffee_list)
                continue

            page_headers = merge_headers(
                {"User-Agent": USER_AGENT},
                roaster.product_page_headers,
                logger,
                f"{roaster.name} product page",
            )
            products_needing_pages = [
                product for product in new_products if not product.body_html
            ]
            page_text_by_id = {product.product_id: "" for product in new_products}
            if products_needing_pages:
                page_tasks = [
                    fetch_product_page_text(
                        http_client,
                        product,
                        settings,
                        robots_cache,
                        logger,
                        page_headers,
                        http_semaphore,
                    )
                    for product in products_needing_pages
                ]
                page_texts = await asyncio.gather(*page_tasks)
                for product, text in zip(products_needing_pages, page_texts):
                    page_text_by_id[product.product_id] = text

            prompt = build_batch_prompt(
                roaster.name,
                new_products,
                page_text_by_id,
                settings.batch_page_text_max_chars,
                language,
            )
            if settings.save_prompt:
                prompt_path = save_prompt_text(
                    settings.reports_dir, run_id, roaster.name, prompt
                )
                logger.info("Saved Gemini prompt for %s to %s", roaster.name, prompt_path)

            if settings.skip_gemini:
                logger.info("Gemini skipped by configuration.")
                if settings.save_report:
                    report_path = make_report_path(
                        settings.reports_dir, roaster.name, run_id
                    )
                    init_report(report_path, roaster.name, run_id)
                    with report_path.open("a", encoding="utf-8") as handle:
                        handle.write("\nGemini skipped by configuration.\n")
                        coffee_list = format_coffee_list(new_products)
                        if coffee_list:
                            handle.write("\n")
                            handle.write(coffee_list)
                continue

            logger.info("Gemini prompt for %s:\n%s", roaster.name, prompt)
            markdown, grounding = await evaluate_roaster_markdown(
                genai_client,
                settings.model,
                roaster.name,
                prompt,
                logger,
            )
            if markdown is None:
                continue
            if grounding:
                logger.info(
                    "Gemini grounding metadata for %s: %s",
                    roaster.name,
                    json.dumps(grounding, ensure_ascii=True),
                )

            if settings.save_report:
                report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
                init_report(report_path, roaster.name, run_id)
                with report_path.open("a", encoding="utf-8") as handle:
                    handle.write(markdown)
                    handle.write("\n")
                    grounding_block = format_grounding_metadata(grounding or {})
                    if grounding_block:
                        handle.write("\n")
                        handle.write(grounding_block)
                    coffee_list = format_coffee_list(new_products)
                    if coffee_list:
                        handle.write("\n")
                        handle.write(coffee_list)

    logger.info("Run complete.")
    return 0


def main() -> None:
    try:
        exit_code = asyncio.run(run())
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
