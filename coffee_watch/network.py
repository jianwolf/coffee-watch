from __future__ import annotations

import asyncio
import json
import logging
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from .config import Settings
from .constants import USER_AGENT
from .models import PaginationConfig, ProductCandidate, ProductFieldConfig, RoasterSource
from .parsing import parse_products_json, parse_products_response
from .reporting import log_products_json_snippet, save_products_json, save_products_json_pretty
from .text_utils import extract_product_jsonld_text, sanitize_html_to_text
from .url_utils import build_url_with_params, canonicalize_url, matches_patterns


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
    jitter_multiplier: float = 1.0,
) -> Optional[httpx.Response]:
    retry_statuses = {429, 500, 502, 503, 504}
    max_retries = 2

    def _retry_delay(attempt: int, response: Optional[httpx.Response]) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return max(float(retry_after), settings.jitter_min_s * jitter_multiplier)
                except ValueError:
                    pass
        base = max(0.5, settings.jitter_min_s * jitter_multiplier)
        cap = max(base, settings.jitter_max_s * jitter_multiplier * 3)
        return min(cap, base * (2**attempt))

    async def _run() -> Optional[httpx.Response]:
        await jitter_sleep(
            settings.jitter_min_s * jitter_multiplier,
            settings.jitter_max_s * jitter_multiplier,
        )
        logger.info("HTTP GET %s", url)
        try:
            response = await client.get(url, headers=headers)
            logger.info("HTTP %s %s", response.status_code, url)
            return response
        except httpx.RequestError as exc:
            logger.warning("HTTP request failed for %s: %s", url, exc)
            return None

    async def _attempt() -> Optional[httpx.Response]:
        if semaphore is None:
            return await _run()
        async with semaphore:
            return await _run()

    response: Optional[httpx.Response] = None
    for attempt in range(max_retries + 1):
        response = await _attempt()
        if response is not None and response.status_code < 400:
            return response
        status = response.status_code if response is not None else None
        should_retry = response is None or status in retry_statuses
        if not should_retry or attempt == max_retries:
            return response
        delay = _retry_delay(attempt, response)
        sleep_for = random.uniform(0, delay)
        logger.warning(
            "Retrying %s in %.2fs after status %s",
            url,
            sleep_for,
            status,
        )
        await asyncio.sleep(sleep_for)
    return response


async def robots_allows(
    client: httpx.AsyncClient,
    products_url: str,
    settings: Settings,
    cache: dict[str, RobotFileParser],
    logger: logging.Logger,
    jitter_multiplier: float = 1.0,
) -> bool:
    parsed = urlsplit(products_url)
    if not parsed.scheme or not parsed.netloc:
        return False
    cache_key = f"{parsed.scheme}://{parsed.netloc}"
    if cache_key in cache:
        return cache[cache_key].can_fetch(USER_AGENT, products_url)

    robots_url = urljoin(cache_key, "/robots.txt")
    response = await fetch_text_with_jitter(
        client,
        robots_url,
        settings,
        logger,
        jitter_multiplier=jitter_multiplier,
    )
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


async def fetch_products_for_roaster(
    http_client: httpx.AsyncClient,
    roaster: RoasterSource,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    assets_dir: Path,
    run_id: str,
    semaphore: Optional[asyncio.Semaphore],
    logger: logging.Logger,
) -> list[ProductCandidate]:
    max_products = roaster.max_products or settings.max_products_per_source
    pagination = roaster.pagination or PaginationConfig(max_pages=1)
    stop_on_empty = pagination.stop_on_empty
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
            http_client,
            products_url,
            settings,
            robots_cache,
            logger,
            jitter_multiplier=roaster.jitter_multiplier,
        )
        if not allowed:
            logger.warning(
                "Robots.txt disallows %s for %s; skipping.",
                products_url,
                roaster.name,
            )
            if stop_on_empty:
                break
            continue

        response = await fetch_text_with_jitter(
            http_client,
            products_url,
            settings,
            logger,
            headers=headers,
            semaphore=semaphore,
            jitter_multiplier=roaster.jitter_multiplier,
        )
        if response is None:
            logger.warning("Request failed for %s", products_url)
            if stop_on_empty:
                break
            continue
        if response.status_code >= 400:
            logger.warning(
                "Non-200 response %s for %s", response.status_code, products_url
            )
            if stop_on_empty:
                break
            continue

        content_type = response.headers.get("content-type", "")
        remaining = max_products - len(products)
        page_products: list[ProductCandidate] = []
        if roaster.products_type == "json" or "json" in content_type.lower():
            json_text = response.text
            raw_path = save_products_json(
                assets_dir, run_id, roaster, page_index, json_text
            )
            logger.info("Saved raw products JSON for %s to %s", roaster.name, raw_path)
            if settings.save_raw_products_json:
                report_raw_path = save_products_json(
                    settings.reports_dir, run_id, roaster, page_index, json_text
                )
                logger.info(
                    "Saved raw products JSON copy for %s to %s",
                    roaster.name,
                    report_raw_path,
                )
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON for %s: %s", roaster.name, exc)
                log_products_json_snippet(
                    logger,
                    roaster,
                    products_url,
                    json_text,
                    settings.log_json_max_chars,
                )
            else:
                pretty_path = save_products_json_pretty(
                    assets_dir, run_id, roaster, page_index, data
                )
                if pretty_path:
                    logger.info(
                        "Saved pretty products JSON for %s to %s",
                        roaster.name,
                        pretty_path,
                    )
                    if settings.save_pretty_products_json:
                        report_pretty_path = save_products_json_pretty(
                            settings.reports_dir, run_id, roaster, page_index, data
                        )
                        if report_pretty_path:
                            logger.info(
                                "Saved pretty products JSON copy for %s to %s",
                                roaster.name,
                                report_pretty_path,
                            )
                        else:
                            logger.warning(
                                "Failed to serialize pretty products JSON copy for %s",
                                roaster.name,
                            )
                else:
                    logger.warning(
                        "Failed to serialize pretty products JSON for %s",
                        roaster.name,
                    )
                product_fields = roaster.product_fields or ProductFieldConfig()
                page_products = parse_products_json(
                    data,
                    roaster.base_url,
                    roaster,
                    remaining,
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
                remaining,
                logger,
            )
        if not page_products and stop_on_empty:
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
    jitter_multiplier: float = 1.0,
) -> tuple[str, str]:
    product_allowed = await robots_allows(
        http_client,
        product.url,
        settings,
        robots_cache,
        logger,
        jitter_multiplier=jitter_multiplier,
    )
    if not product_allowed:
        logger.warning(
            "Robots.txt disallows product page %s; skipping page fetch.",
            product.url,
        )
        return "", ""

    page_response = await fetch_text_with_jitter(
        http_client,
        product.url,
        settings,
        logger,
        headers=headers,
        semaphore=semaphore,
        jitter_multiplier=jitter_multiplier,
    )
    if page_response is None:
        logger.warning("Request failed for product page %s", product.url)
        return "", ""
    if page_response.status_code >= 400:
        logger.warning(
            "Non-200 response %s for product page %s",
            page_response.status_code,
            product.url,
        )
        return "", ""
    html = page_response.text
    page_text = extract_product_jsonld_text(html, settings.page_text_max_chars)
    if not page_text:
        page_text = sanitize_html_to_text(html, settings.page_text_max_chars)
    logger.info(
        "Sanitized %s chars of page text for %s", len(page_text), product.url
    )
    http_last_modified = page_response.headers.get("last-modified", "")
    return page_text, http_last_modified


def _strip_xml_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_xml_text(node: Optional[ET.Element], tag: str) -> str:
    if node is None:
        return ""
    child = node.find(f".//{{*}}{tag}")
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _normalize_sitemap_loc(loc: str) -> list[str]:
    loc = loc.strip()
    if not loc:
        return []
    canonical = canonicalize_url(loc)
    if not canonical:
        return []
    urls = [canonical]
    parts = urlsplit(canonical)
    if parts.path.endswith("/") and parts.path != "/":
        trimmed = canonical.rstrip("/")
        if trimmed not in urls:
            urls.append(trimmed)
    return urls


def _parse_sitemap_xml(xml_text: str) -> tuple[dict[str, str], list[str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}, []

    tag = _strip_xml_namespace(root.tag).lower()
    if tag == "urlset":
        lastmods: dict[str, str] = {}
        for url_node in root.findall(".//{*}url"):
            loc = _find_xml_text(url_node, "loc")
            lastmod = _find_xml_text(url_node, "lastmod")
            if not loc or not lastmod:
                continue
            for normalized in _normalize_sitemap_loc(loc):
                lastmods[normalized] = lastmod
        return lastmods, []

    if tag == "sitemapindex":
        sitemaps: list[str] = []
        for sitemap_node in root.findall(".//{*}sitemap"):
            loc = _find_xml_text(sitemap_node, "loc")
            if loc:
                sitemaps.append(loc)
        return {}, sitemaps

    return {}, []


async def fetch_wix_product_sitemap_lastmods(
    http_client: httpx.AsyncClient,
    roaster: RoasterSource,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    logger: logging.Logger,
    headers: dict[str, str],
    semaphore: Optional[asyncio.Semaphore],
    jitter_multiplier: float = 1.0,
) -> dict[str, str]:
    base = roaster.base_url if roaster.base_url.endswith("/") else f"{roaster.base_url}/"
    store_products_sitemap = urljoin(base, "store-products-sitemap.xml")
    index_sitemap = urljoin(base, "sitemap.xml")

    def _filter_lastmods(lastmods: dict[str, str]) -> dict[str, str]:
        if not lastmods:
            return {}
        include = roaster.product_link_patterns
        exclude = roaster.product_link_exclude_patterns
        if not include and not exclude:
            return lastmods
        return {
            url: lastmod
            for url, lastmod in lastmods.items()
            if matches_patterns(url, include, exclude)
        }

    async def _fetch(url: str) -> tuple[dict[str, str], list[str]]:
        allowed = await robots_allows(
            http_client,
            url,
            settings,
            robots_cache,
            logger,
            jitter_multiplier=jitter_multiplier,
        )
        if not allowed:
            logger.warning("Robots.txt disallows sitemap %s; skipping.", url)
            return {}, []
        response = await fetch_text_with_jitter(
            http_client,
            url,
            settings,
            logger,
            headers=headers,
            semaphore=semaphore,
            jitter_multiplier=jitter_multiplier,
        )
        if response is None:
            logger.warning("Request failed for sitemap %s", url)
            return {}, []
        if response.status_code >= 400:
            logger.info("Sitemap not available (%s): %s", response.status_code, url)
            return {}, []
        return _parse_sitemap_xml(response.text)

    async def _collect(url: str) -> dict[str, str]:
        queue = [url]
        seen: set[str] = set()
        combined: dict[str, str] = {}
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            lastmods, children = await _fetch(current)
            combined.update(_filter_lastmods(lastmods))
            for child in children:
                if "store-products-sitemap" in child.lower():
                    queue.append(child)
            if len(seen) >= 8:
                break
        return combined

    lastmods = await _collect(store_products_sitemap)
    if lastmods:
        logger.info(
            "Loaded %d Wix sitemap lastmod entries from %s",
            len(lastmods),
            store_products_sitemap,
        )
        return lastmods

    lastmods = await _collect(index_sitemap)
    if lastmods:
        logger.info(
            "Loaded %d Wix sitemap lastmod entries from %s",
            len(lastmods),
            index_sitemap,
        )
    return lastmods


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
