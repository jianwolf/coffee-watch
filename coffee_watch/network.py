from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from .config import Settings
from .constants import USER_AGENT
from .models import PaginationConfig, ProductCandidate, ProductFieldConfig, RoasterSource
from .parsing import parse_products_json, parse_products_response
from .reporting import log_products_json, save_products_json, save_products_json_pretty
from .text_utils import sanitize_html_to_text
from .url_utils import build_url_with_params


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
            http_client, products_url, settings, robots_cache, logger
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
            http_client, products_url, settings, logger, headers=headers, semaphore=semaphore
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
