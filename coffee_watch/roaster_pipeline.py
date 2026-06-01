"""Per-roaster scrape, classification, and catalog writing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from .catalog import build_roaster_catalog, catalog_product_from_candidate
from .classify import (
    NEW_PRODUCTS_WINDOW_DAYS,
    classify_new_products,
    log_new_products_summary,
    resolve_update_date,
)
from .constants import USER_AGENT
from .context import RunContext
from .models import ProductCandidate, RoasterRunStatus, RoasterSource
from .network import (
    fetch_product_page_text,
    fetch_products_for_roaster,
    fetch_wix_product_sitemap_lastmods,
    merge_headers,
)
from .report_status import (
    STATUS_EMPTY,
    STATUS_FAILURE,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    utc_now_iso,
    write_status_sidecar,
)
from .reporting import make_roaster_catalog_path, save_json
from .text_utils import sanitize_html_to_text, trim_text_at_phrases
from .url_utils import normalize_base_url


@dataclass(frozen=True)
class RoasterScrapeResult:
    catalog_path: Optional[Path]
    catalog: dict[str, Any]
    status: RoasterRunStatus


async def process_roaster(
    ctx: RunContext,
    roaster: RoasterSource,
) -> RoasterScrapeResult:
    try:
        return await _process_roaster_inner(ctx, roaster)
    except Exception as exc:
        ctx.logger.exception(
            "Roaster processing failed for %s: %s", roaster.name, exc
        )
        path = make_roaster_catalog_path(
            ctx.settings.reports_dir, roaster.name, ctx.run_id
        )
        status = RoasterRunStatus(
            roaster=roaster.name,
            run_id=ctx.run_id,
            status=STATUS_FAILURE,
            attempts=1,
            catalog_path=path.name,
            products_found=0,
            new_products=0,
            fetched_pages=0,
            platform=roaster.platform,
            completed_at=utc_now_iso(),
            note=f"unhandled exception: {exc.__class__.__name__}",
        )
        catalog = build_roaster_catalog(
            roaster=roaster,
            run_id=ctx.run_id,
            generated_at=utc_now_iso(),
            status=status,
            products=[],
            errors=[str(exc)],
        )
        save_json(path, catalog)
        write_status_sidecar(ctx.settings.reports_dir, status, ctx.logger)
        return RoasterScrapeResult(path, catalog, status)


def _description_for_product(
    product: ProductCandidate,
    page_text_by_id: dict[str, str],
    max_chars: int,
) -> str:
    body_text = ""
    if product.body_html:
        body_text = sanitize_html_to_text(
            product.body_html,
            max_chars,
            remove_boilerplate=False,
        )
    return page_text_by_id.get(product.product_id, "") or body_text


def _build_descriptions_by_url(
    products: list[ProductCandidate],
    page_text_by_id: dict[str, str],
    max_chars: int,
) -> dict[str, str]:
    return {
        product.url: _description_for_product(product, page_text_by_id, max_chars)
        for product in products
    }


async def _fetch_missing_product_pages(
    ctx: RunContext,
    roaster: RoasterSource,
    products: list[ProductCandidate],
    page_text_by_id: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]], int]:
    if not ctx.settings.fetch_product_pages:
        ctx.logger.info(
            "Product page fetching disabled for %s; using catalog text only.",
            roaster.name,
        )
        return {}, {}, 0
    if roaster.platform == "shopify":
        ctx.logger.info(
            "Skipping product page fetches for %s (platform shopify).", roaster.name
        )
        return {}, {}, 0

    page_headers = merge_headers(
        {"User-Agent": USER_AGENT},
        roaster.product_page_headers,
        ctx.logger,
        f"{roaster.name} product page",
    )
    products_needing_pages = [product for product in products if not product.body_html]
    page_fetch_count = len(products_needing_pages)
    http_last_modified_by_url: dict[str, str] = {}
    errors_by_url: dict[str, list[str]] = {}
    if not products_needing_pages:
        return http_last_modified_by_url, errors_by_url, 0

    page_tasks = [
        fetch_product_page_text(
            ctx.http_client,
            product,
            ctx.settings,
            ctx.robots_cache,
            ctx.logger,
            page_headers,
            ctx.limiter,
            jitter_multiplier=roaster.jitter_multiplier,
        )
        for product in products_needing_pages
    ]
    page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
    for product, result in zip(products_needing_pages, page_results):
        if isinstance(result, BaseException):
            message = f"product page fetch raised: {result}"
            ctx.logger.warning("Page text fetch raised for %s: %s", product.url, result)
            errors_by_url.setdefault(product.url, []).append(message)
            continue
        text, last_modified = result
        page_text_by_id[product.product_id] = trim_text_at_phrases(
            text, roaster.page_text_stop_phrases
        )
        if last_modified:
            http_last_modified_by_url[product.url] = last_modified
    return http_last_modified_by_url, errors_by_url, page_fetch_count


async def _load_wix_lastmods(
    ctx: RunContext,
    roaster: RoasterSource,
) -> dict[str, str]:
    if roaster.platform != "wix":
        return {}
    sitemap_headers = merge_headers(
        {"User-Agent": USER_AGENT},
        roaster.product_page_headers,
        ctx.logger,
        f"{roaster.name} sitemap",
    )
    return await fetch_wix_product_sitemap_lastmods(
        ctx.http_client,
        roaster,
        ctx.settings,
        ctx.robots_cache,
        ctx.logger,
        sitemap_headers,
        ctx.limiter,
        jitter_multiplier=roaster.jitter_multiplier,
    )


def _write_catalog_and_status(
    ctx: RunContext,
    roaster: RoasterSource,
    status: RoasterRunStatus,
    products: list[dict[str, Any]],
    errors: list[str],
    path: Path,
) -> RoasterScrapeResult:
    catalog = build_roaster_catalog(
        roaster=roaster,
        run_id=ctx.run_id,
        generated_at=utc_now_iso(),
        status=status,
        products=products,
        errors=errors,
    )
    save_json(path, catalog)
    write_status_sidecar(ctx.settings.reports_dir, status, ctx.logger)
    ctx.logger.info("Saved roaster catalog to %s", path)
    return RoasterScrapeResult(path, catalog, status)


async def _process_roaster_inner(
    ctx: RunContext,
    roaster: RoasterSource,
) -> RoasterScrapeResult:
    settings = ctx.settings
    log = ctx.logger
    run_id = ctx.run_id
    path = make_roaster_catalog_path(settings.reports_dir, roaster.name, run_id)

    base_url = normalize_base_url(roaster.base_url)
    domain = urlsplit(base_url).netloc.lower()
    if domain and domain in ctx.denylist:
        log.info("Skipping denylisted domain %s", domain)
        status = RoasterRunStatus(
            roaster=roaster.name,
            run_id=run_id,
            status=STATUS_SKIPPED,
            attempts=0,
            catalog_path=path.name,
            products_found=0,
            new_products=0,
            fetched_pages=0,
            platform=roaster.platform,
            completed_at=utc_now_iso(),
            note="denylisted domain",
        )
        return _write_catalog_and_status(
            ctx,
            roaster,
            status,
            [],
            ["denylisted domain"],
            path,
        )

    products = await fetch_products_for_roaster(
        ctx.http_client,
        roaster,
        settings,
        ctx.robots_cache,
        ctx.assets_dir,
        run_id,
        ctx.limiter,
        log,
    )
    if not products:
        log.info("No products parsed for %s", roaster.name)
        status = RoasterRunStatus(
            roaster=roaster.name,
            run_id=run_id,
            status=STATUS_EMPTY,
            attempts=1,
            catalog_path=path.name,
            products_found=0,
            new_products=0,
            fetched_pages=0,
            platform=roaster.platform,
            completed_at=utc_now_iso(),
            note="no products parsed",
        )
        return _write_catalog_and_status(ctx, roaster, status, [], [], path)

    run_day = datetime.strptime(run_id, "%Y%m%d").date()
    log.info("Found %d products for %s.", len(products), roaster.name)

    page_text_by_id = {product.product_id: "" for product in products}
    wix_lastmod_by_url = await _load_wix_lastmods(ctx, roaster)
    (
        http_last_modified_by_url,
        errors_by_url,
        page_fetch_count,
    ) = await _fetch_missing_product_pages(ctx, roaster, products, page_text_by_id)

    descriptions_by_url = _build_descriptions_by_url(
        products,
        page_text_by_id,
        settings.page_text_max_chars,
    )
    new_urls, by_source, undated, outside_window = classify_new_products(
        products,
        run_day,
        ctx.seen_products,
        descriptions_by_url,
        http_last_modified_by_url,
        wix_lastmod_by_url,
        roaster.platform,
        window_days=NEW_PRODUCTS_WINDOW_DAYS,
        persist_seen=True,
    )
    first_seen_by_url = ctx.seen_products.first_seen_for_urls(
        [product.url for product in products]
    )

    catalog_products: list[dict[str, Any]] = []
    for product in products:
        first_seen_at = first_seen_by_url.get(product.url, "")
        update_date, date_source = resolve_update_date(
            product,
            http_last_modified_by_url.get(product.url, ""),
            wix_lastmod_by_url.get(product.url, ""),
            first_seen_at,
        )
        catalog_products.append(
            catalog_product_from_candidate(
                product=product,
                roaster=roaster,
                raw_product_text=descriptions_by_url.get(product.url, ""),
                first_seen_at=first_seen_at,
                is_new=product.url in new_urls,
                date_source=date_source,
                update_date=update_date,
                http_last_modified=http_last_modified_by_url.get(product.url, ""),
                wix_lastmod=wix_lastmod_by_url.get(product.url, ""),
                errors=errors_by_url.get(product.url, []),
            )
        )

    log_new_products_summary(
        log,
        roaster.name,
        run_id,
        new_urls,
        by_source,
        undated,
        outside_window,
        page_fetch_count,
    )
    status = RoasterRunStatus(
        roaster=roaster.name,
        run_id=run_id,
        status=STATUS_SUCCESS,
        attempts=1,
        catalog_path=path.name,
        products_found=len(products),
        new_products=len(new_urls),
        fetched_pages=page_fetch_count,
        platform=roaster.platform,
        classified_by_source=dict(by_source),
        undated=undated,
        outside_window=outside_window,
        completed_at=utc_now_iso(),
    )
    return _write_catalog_and_status(
        ctx,
        roaster,
        status,
        catalog_products,
        [],
        path,
    )
