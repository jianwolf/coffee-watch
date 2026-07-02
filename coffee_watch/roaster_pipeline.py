"""Per-roaster scrape, classification, and catalog writing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .catalog import (
    STOREFRONT_UNAVAILABLE_STATUS,
    build_roaster_catalog,
    catalog_product_from_candidate,
)
from .classify import (
    classify_new_products,
    log_new_products_summary,
    resolve_update_date,
)
from .constants import USER_AGENT
from .context import RunContext
from .models import ProductCandidate, RoasterRunStatus, RoasterSource, VariantInfo
from .network import (
    fetch_product_page_text,
    fetch_products_for_roaster,
    fetch_text_with_jitter,
    fetch_wix_product_sitemap_lastmods,
    merge_headers,
    robots_allows,
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
from .text_utils import (
    extract_product_jsonld_text,
    extract_product_page_price,
    extract_product_page_size_labels,
    extract_visible_text,
    finalize_visible_text,
    grams_from_size_label,
    sanitize_html_to_text,
    trim_text_at_phrases,
)
from .url_utils import normalize_base_url

STOREFRONT_UNAVAILABLE_STATUS_CODES = {401, 403, 404, 410}

# Recorded per product when the host's 429 gate cut the storefront pass short.
GATED_SKIP_ERROR = "storefront page skipped: host bot-gated by consecutive 429s"


@dataclass(frozen=True)
class RoasterScrapeResult:
    catalog_path: Path | None
    catalog: dict[str, Any]
    status: RoasterRunStatus


@dataclass(frozen=True)
class StorefrontPageEvidence:
    """Per-URL evidence from the verify_variant_pages storefront pass."""

    titles_by_url: dict[str, tuple[str, ...]] = field(default_factory=dict)
    price_by_url: dict[str, str] = field(default_factory=dict)
    status_by_url: dict[str, str] = field(default_factory=dict)
    page_text_by_url: dict[str, str] = field(default_factory=dict)
    last_modified_by_url: dict[str, str] = field(default_factory=dict)
    errors_by_url: dict[str, list[str]] = field(default_factory=dict)
    fetched_pages: int = 0
    gated_skips: int = 0


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


async def _prewarm_robots_cache(
    ctx: RunContext,
    roaster: RoasterSource,
    products: list[ProductCandidate],
) -> None:
    """Fetch robots.txt once per distinct host before fanning out page fetches.

    The concurrent per-product tasks all check robots first; without this
    serial warm-up, every task on a cold host would issue its own robots.txt
    request.
    """
    seen_robots_hosts: set[str] = set()
    for product in products:
        parsed = urlsplit(product.url)
        if not parsed.scheme or not parsed.netloc:
            continue
        cache_key = f"{parsed.scheme}://{parsed.netloc}"
        if cache_key in seen_robots_hosts or cache_key in ctx.robots_cache:
            continue
        seen_robots_hosts.add(cache_key)
        await robots_allows(
            ctx.http_client,
            product.url,
            ctx.settings,
            ctx.robots_cache,
            ctx.logger,
            limiter=ctx.limiter,
            jitter_multiplier=roaster.jitter_multiplier,
            denylist=ctx.denylist,
        )


async def _fetch_missing_product_pages(
    ctx: RunContext,
    roaster: RoasterSource,
    products: list[ProductCandidate],
    page_text_by_id: dict[str, str],
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
    dict[str, str],
    dict[str, tuple[str, ...]],
    int,
]:
    if not ctx.settings.fetch_product_pages:
        ctx.logger.info(
            "Product page fetching disabled for %s; using catalog text only.",
            roaster.name,
        )
        return {}, {}, {}, {}, 0
    if roaster.platform == "shopify":
        ctx.logger.info(
            "Skipping product page fetches for %s (platform shopify).", roaster.name
        )
        return {}, {}, {}, {}, 0

    page_headers = merge_headers(
        {"User-Agent": USER_AGENT},
        roaster.product_page_headers,
        ctx.logger,
        f"{roaster.name} product page",
    )
    products_needing_pages = [product for product in products if not product.body_html]
    page_fetch_count = len(products_needing_pages)
    http_last_modified_by_url: dict[str, str] = {}
    page_price_by_url: dict[str, str] = {}
    page_visible_titles_by_url: dict[str, tuple[str, ...]] = {}
    errors_by_url: dict[str, list[str]] = {}
    if not products_needing_pages:
        return http_last_modified_by_url, errors_by_url, {}, {}, 0

    await _prewarm_robots_cache(ctx, roaster, products_needing_pages)
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
            denylist=ctx.denylist,
            gate=ctx.page_gate,
        )
        for product in products_needing_pages
    ]
    page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
    for product, result in zip(products_needing_pages, page_results, strict=True):
        if isinstance(result, BaseException):
            message = f"product page fetch raised: {result}"
            ctx.logger.warning("Page text fetch raised for %s: %s", product.url, result)
            errors_by_url.setdefault(product.url, []).append(message)
            continue
        text, last_modified, page_price, visible_titles = result
        page_text_by_id[product.product_id] = trim_text_at_phrases(
            text, roaster.page_text_stop_phrases
        )
        if last_modified:
            http_last_modified_by_url[product.url] = last_modified
        if page_price:
            page_price_by_url[product.url] = page_price
        if visible_titles:
            page_visible_titles_by_url[product.url] = visible_titles
    return (
        http_last_modified_by_url,
        errors_by_url,
        page_price_by_url,
        page_visible_titles_by_url,
        page_fetch_count,
    )


async def _fetch_storefront_product_pages(
    ctx: RunContext,
    roaster: RoasterSource,
    products: list[ProductCandidate],
) -> StorefrontPageEvidence:
    if not ctx.settings.fetch_product_pages or not roaster.verify_variant_pages:
        return StorefrontPageEvidence()

    products_needing_pages = [product for product in products if product.url]
    if not products_needing_pages:
        return StorefrontPageEvidence()

    page_headers = merge_headers(
        {"User-Agent": USER_AGENT},
        roaster.product_page_headers,
        ctx.logger,
        f"{roaster.name} variant page",
    )

    await _prewarm_robots_cache(ctx, roaster, products_needing_pages)

    async def fetch(
        product: ProductCandidate,
    ) -> tuple[str, tuple[str, ...], str, str, str, str, str]:
        allowed = await robots_allows(
            ctx.http_client,
            product.url,
            ctx.settings,
            ctx.robots_cache,
            ctx.logger,
            limiter=ctx.limiter,
            jitter_multiplier=roaster.jitter_multiplier,
            denylist=ctx.denylist,
        )
        if not allowed:
            return product.url, (), "", "", "", "", "robots.txt disallows product page"
        response = await fetch_text_with_jitter(
            ctx.http_client,
            product.url,
            ctx.settings,
            ctx.logger,
            headers=page_headers,
            limiter=ctx.limiter,
            jitter_multiplier=roaster.jitter_multiplier,
            denylist=ctx.denylist,
            gate=ctx.page_gate,
        )
        if response is None:
            if ctx.page_gate.is_gated(product.url):
                return product.url, (), "", "", "", "", GATED_SKIP_ERROR
            return product.url, (), "", "", "", "", "storefront page request failed"
        if response.status_code in STOREFRONT_UNAVAILABLE_STATUS_CODES:
            return (
                product.url,
                (),
                "",
                STOREFRONT_UNAVAILABLE_STATUS,
                "",
                "",
                (
                    f"storefront page returned status {response.status_code}; "
                    "treating product as not publicly buyable"
                ),
            )
        if response.status_code >= 400:
            return (
                product.url,
                (),
                "",
                "",
                "",
                "",
                f"storefront page returned status {response.status_code}",
            )
        html = response.text
        # Parse the HTML once and share the visible text between the
        # description fallback, price, and size-label extraction.
        raw_visible = extract_visible_text(html)
        plain_text = finalize_visible_text(raw_visible, 0, remove_boilerplate=False)
        page_text = extract_product_jsonld_text(
            html,
            ctx.settings.page_text_max_chars,
            page_url=product.url,
        )
        if not page_text:
            page_text = finalize_visible_text(
                raw_visible,
                ctx.settings.page_text_max_chars,
            )
        return (
            product.url,
            extract_product_page_size_labels(html, plain_text=plain_text),
            extract_product_page_price(html, plain_text=plain_text),
            "",
            trim_text_at_phrases(page_text, roaster.page_text_stop_phrases),
            response.headers.get("last-modified", ""),
            "",
        )

    evidence = StorefrontPageEvidence(
        fetched_pages=len(products_needing_pages),
    )
    gated_skips = 0
    results = await asyncio.gather(
        *(fetch(product) for product in products_needing_pages),
        return_exceptions=True,
    )
    for product, result in zip(products_needing_pages, results, strict=True):
        if isinstance(result, BaseException):
            evidence.errors_by_url.setdefault(product.url, []).append(
                f"storefront page fetch raised: {result}"
            )
            continue
        (
            url,
            titles,
            price,
            storefront_status,
            page_text,
            http_last_modified,
            error,
        ) = result
        if titles:
            evidence.titles_by_url[url] = titles
        if price:
            evidence.price_by_url[url] = price
        if storefront_status:
            evidence.status_by_url[url] = storefront_status
        if page_text:
            evidence.page_text_by_url[url] = page_text
        if http_last_modified:
            evidence.last_modified_by_url[url] = http_last_modified
        if error:
            evidence.errors_by_url.setdefault(url, []).append(error)
            if error == GATED_SKIP_ERROR:
                gated_skips += 1
    if not gated_skips:
        return evidence
    # Skipped pages were never requested; keep fetched_pages an honest count.
    return replace(
        evidence,
        fetched_pages=len(products_needing_pages) - gated_skips,
        gated_skips=gated_skips,
    )


def _variant_price_from_page_price(price: str) -> str:
    # Variant prices stay machine-parseable: no "$", no thousands commas.
    return price.removeprefix("$").replace(",", "").strip()


def _with_storefront_purchase_details(
    product: ProductCandidate,
    visible_titles: tuple[str, ...],
    page_price: str,
    storefront_status: str,
) -> ProductCandidate:
    variants = product.variants
    list_price = product.list_price
    available_price = page_price or list_price
    if not variants and visible_titles and available_price:
        title = visible_titles[0]
        variants = (
            VariantInfo(
                title=title,
                price=_variant_price_from_page_price(available_price),
                grams=grams_from_size_label(title),
                available=True,
            ),
        )
        list_price = ""
    elif page_price and not list_price:
        list_price = page_price

    if (
        not visible_titles
        and not storefront_status
        and variants == product.variants
        and list_price == product.list_price
    ):
        return product

    return replace(
        product,
        visible_variant_titles=visible_titles,
        storefront_status=storefront_status,
        variants=variants,
        list_price=list_price,
    )


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
        denylist=ctx.denylist,
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
        denylist=ctx.denylist,
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
        page_price_by_url,
        page_visible_titles_by_url,
        page_fetch_count,
    ) = await _fetch_missing_product_pages(ctx, roaster, products, page_text_by_id)
    storefront = await _fetch_storefront_product_pages(ctx, roaster, products)
    for url, errors in storefront.errors_by_url.items():
        errors_by_url.setdefault(url, []).extend(errors)
    http_last_modified_by_url.update(
        {
            url: value
            for url, value in storefront.last_modified_by_url.items()
            if url not in http_last_modified_by_url
        }
    )
    for product in products:
        page_text = storefront.page_text_by_url.get(product.url, "")
        if page_text:
            page_text_by_id[product.product_id] = page_text

    descriptions_by_url = _build_descriptions_by_url(
        products,
        page_text_by_id,
        settings.page_text_max_chars,
    )
    # SQLite access is synchronous; run classification in a worker thread so
    # DB reads/writes don't stall the other roasters' event-loop tasks.
    classification = await asyncio.to_thread(
        classify_new_products,
        products,
        run_day,
        ctx.seen_products,
        descriptions_by_url,
        http_last_modified_by_url,
        wix_lastmod_by_url,
        roaster.platform,
        window_days=settings.new_window_days,
        persist_seen=True,
    )
    new_urls = classification.new_urls
    first_seen_by_url = classification.first_seen_by_url

    catalog_products: list[dict[str, Any]] = []
    for product in products:
        visible_titles = (
            storefront.titles_by_url.get(product.url)
            or page_visible_titles_by_url.get(product.url, ())
        )
        page_price = (
            storefront.price_by_url.get(product.url)
            or page_price_by_url.get(product.url, "")
        )
        storefront_status = storefront.status_by_url.get(product.url, "")
        if visible_titles or page_price or storefront_status:
            product = _with_storefront_purchase_details(
                product,
                visible_titles,
                page_price,
                storefront_status,
            )
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
        classification.by_source,
        classification.undated,
        classification.outside_window,
        page_fetch_count,
        window_days=settings.new_window_days,
    )
    note = ""
    if storefront.gated_skips:
        note = (
            f"storefront pages bot-gated; skipped {storefront.gated_skips} of "
            f"{storefront.fetched_pages + storefront.gated_skips} page fetches"
        )
    status = RoasterRunStatus(
        roaster=roaster.name,
        run_id=run_id,
        status=STATUS_SUCCESS,
        attempts=1,
        catalog_path=path.name,
        products_found=len(products),
        new_products=len(new_urls),
        fetched_pages=page_fetch_count + storefront.fetched_pages,
        platform=roaster.platform,
        classified_by_source=dict(classification.by_source),
        undated=classification.undated,
        outside_window=classification.outside_window,
        completed_at=utc_now_iso(),
        note=note,
    )
    return _write_catalog_and_status(
        ctx,
        roaster,
        status,
        catalog_products,
        [],
        path,
    )
