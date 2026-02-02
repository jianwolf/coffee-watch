from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from google import genai

from .config import Settings
from .constants import USER_AGENT
from .gemini import evaluate_roaster_markdown, format_grounding_metadata, generate_digest_markdown
from .logging_utils import setup_logging
from .models import ProductCandidate, RoasterSource
from .network import fetch_product_page_text, fetch_products_for_roaster, merge_headers
from .page_cache import CachedPage, PageCache
from .parsing import load_denylist, load_roasters
from .prompts import (
    build_batch_prompt,
    build_digest_prompt,
    build_new_products_digest_prompt,
    build_roaster_ratings_digest_prompt,
    format_coffee_list,
    normalize_language,
)
from .text_utils import format_variant_lines, sanitize_html_to_text, trim_text_at_phrases
from .reporting import (
    extract_coffee_list_items,
    init_report,
    load_reports_for_digest,
    make_report_path,
    report_file_path,
    save_prompt_text,
)
from .url_utils import normalize_base_url


@dataclass(frozen=True)
class DigestJob:
    name: str
    prompt: str


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
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date()


def _resolve_update_date(
    product: ProductCandidate, cached: Optional[CachedPage]
) -> tuple[Optional[date], str]:
    shopify_value = product.shopify_updated_at
    if not shopify_value and cached:
        shopify_value = cached.shopify_updated_at
    shopify_date = _parse_iso_date(shopify_value)
    if shopify_date:
        return shopify_date, "shopify_updated_at"
    if cached:
        html_value = cached.html_last_modified or cached.last_modified
        html_date = _parse_http_date(html_value)
        if html_date:
            return html_date, "html_last_modified"
        cached_date = _parse_iso_date(cached.cached_at)
        if cached_date:
            return cached_date, "cached_at"
    return None, "unknown"


def classify_new_products(
    products: list[ProductCandidate],
    cached_pages: dict[str, Optional[CachedPage]],
    run_day: date,
) -> tuple[set[str], dict[str, int], int, int]:
    new_urls: set[str] = set()
    by_source = {"shopify_updated_at": 0, "html_last_modified": 0, "cached_at": 0}
    undated = 0
    not_today = 0
    for product in products:
        cached = cached_pages.get(product.url)
        updated_date, source = _resolve_update_date(product, cached)
        if updated_date is None:
            undated += 1
            continue
        if updated_date != run_day:
            not_today += 1
            continue
        new_urls.add(product.url)
        if source in by_source:
            by_source[source] += 1
    return new_urls, by_source, undated, not_today


def build_digest_jobs(
    reports: list[tuple[str, str]],
    new_items: list[dict[str, Any]],
    language: str,
    max_chars: int,
) -> list[DigestJob]:
    jobs: list[DigestJob] = []
    if reports:
        jobs.append(DigestJob("digest", build_digest_prompt(reports, language)))
        jobs.append(
            DigestJob(
                "roaster-digest",
                build_roaster_ratings_digest_prompt(reports, language),
            )
        )
    if new_items:
        jobs.append(
            DigestJob(
                "new-digest",
                build_new_products_digest_prompt(new_items, max_chars, language),
            )
        )
    return jobs


def save_digest_prompts(
    jobs: list[DigestJob],
    assets_dir: Path,
    reports_dir: Path,
    run_id: str,
    save_prompt: bool,
    logger: logging.Logger,
) -> None:
    for job in jobs:
        prompt_path = save_prompt_text(assets_dir, run_id, job.name, job.prompt)
        logger.info("Saved Gemini %s prompt to %s", job.name, prompt_path)
        if save_prompt:
            report_prompt_path = save_prompt_text(
                reports_dir, run_id, job.name, job.prompt
            )
            logger.info(
                "Saved Gemini %s prompt copy to %s", job.name, report_prompt_path
            )


async def generate_digest_reports(
    jobs: list[DigestJob],
    genai_client: genai.Client,
    settings: Settings,
    logger: logging.Logger,
    run_id: str,
) -> None:
    if not jobs:
        return
    tasks = [
        generate_digest_markdown(
            genai_client,
            settings.model,
            job.prompt,
            logger,
            settings.gemini_timeout_s,
            request_name=job.name,
        )
        for job in jobs
    ]
    results = await asyncio.gather(*tasks)
    for job, digest in zip(jobs, results):
        if digest and settings.save_report:
            digest_path = report_file_path(
                settings.reports_dir, job.name, run_id, None, "md"
            )
            digest_path.write_text(digest, encoding="utf-8")
            logger.info("Saved %s report to %s", job.name, digest_path)
        elif not digest:
            logger.warning("Gemini returned no %s text.", job.name)


async def process_roaster(
    roaster: RoasterSource,
    http_client: httpx.AsyncClient,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    run_id: str,
    assets_dir: Path,
    http_semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    api_key: Optional[str],
    language: str,
    denylist: set[str],
    page_cache: Optional[PageCache],
) -> tuple[Optional[Path], list[dict[str, Any]]]:
    try:
        return await _process_roaster_inner(
            roaster,
            http_client,
            settings,
            robots_cache,
            run_id,
            assets_dir,
            http_semaphore,
            logger,
            api_key,
            language,
            denylist,
            page_cache,
        )
    except Exception as exc:
        logger.exception("Roaster processing failed for %s: %s", roaster.name, exc)
        return None, []


async def _process_roaster_inner(
    roaster: RoasterSource,
    http_client: httpx.AsyncClient,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    run_id: str,
    assets_dir: Path,
    http_semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    api_key: Optional[str],
    language: str,
    denylist: set[str],
    page_cache: Optional[PageCache],
) -> tuple[Optional[Path], list[dict[str, Any]]]:
    base_url = normalize_base_url(roaster.base_url)
    domain = urlsplit(base_url).netloc.lower()
    if domain and domain in denylist:
        logger.info("Skipping denylisted domain %s", domain)
        return None

    products = await fetch_products_for_roaster(
        http_client,
        roaster,
        settings,
        robots_cache,
        assets_dir,
        run_id,
        http_semaphore,
        logger,
    )
    if not products:
        logger.info("No products parsed for %s", roaster.name)
        return None, []

    new_products = list(products)
    cached_pages: dict[str, Optional[CachedPage]] = {}
    if page_cache:
        for product in new_products:
            cached_pages[product.url] = page_cache.get(product.url)
            page_cache.update_shopify_updated_at(
                product.url, product.shopify_updated_at
            )
    else:
        for product in new_products:
            cached_pages[product.url] = None
    run_day = datetime.strptime(run_id, "%Y%m%d").date()
    coffee_list = ""
    logger.info(
        "Found %d products for %s (%d to evaluate).",
        len(products),
        roaster.name,
        len(new_products),
    )

    page_text_by_id = {product.product_id: "" for product in new_products}
    cache_usage: dict[str, int] = {"hits": 0}

    def write_report(
        report_path: Path,
        note: Optional[str] = None,
        body: Optional[str] = None,
        grounding_payload: Optional[dict[str, Any]] = None,
    ) -> None:
        init_report(report_path, roaster.name, run_id)
        with report_path.open("a", encoding="utf-8") as handle:
            if note:
                handle.write(f"\n{note}\n")
            if body:
                handle.write(body)
                handle.write("\n")
            grounding_block = format_grounding_metadata(grounding_payload or {})
            if grounding_block:
                handle.write("\n")
                handle.write(grounding_block)
            if coffee_list:
                handle.write("\n")
                handle.write(coffee_list)
        logger.info("Saved roaster report to %s", report_path)

    if settings.fetch_only:
        logger.info(
            "Fetch-only mode enabled; skipping product page fetches and Gemini."
        )
        new_urls, by_source, undated, not_today = classify_new_products(
            new_products,
            cached_pages,
            run_day,
        )
        report_path = None
        if settings.save_report:
            coffee_list = format_coffee_list(
                new_products,
                page_text_by_id,
                settings.batch_page_text_max_chars,
            )
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            write_report(report_path, note="Fetch-only mode enabled; no LLM output.")
        logger.info(
            "New products for %s (updated %s): %d "
            "[shopify_updated_at=%d, html_last_modified=%d, cached_at=%d, undated=%d, not_today=%d]. "
            "Cached page text used for %d products.",
            roaster.name,
            run_id,
            len(new_urls),
            by_source["shopify_updated_at"],
            by_source["html_last_modified"],
            by_source["cached_at"],
            undated,
            not_today,
            cache_usage.get("hits", 0),
        )
        return report_path, []

    page_headers = merge_headers(
        {"User-Agent": USER_AGENT},
        roaster.product_page_headers,
        logger,
        f"{roaster.name} product page",
    )
    products_needing_pages = [
        product for product in new_products if not product.body_html
    ]
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
                page_cache,
                cache_usage,
                jitter_multiplier=roaster.jitter_multiplier,
            )
            for product in products_needing_pages
        ]
        page_texts = await asyncio.gather(*page_tasks)
        for product, text in zip(products_needing_pages, page_texts):
            page_text_by_id[product.product_id] = trim_text_at_phrases(
                text, roaster.page_text_stop_phrases
            )
    if page_cache:
        for product in new_products:
            cached_pages[product.url] = page_cache.get(product.url)

    new_urls, by_source, undated, not_today = classify_new_products(
        new_products,
        cached_pages,
        run_day,
    )
    new_items: list[dict[str, Any]] = []
    if settings.save_report and new_urls:
        for product in new_products:
            if product.url not in new_urls:
                continue
            body_text = ""
            if product.body_html:
                body_text = sanitize_html_to_text(
                    product.body_html,
                    settings.batch_page_text_max_chars,
                    remove_boilerplate=False,
                )
            description = page_text_by_id.get(product.product_id, "") or body_text
            variant_lines = format_variant_lines(product.variants)
            new_items.append(
                {
                    "roaster": roaster.name,
                    "product_id": product.product_id,
                    "name": product.name,
                    "url": product.url,
                    "list_price": product.list_price,
                    "badge": product.list_badge,
                    "variant_lines": variant_lines,
                    "description": description,
                }
            )

    prompt = build_batch_prompt(
        roaster.name,
        new_products,
        page_text_by_id,
        settings.batch_page_text_max_chars,
        language,
    )
    prompt_path = save_prompt_text(assets_dir, run_id, roaster.name, prompt)
    logger.info("Saved Gemini prompt for %s to %s", roaster.name, prompt_path)
    if settings.save_prompt:
        report_prompt_path = save_prompt_text(
            settings.reports_dir, run_id, roaster.name, prompt
        )
        logger.info(
            "Saved Gemini prompt copy for %s to %s", roaster.name, report_prompt_path
        )

    if settings.skip_gemini:
        logger.info("Gemini skipped by configuration.")
        report_path = None
        if settings.save_report:
            coffee_list = format_coffee_list(
                new_products,
                page_text_by_id,
                settings.batch_page_text_max_chars,
            )
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            write_report(report_path, note="Gemini skipped by configuration.")
        logger.info(
            "New products for %s (updated %s): %d "
            "[shopify_updated_at=%d, html_last_modified=%d, cached_at=%d, undated=%d, not_today=%d]. "
            "Cached page text used for %d products.",
            roaster.name,
            run_id,
            len(new_urls),
            by_source["shopify_updated_at"],
            by_source["html_last_modified"],
            by_source["cached_at"],
            undated,
            not_today,
            cache_usage.get("hits", 0),
        )
        return report_path, new_items

    genai_client = genai.Client(api_key=api_key) if api_key else genai.Client()
    markdown, grounding = await evaluate_roaster_markdown(
        genai_client,
        settings.model,
        roaster.name,
        prompt,
        logger,
        settings.gemini_timeout_s,
    )
    if markdown is None:
        logger.warning("Gemini returned no text for %s", roaster.name)
        report_path = None
        if settings.save_report:
            coffee_list = format_coffee_list(
                new_products,
                page_text_by_id,
                settings.batch_page_text_max_chars,
            )
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            write_report(
                report_path,
                note="Gemini returned no text for this roaster.",
                grounding_payload=grounding,
            )
        logger.info(
            "New products for %s (updated %s): %d "
            "[shopify_updated_at=%d, html_last_modified=%d, cached_at=%d, undated=%d, not_today=%d]. "
            "Cached page text used for %d products.",
            roaster.name,
            run_id,
            len(new_urls),
            by_source["shopify_updated_at"],
            by_source["html_last_modified"],
            by_source["cached_at"],
            undated,
            not_today,
            cache_usage.get("hits", 0),
        )
        return report_path, new_items
    if settings.save_report:
        coffee_list = format_coffee_list(
            new_products,
            page_text_by_id,
            settings.batch_page_text_max_chars,
        )
        report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
        write_report(report_path, body=markdown, grounding_payload=grounding)
    else:
        report_path = None
    logger.info(
        "New products for %s (updated %s): %d "
        "[shopify_updated_at=%d, html_last_modified=%d, cached_at=%d, undated=%d, not_today=%d]. "
        "Cached page text used for %d products.",
        roaster.name,
        run_id,
        len(new_urls),
        by_source["shopify_updated_at"],
        by_source["html_last_modified"],
        by_source["cached_at"],
        undated,
        not_today,
        cache_usage.get("hits", 0),
    )
    return report_path, new_items


async def run(settings: Settings) -> int:
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

    assets_dir = settings.log_path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GEMINI_API_KEY found; relying on default SDK auth.")

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
    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = run_date
    timeout = httpx.Timeout(settings.http_timeout_s)
    http_semaphore = asyncio.Semaphore(max(1, settings.http_concurrency))

    if settings.digest_only:
        report_paths = [
            path
            for path in settings.reports_dir.glob(f"{run_id}-*.md")
            if "-digest" not in path.name
        ]
        if not report_paths:
            logger.error(
                "Digest-only mode: no reports found for %s in %s",
                run_id,
                settings.reports_dir,
            )
            return 1
        reports = load_reports_for_digest(report_paths, logger)
        if not reports:
            logger.error("Digest-only mode: no readable reports found for %s.", run_id)
            return 1
        new_items = extract_coffee_list_items(reports, logger)
        filtered_new_items = new_items
        if new_items:
            page_cache = PageCache(settings.cache_db_path, logger)
            try:
                digest_products: list[ProductCandidate] = []
                cached_pages: dict[str, Optional[CachedPage]] = {}
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
                    )
                    digest_products.append(product)
                    cached_pages[url] = page_cache.get(url) if url else None
                run_day = datetime.strptime(run_id, "%Y%m%d").date()
                new_urls, by_source, undated, not_today = classify_new_products(
                    digest_products,
                    cached_pages,
                    run_day,
                )
                filtered_new_items = [
                    item
                    for item, product in zip(new_items, digest_products)
                    if product.url and product.url in new_urls
                ]
                logger.info(
                    "Digest-only new products (updated %s): %d "
                    "[shopify_updated_at=%d, html_last_modified=%d, cached_at=%d, undated=%d, not_today=%d].",
                    run_id,
                    len(filtered_new_items),
                    by_source["shopify_updated_at"],
                    by_source["html_last_modified"],
                    by_source["cached_at"],
                    undated,
                    not_today,
                )
            finally:
                page_cache.close()
        digest_jobs = build_digest_jobs(
            reports,
            filtered_new_items,
            language,
            settings.batch_page_text_max_chars,
        )
        save_digest_prompts(
            digest_jobs,
            assets_dir,
            settings.reports_dir,
            run_id,
            settings.save_prompt,
            logger,
        )
        if settings.skip_gemini:
            logger.info("Digest-only mode: Gemini skipped by configuration.")
            return 0
        genai_client = genai.Client(api_key=api_key) if api_key else genai.Client()
        await generate_digest_reports(
            digest_jobs,
            genai_client,
            settings,
            logger,
            run_id,
        )
        logger.info("Run complete.")
        return 0
    page_cache = PageCache(settings.cache_db_path, logger)
    try:
        async with httpx.AsyncClient(
            http2=True,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=timeout,
        ) as http_client:
            tasks = [
                process_roaster(
                    roaster,
                    http_client,
                    settings,
                    robots_cache,
                    run_id,
                    assets_dir,
                    http_semaphore,
                    logger,
                    api_key,
                    language,
                    denylist,
                    page_cache,
                )
                for roaster in roasters
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        page_cache.close()

    report_paths: list[Path] = []
    new_items: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, Exception):
            logger.exception("Roaster task failed: %s", result)
            continue
        path, items = result
        if path:
            report_paths.append(path)
        new_items.extend(items)
    reports: list[tuple[str, str]] = []
    if report_paths:
        reports = load_reports_for_digest(report_paths, logger)
    if settings.save_report:
        digest_jobs = build_digest_jobs(
            reports,
            new_items,
            language,
            settings.batch_page_text_max_chars,
        )
        if digest_jobs:
            save_digest_prompts(
                digest_jobs,
                assets_dir,
                settings.reports_dir,
                run_id,
                settings.save_prompt,
                logger,
            )
            genai_client = genai.Client(api_key=api_key) if api_key else genai.Client()
            await generate_digest_reports(
                digest_jobs,
                genai_client,
                settings,
                logger,
                run_id,
            )

    logger.info("Run complete.")
    return 0
