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
from .network import (
    fetch_product_page_text,
    fetch_products_for_roaster,
    fetch_wix_product_sitemap_lastmods,
    merge_headers,
)
from .seen_products import SeenProducts
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


def _parse_wix_lastmod(value: str) -> Optional[date]:
    return _parse_http_date(value) or _parse_iso_date(value)


def _resolve_update_date(
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
) -> tuple[set[str], dict[str, int], int, int]:
    new_urls: set[str] = set()
    by_source = {
        "shopify_published_at": 0,
        "http_last_modified": 0,
        "wix_lastmod": 0,
        "seen_at": 0,
    }
    undated = 0
    not_today = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for product in products:
        description = descriptions_by_url.get(product.url, "")
        seen_hash = seen_products.compute_hash(product.url, product.name, description)
        seen_entry = seen_products.get(seen_hash)
        seen_at = seen_entry.first_seen_at if seen_entry else ""
        http_last_modified = http_last_modified_by_url.get(product.url, "")
        wix_lastmod = wix_lastmod_by_url.get(product.url, "")

        updated_date, source = _resolve_update_date(
            product, http_last_modified, wix_lastmod, seen_at
        )
        if updated_date is None:
            undated += 1
            source = "undated"
            updated_date = run_day

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
    include_new_digest: bool,
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
    if include_new_digest and new_items:
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
            settings.digest_model,
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
    seen_products: SeenProducts,
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
        seen_products,
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
    seen_products: SeenProducts,
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
    run_day = datetime.strptime(run_id, "%Y%m%d").date()
    coffee_list = ""
    logger.info(
        "Found %d products for %s (%d to evaluate).",
        len(products),
        roaster.name,
        len(new_products),
    )

    page_text_by_id = {product.product_id: "" for product in new_products}
    http_last_modified_by_url: dict[str, str] = {}
    wix_lastmod_by_url: dict[str, str] = {}
    page_fetch_count = 0

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

    def build_descriptions_by_url() -> dict[str, str]:
        descriptions: dict[str, str] = {}
        for product in new_products:
            body_text = ""
            if product.body_html:
                body_text = sanitize_html_to_text(
                    product.body_html,
                    settings.batch_page_text_max_chars,
                    remove_boilerplate=False,
                )
            description = page_text_by_id.get(product.product_id, "") or body_text
            descriptions[product.url] = description
        return descriptions

    if roaster.platform == "wix":
        sitemap_headers = merge_headers(
            {"User-Agent": USER_AGENT},
            roaster.product_page_headers,
            logger,
            f"{roaster.name} sitemap",
        )
        sitemap_lastmods = await fetch_wix_product_sitemap_lastmods(
            http_client,
            roaster,
            settings,
            robots_cache,
            logger,
            sitemap_headers,
            http_semaphore,
            jitter_multiplier=roaster.jitter_multiplier,
        )
        if sitemap_lastmods:
            wix_lastmod_by_url.update(sitemap_lastmods)

    if settings.fetch_only:
        logger.info(
            "Fetch-only mode enabled; skipping product page fetches and Gemini."
        )
        descriptions_by_url = build_descriptions_by_url()
        new_urls, by_source, undated, not_today = classify_new_products(
            new_products,
            run_day,
            seen_products,
            descriptions_by_url,
            http_last_modified_by_url,
            wix_lastmod_by_url,
            roaster.platform,
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
            "New products for %s (dated %s): %d "
            "[shopify_published_at=%d, http_last_modified=%d, wix_lastmod=%d, "
            "seen_at=%d, "
            "undated=%d, not_today=%d]. Page text fetched for %d products.",
            roaster.name,
            run_id,
            len(new_urls),
            by_source["shopify_published_at"],
            by_source["http_last_modified"],
            by_source["wix_lastmod"],
            by_source["seen_at"],
            undated,
            not_today,
            page_fetch_count,
        )
        return report_path, []

    if roaster.platform != "shopify":
        page_headers = merge_headers(
            {"User-Agent": USER_AGENT},
            roaster.product_page_headers,
            logger,
            f"{roaster.name} product page",
        )
        products_needing_pages = [
            product for product in new_products if not product.body_html
        ]
        page_fetch_count = len(products_needing_pages)
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
                    jitter_multiplier=roaster.jitter_multiplier,
                )
                for product in products_needing_pages
            ]
            page_results = await asyncio.gather(*page_tasks)
            for product, (text, last_modified) in zip(
                products_needing_pages, page_results
            ):
                page_text_by_id[product.product_id] = trim_text_at_phrases(
                    text, roaster.page_text_stop_phrases
                )
                if last_modified:
                    http_last_modified_by_url[product.url] = last_modified
    else:
        logger.info(
            "Skipping product page fetches for %s (platform shopify).", roaster.name
        )

    descriptions_by_url = build_descriptions_by_url()
    new_urls, by_source, undated, not_today = classify_new_products(
        new_products,
        run_day,
        seen_products,
        descriptions_by_url,
        http_last_modified_by_url,
        wix_lastmod_by_url,
        roaster.platform,
    )
    new_items: list[dict[str, Any]] = []
    if settings.save_report and new_urls:
        for product in new_products:
            if product.url not in new_urls:
                continue
            description = descriptions_by_url.get(product.url, "")
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
            "New products for %s (dated %s): %d "
            "[shopify_published_at=%d, http_last_modified=%d, wix_lastmod=%d, "
            "seen_at=%d, "
            "undated=%d, not_today=%d]. Page text fetched for %d products.",
            roaster.name,
            run_id,
            len(new_urls),
            by_source["shopify_published_at"],
            by_source["http_last_modified"],
            by_source["wix_lastmod"],
            by_source["seen_at"],
            undated,
            not_today,
            page_fetch_count,
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
            "New products for %s (dated %s): %d "
            "[shopify_published_at=%d, http_last_modified=%d, wix_lastmod=%d, "
            "seen_at=%d, "
            "undated=%d, not_today=%d]. Page text fetched for %d products.",
            roaster.name,
            run_id,
            len(new_urls),
            by_source["shopify_published_at"],
            by_source["http_last_modified"],
            by_source["wix_lastmod"],
            by_source["seen_at"],
            undated,
            not_today,
            page_fetch_count,
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
        "New products for %s (dated %s): %d "
        "[shopify_published_at=%d, http_last_modified=%d, wix_lastmod=%d, "
        "seen_at=%d, "
        "undated=%d, not_today=%d]. Page text fetched for %d products.",
        roaster.name,
        run_id,
        len(new_urls),
        by_source["shopify_published_at"],
        by_source["http_last_modified"],
        by_source["wix_lastmod"],
        by_source["seen_at"],
        undated,
        not_today,
        page_fetch_count,
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
    logger.info(
        "Using Gemini models: roaster=%s digest=%s",
        settings.model,
        settings.digest_model,
    )
    logger.info(
        "New-products digest: %s",
        "enabled" if settings.new_products_digest else "disabled",
    )

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
                        shopify_published_at=str(
                            item.get("shopify_published_at", "") or ""
                        ),
                    )
                    digest_products.append(product)
                    descriptions_by_url[url] = str(item.get("description", "") or "")
                run_day = datetime.strptime(run_id, "%Y%m%d").date()
                new_urls, by_source, undated, not_today = classify_new_products(
                    digest_products,
                    run_day,
                    seen_products,
                    descriptions_by_url,
                    http_last_modified_by_url,
                    wix_lastmod_by_url,
                    "unknown",
                )
                filtered_new_items = [
                    item
                    for item, product in zip(new_items, digest_products)
                    if product.url and product.url in new_urls
                ]
                logger.info(
                    "Digest-only new products (dated %s): %d "
                    "[shopify_published_at=%d, http_last_modified=%d, "
                    "wix_lastmod=%d, seen_at=%d, "
                    "undated=%d, not_today=%d].",
                    run_id,
                    len(filtered_new_items),
                    by_source["shopify_published_at"],
                    by_source["http_last_modified"],
                    by_source["wix_lastmod"],
                    by_source["seen_at"],
                    undated,
                    not_today,
                )
            finally:
                seen_products.close()
        digest_jobs = build_digest_jobs(
            reports,
            filtered_new_items,
            language,
            settings.batch_page_text_max_chars,
            settings.new_products_digest,
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
    seen_products = SeenProducts(settings.seen_db_path, logger)
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
                    seen_products,
                )
                for roaster in roasters
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        seen_products.close()

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
            settings.new_products_digest,
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
