from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
from dataclasses import dataclass
from datetime import datetime, timezone
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
from .models import RoasterSource
from .network import fetch_product_page_text, fetch_products_for_roaster, merge_headers
from .page_cache import PageCache
from .parsing import load_denylist, load_roasters
from .prompts import (
    build_batch_prompt,
    build_digest_prompt,
    build_new_products_digest_prompt,
    build_roaster_ratings_digest_prompt,
    format_coffee_list,
    normalize_language,
)
from .text_utils import sanitize_html_to_text, trim_text_at_phrases
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


def build_digest_jobs(
    reports: list[tuple[str, str]],
    new_items: list[dict[str, str]],
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
) -> tuple[Optional[Path], list[dict[str, str]]]:
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
) -> tuple[Optional[Path], list[dict[str, str]]]:
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
    coffee_list = format_coffee_list(new_products) if settings.save_report else ""
    cached_by_url: dict[str, bool] = {}
    if settings.save_report:
        if page_cache:
            for product in new_products:
                cached_by_url[product.url] = page_cache.get(product.url) is not None
        else:
            for product in new_products:
                cached_by_url[product.url] = False
    logger.info(
        "Found %d products for %s (%d to evaluate).",
        len(products),
        roaster.name,
        len(new_products),
    )

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
        if settings.save_report:
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            write_report(report_path, note="Fetch-only mode enabled; no LLM output.")
            return report_path, []
        return None, []

    page_headers = merge_headers(
        {"User-Agent": USER_AGENT},
        roaster.product_page_headers,
        logger,
        f"{roaster.name} product page",
    )
    products_needing_pages = [product for product in new_products if not product.body_html]
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
                page_cache,
                roaster.jitter_multiplier,
            )
            for product in products_needing_pages
        ]
        page_texts = await asyncio.gather(*page_tasks)
        for product, text in zip(products_needing_pages, page_texts):
            page_text_by_id[product.product_id] = trim_text_at_phrases(
                text, roaster.page_text_stop_phrases
            )

    new_items: list[dict[str, str]] = []
    if settings.save_report:
        for product in new_products:
            if cached_by_url.get(product.url, False):
                continue
            body_text = ""
            if product.body_html:
                body_text = sanitize_html_to_text(
                    product.body_html,
                    settings.batch_page_text_max_chars,
                    remove_boilerplate=False,
                )
            description = page_text_by_id.get(product.product_id, "") or body_text
            new_items.append(
                {
                    "roaster": roaster.name,
                    "name": product.name,
                    "url": product.url,
                    "list_price": product.list_price,
                    "badge": product.list_badge,
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
        if settings.save_report:
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            write_report(report_path, note="Gemini skipped by configuration.")
            return report_path, new_items
        return None, new_items

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
        if settings.save_report:
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            write_report(
                report_path,
                note="Gemini returned no text for this roaster.",
                grounding_payload=grounding,
            )
            return report_path, new_items
        return None, new_items
    if settings.save_report:
        report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
        write_report(report_path, body=markdown, grounding_payload=grounding)
        return report_path, new_items
    return None, new_items


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
        digest_jobs = build_digest_jobs(
            reports,
            new_items,
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
    new_items: list[dict[str, str]] = []
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
