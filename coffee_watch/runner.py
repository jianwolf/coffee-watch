from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
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
from .parsing import load_denylist, load_roasters
from .prompts import (
    build_batch_prompt,
    build_digest_prompt,
    format_coffee_list,
    normalize_language,
)
from .reporting import (
    init_report,
    load_reports_for_digest,
    make_report_path,
    report_file_path,
    save_prompt_text,
    select_reports_for_digest,
)
from .url_utils import normalize_base_url


async def process_roaster(
    roaster: RoasterSource,
    http_client: httpx.AsyncClient,
    settings: Settings,
    robots_cache: dict[str, RobotFileParser],
    run_id: str,
    http_semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    api_key: Optional[str],
    language: str,
    denylist: set[str],
) -> Optional[Path]:
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
        settings.reports_dir,
        run_id,
        http_semaphore,
        logger,
    )
    if not products:
        logger.info("No products parsed for %s", roaster.name)
        return None

    new_products = list(products)
    coffee_list = format_coffee_list(new_products) if settings.save_report else ""
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

    if settings.fetch_only:
        logger.info(
            "Fetch-only mode enabled; skipping product page fetches and Gemini."
        )
        if settings.save_report:
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            write_report(report_path, note="Fetch-only mode enabled; no LLM output.")
            return report_path
        return None

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
        prompt_path = save_prompt_text(settings.reports_dir, run_id, roaster.name, prompt)
        logger.info("Saved Gemini prompt for %s to %s", roaster.name, prompt_path)

    if settings.skip_gemini:
        logger.info("Gemini skipped by configuration.")
        if settings.save_report:
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            write_report(report_path, note="Gemini skipped by configuration.")
            return report_path
        return None

    logger.info("Gemini prompt for %s:\n%s", roaster.name, prompt)
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
            return report_path
        return None
    if grounding:
        logger.info(
            "Gemini grounding metadata for %s: %s",
            roaster.name,
            json.dumps(grounding, ensure_ascii=True),
        )

    if settings.save_report:
        report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
        write_report(report_path, body=markdown, grounding_payload=grounding)
        return report_path
    return None


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
                http_semaphore,
                logger,
                api_key,
                language,
                denylist,
            )
            for roaster in roasters
        ]
        results = await asyncio.gather(*tasks)

    report_paths = [path for path in results if path]
    if settings.save_report and report_paths:
        selected_paths = select_reports_for_digest(report_paths, 5)
        if len(selected_paths) < 5:
            logger.info(
                "Digest will use %d report(s); fewer than 5 available.",
                len(selected_paths),
            )
        reports = load_reports_for_digest(selected_paths, logger)
        if reports:
            digest_prompt = build_digest_prompt(reports, language)
            genai_client = genai.Client(api_key=api_key) if api_key else genai.Client()
            digest = await generate_digest_markdown(
                genai_client,
                settings.model,
                digest_prompt,
                logger,
                settings.gemini_timeout_s,
            )
            if digest:
                digest_path = report_file_path(
                    settings.reports_dir, "digest", run_id, None, "md"
                )
                digest_path.write_text(digest, encoding="utf-8")
                logger.info("Saved digest report to %s", digest_path)
            else:
                logger.warning("Gemini returned no digest text.")

    logger.info("Run complete.")
    return 0
