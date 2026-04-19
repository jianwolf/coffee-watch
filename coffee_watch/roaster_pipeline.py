"""Per-roaster fetch → classify → LLM → report pipeline and digest jobs."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from .classify import (
    NEW_PRODUCTS_WINDOW_DAYS,
    classify_new_products,
    log_new_products_summary,
)
from .config import Settings
from .constants import USER_AGENT
from .context import RunContext
from .gemini import format_grounding_metadata, grounding_queries
from .llm_backend import LLMBackend
from .models import ProductCandidate, RoasterRunStatus, RoasterSource
from .network import (
    fetch_product_page_text,
    fetch_products_for_roaster,
    fetch_wix_product_sitemap_lastmods,
    merge_headers,
)
from .prompts import (
    build_batch_prompt,
    build_digest_prompt,
    build_new_products_digest_prompt,
    build_roaster_ratings_digest_prompt,
    format_coffee_list,
)
from .report_status import (
    STATUS_EMPTY,
    STATUS_FAILURE,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    append_failed_roaster_lines_to_digest,
    build_roaster_failure_line,
    digest_report_name,
    utc_now_iso,
    write_items_sidecar,
    write_status_sidecar,
)
from .reporting import (
    init_report,
    make_report_path,
    report_file_path,
    save_prompt_text,
)
from .text_utils import format_variant_lines, sanitize_html_to_text, trim_text_at_phrases
from .url_utils import normalize_base_url

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DigestJob:
    name: str
    prompt: str


async def process_roaster(
    ctx: RunContext,
    roaster: RoasterSource,
) -> tuple[Optional[Path], list[dict[str, Any]], RoasterRunStatus]:
    try:
        return await _process_roaster_inner(ctx, roaster)
    except Exception as exc:
        ctx.logger.exception(
            "Roaster processing failed for %s: %s", roaster.name, exc
        )
        status = RoasterRunStatus(
            roaster=roaster.name,
            run_id=ctx.run_id,
            status=STATUS_FAILURE,
            attempts=0,
            report_path=None,
            products_found=0,
            new_products=0,
            fetched_pages=0,
            platform=roaster.platform,
            completed_at=utc_now_iso(),
            note=f"unhandled exception: {exc.__class__.__name__}",
        )
        write_status_sidecar(ctx.settings.reports_dir, status, ctx.logger)
        return None, [], status


def _write_roaster_markdown(
    report_path: Path,
    roaster_name: str,
    run_id: str,
    body: Optional[str],
    note: Optional[str],
    grounding_payload: Optional[dict[str, Any]],
    coffee_list: str,
) -> None:
    init_report(report_path, roaster_name, run_id)
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


def _build_new_items(
    roaster_name: str,
    products: list[ProductCandidate],
    new_urls: set[str],
    descriptions_by_url: dict[str, str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for product in products:
        if product.url not in new_urls:
            continue
        items.append(
            {
                "roaster": roaster_name,
                "product_id": product.product_id,
                "name": product.name,
                "url": product.url,
                "list_price": product.list_price,
                "badge": product.list_badge,
                "variant_lines": format_variant_lines(product.variants),
                "description": descriptions_by_url.get(product.url, ""),
                "shopify_updated_at": product.shopify_updated_at,
                "shopify_published_at": product.shopify_published_at,
            }
        )
    return items


async def _process_roaster_inner(
    ctx: RunContext,
    roaster: RoasterSource,
) -> tuple[Optional[Path], list[dict[str, Any]], RoasterRunStatus]:
    settings = ctx.settings
    log = ctx.logger
    run_id = ctx.run_id

    base_url = normalize_base_url(roaster.base_url)
    domain = urlsplit(base_url).netloc.lower()
    if domain and domain in ctx.denylist:
        log.info("Skipping denylisted domain %s", domain)
        status = RoasterRunStatus(
            roaster=roaster.name,
            run_id=run_id,
            status=STATUS_SKIPPED,
            attempts=0,
            report_path=None,
            products_found=0,
            new_products=0,
            fetched_pages=0,
            platform=roaster.platform,
            completed_at=utc_now_iso(),
            note="denylisted domain",
        )
        write_status_sidecar(settings.reports_dir, status, log)
        return None, [], status

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
            attempts=0,
            report_path=None,
            products_found=0,
            new_products=0,
            fetched_pages=0,
            platform=roaster.platform,
            completed_at=utc_now_iso(),
            note="no products parsed",
        )
        write_status_sidecar(settings.reports_dir, status, log)
        return None, [], status

    new_products = list(products)
    run_day = datetime.strptime(run_id, "%Y%m%d").date()
    log.info(
        "Found %d products for %s (%d to evaluate).",
        len(products),
        roaster.name,
        len(new_products),
    )

    page_text_by_id = {product.product_id: "" for product in new_products}
    http_last_modified_by_url: dict[str, str] = {}
    wix_lastmod_by_url: dict[str, str] = {}
    page_fetch_count = 0

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
            log,
            f"{roaster.name} sitemap",
        )
        sitemap_lastmods = await fetch_wix_product_sitemap_lastmods(
            ctx.http_client,
            roaster,
            settings,
            ctx.robots_cache,
            log,
            sitemap_headers,
            ctx.limiter,
            jitter_multiplier=roaster.jitter_multiplier,
        )
        if sitemap_lastmods:
            wix_lastmod_by_url.update(sitemap_lastmods)

    if settings.fetch_only:
        log.info("Fetch-only mode enabled; skipping product page fetches and LLM.")
        descriptions_by_url = build_descriptions_by_url()
        new_urls, by_source, undated, outside_window = classify_new_products(
            new_products,
            run_day,
            ctx.seen_products,
            descriptions_by_url,
            http_last_modified_by_url,
            wix_lastmod_by_url,
            roaster.platform,
            window_days=NEW_PRODUCTS_WINDOW_DAYS,
            persist_seen=True,
        )
        report_path: Optional[Path] = None
        coffee_list = ""
        new_items = _build_new_items(
            roaster.name, new_products, new_urls, descriptions_by_url
        )
        if settings.save_report:
            coffee_list = format_coffee_list(
                new_products,
                page_text_by_id,
                settings.batch_page_text_max_chars,
            )
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            _write_roaster_markdown(
                report_path,
                roaster.name,
                run_id,
                body=None,
                note="Fetch-only mode enabled; no LLM output.",
                grounding_payload=None,
                coffee_list=coffee_list,
            )
            log.info("Saved roaster report to %s", report_path)
            write_items_sidecar(
                settings.reports_dir, roaster.name, run_id, new_items, log
            )
        log_new_products_summary(
            log, roaster.name, run_id, new_urls, by_source, undated,
            outside_window, page_fetch_count,
        )
        status = RoasterRunStatus(
            roaster=roaster.name,
            run_id=run_id,
            status=STATUS_SUCCESS,
            attempts=0,
            report_path=report_path.name if report_path else None,
            products_found=len(products),
            new_products=len(new_urls),
            fetched_pages=page_fetch_count,
            platform=roaster.platform,
            classified_by_source=dict(by_source),
            undated=undated,
            outside_window=outside_window,
            completed_at=utc_now_iso(),
            note="fetch-only",
        )
        write_status_sidecar(settings.reports_dir, status, log)
        return report_path, new_items, status

    if roaster.platform != "shopify":
        page_headers = merge_headers(
            {"User-Agent": USER_AGENT},
            roaster.product_page_headers,
            log,
            f"{roaster.name} product page",
        )
        products_needing_pages = [
            product for product in new_products if not product.body_html
        ]
        page_fetch_count = len(products_needing_pages)
        if products_needing_pages:
            page_tasks = [
                fetch_product_page_text(
                    ctx.http_client,
                    product,
                    settings,
                    ctx.robots_cache,
                    log,
                    page_headers,
                    ctx.limiter,
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
        log.info(
            "Skipping product page fetches for %s (platform shopify).", roaster.name
        )

    descriptions_by_url = build_descriptions_by_url()
    new_urls, by_source, undated, outside_window = classify_new_products(
        new_products,
        run_day,
        ctx.seen_products,
        descriptions_by_url,
        http_last_modified_by_url,
        wix_lastmod_by_url,
        roaster.platform,
        window_days=NEW_PRODUCTS_WINDOW_DAYS,
        persist_seen=True,
    )
    new_items = (
        _build_new_items(roaster.name, new_products, new_urls, descriptions_by_url)
        if settings.save_report and new_urls
        else []
    )

    prompt = build_batch_prompt(
        roaster.name,
        new_products,
        page_text_by_id,
        settings.batch_page_text_max_chars,
        ctx.language,
        settings.user_ask,
    )
    prompt_path = save_prompt_text(ctx.assets_dir, run_id, roaster.name, prompt)
    log.info("Saved LLM prompt for %s to %s", roaster.name, prompt_path)
    if settings.save_prompt:
        report_prompt_path = save_prompt_text(
            settings.reports_dir, run_id, roaster.name, prompt
        )
        log.info(
            "Saved LLM prompt copy for %s to %s", roaster.name, report_prompt_path
        )

    if settings.skip_llm:
        log.info("LLM skipped by configuration.")
        report_path = None
        coffee_list = ""
        if settings.save_report:
            coffee_list = format_coffee_list(
                new_products,
                page_text_by_id,
                settings.batch_page_text_max_chars,
            )
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            _write_roaster_markdown(
                report_path,
                roaster.name,
                run_id,
                body=None,
                note="LLM skipped by configuration.",
                grounding_payload=None,
                coffee_list=coffee_list,
            )
            log.info("Saved roaster report to %s", report_path)
            write_items_sidecar(
                settings.reports_dir, roaster.name, run_id, new_items, log
            )
        log_new_products_summary(
            log, roaster.name, run_id, new_urls, by_source, undated,
            outside_window, page_fetch_count,
        )
        status = RoasterRunStatus(
            roaster=roaster.name,
            run_id=run_id,
            status=STATUS_SKIPPED,
            attempts=0,
            report_path=report_path.name if report_path else None,
            products_found=len(products),
            new_products=len(new_urls),
            fetched_pages=page_fetch_count,
            platform=roaster.platform,
            classified_by_source=dict(by_source),
            undated=undated,
            outside_window=outside_window,
            completed_at=utc_now_iso(),
            note="skip-llm",
        )
        write_status_sidecar(settings.reports_dir, status, log)
        return report_path, new_items, status

    markdown: Optional[str] = None
    grounding: Optional[dict[str, Any]] = None
    max_attempts = max(1, settings.max_llm_attempts)
    base_delay = max(0.0, settings.llm_retry_base_delay_s)
    max_delay = max(base_delay, settings.llm_retry_max_delay_s)
    attempts_taken = 0
    for attempt in range(1, max_attempts + 1):
        attempts_taken = attempt
        markdown, grounding = await ctx.llm.evaluate_roaster_markdown(
            settings.model,
            roaster.name,
            prompt,
            log,
            settings.llm_timeout_s,
        )
        if markdown:
            if attempt > 1:
                log.info(
                    "LLM succeeded for %s on attempt %d/%d.",
                    roaster.name,
                    attempt,
                    max_attempts,
                )
            break
        log.warning(
            "LLM returned no text for %s (attempt %d/%d).",
            roaster.name,
            attempt,
            max_attempts,
        )
        if attempt < max_attempts and base_delay > 0:
            delay_cap = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_for = random.uniform(base_delay, delay_cap) if delay_cap > base_delay else base_delay
            await asyncio.sleep(sleep_for)

    report_path: Optional[Path] = None
    coffee_list = ""
    if markdown is None:
        log.warning(
            "LLM returned no text for %s after %d attempts.",
            roaster.name,
            max_attempts,
        )
        if settings.save_report:
            coffee_list = format_coffee_list(
                new_products,
                page_text_by_id,
                settings.batch_page_text_max_chars,
            )
            report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
            _write_roaster_markdown(
                report_path,
                roaster.name,
                run_id,
                body=None,
                note=build_roaster_failure_line(roaster.name),
                grounding_payload=grounding,
                coffee_list=coffee_list,
            )
            log.info("Saved roaster report to %s", report_path)
            write_items_sidecar(
                settings.reports_dir, roaster.name, run_id, new_items, log
            )
        log_new_products_summary(
            log, roaster.name, run_id, new_urls, by_source, undated,
            outside_window, page_fetch_count,
        )
        status = RoasterRunStatus(
            roaster=roaster.name,
            run_id=run_id,
            status=STATUS_FAILURE,
            attempts=attempts_taken,
            report_path=report_path.name if report_path else None,
            products_found=len(products),
            new_products=len(new_urls),
            fetched_pages=page_fetch_count,
            platform=roaster.platform,
            classified_by_source=dict(by_source),
            undated=undated,
            outside_window=outside_window,
            grounding_queries=tuple(grounding_queries(grounding)),
            completed_at=utc_now_iso(),
            note=f"llm empty after {attempts_taken} attempts",
        )
        write_status_sidecar(settings.reports_dir, status, log)
        return report_path, new_items, status

    if settings.save_report:
        coffee_list = format_coffee_list(
            new_products,
            page_text_by_id,
            settings.batch_page_text_max_chars,
        )
        report_path = make_report_path(settings.reports_dir, roaster.name, run_id)
        _write_roaster_markdown(
            report_path,
            roaster.name,
            run_id,
            body=markdown,
            note=None,
            grounding_payload=grounding,
            coffee_list=coffee_list,
        )
        log.info("Saved roaster report to %s", report_path)
        write_items_sidecar(
            settings.reports_dir, roaster.name, run_id, new_items, log
        )
    log_new_products_summary(
        log, roaster.name, run_id, new_urls, by_source, undated,
        outside_window, page_fetch_count,
    )
    status = RoasterRunStatus(
        roaster=roaster.name,
        run_id=run_id,
        status=STATUS_SUCCESS,
        attempts=attempts_taken,
        report_path=report_path.name if report_path else None,
        products_found=len(products),
        new_products=len(new_urls),
        fetched_pages=page_fetch_count,
        platform=roaster.platform,
        classified_by_source=dict(by_source),
        undated=undated,
        outside_window=outside_window,
        grounding_queries=tuple(grounding_queries(grounding)),
        completed_at=utc_now_iso(),
    )
    write_status_sidecar(settings.reports_dir, status, log)
    return report_path, new_items, status


def build_digest_jobs(
    reports: list[tuple[str, str]],
    new_items: list[dict[str, Any]],
    language: str,
    max_chars: int,
    include_new_digest: bool,
    user_ask: str,
) -> list[DigestJob]:
    jobs: list[DigestJob] = []
    if reports:
        jobs.append(DigestJob("digest", build_digest_prompt(reports, language, user_ask)))
        jobs.append(
            DigestJob(
                "roaster-digest",
                build_roaster_ratings_digest_prompt(reports, language, user_ask),
            )
        )
    if include_new_digest and new_items:
        jobs.append(
            DigestJob(
                "new-digest",
                build_new_products_digest_prompt(new_items, max_chars, language, user_ask),
            )
        )
    return jobs


def save_digest_prompts(
    jobs: list[DigestJob],
    assets_dir: Path,
    reports_dir: Path,
    run_id: str,
    save_prompt: bool,
    log: logging.Logger,
) -> None:
    for job in jobs:
        prompt_path = save_prompt_text(assets_dir, run_id, job.name, job.prompt)
        log.info("Saved LLM %s prompt to %s", job.name, prompt_path)
        if save_prompt:
            report_prompt_path = save_prompt_text(
                reports_dir, run_id, job.name, job.prompt
            )
            log.info(
                "Saved LLM %s prompt copy to %s", job.name, report_prompt_path
            )


async def generate_digest_reports(
    jobs: list[DigestJob],
    llm: LLMBackend,
    settings: Settings,
    log: logging.Logger,
    run_id: str,
    failed_roasters: Optional[list[str]] = None,
) -> None:
    if not jobs:
        return
    failed_names = failed_roasters or []

    async def _run_digest_job(job: DigestJob) -> tuple[DigestJob, Optional[str]]:
        digest = await llm.generate_digest_markdown(
            settings.digest_model,
            job.prompt,
            log,
            settings.llm_timeout_s,
            request_name=job.name,
        )
        return job, digest

    tasks = [asyncio.create_task(_run_digest_job(job)) for job in jobs]

    for task in asyncio.as_completed(tasks):
        job, digest = await task
        output_text = append_failed_roaster_lines_to_digest(digest or "", failed_names)
        if output_text and settings.save_report:
            digest_path = report_file_path(
                settings.reports_dir, digest_report_name(job.name), run_id, None, "md"
            )
            digest_path.write_text(output_text, encoding="utf-8")
            log.info("Saved %s report to %s", job.name, digest_path)
            if not digest:
                log.warning(
                    "LLM returned no %s text; saved failure-only digest.", job.name
                )
        elif not digest:
            log.warning("LLM returned no %s text.", job.name)
