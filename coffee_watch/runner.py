"""Top-level scrape orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from .catalog import build_combined_catalog, build_new_products_catalog
from .config import Settings
from .constants import USER_AGENT
from .context import RunContext
from .http_limits import PerHostLimiter
from .logging_utils import setup_logging
from .models import RoasterSource
from .parsing import load_denylist, load_roasters
from .report_status import (
    collect_missing_roaster_names,
    collect_resume_targets,
    merge_failed_roaster_names,
)
from .reporting import (
    combined_catalog_path,
    load_roaster_catalogs,
    new_products_catalog_path,
    save_json,
    today_roaster_catalog_paths,
)
from .roaster_pipeline import RoasterScrapeResult, process_roaster
from .seen_products import SeenProducts


async def _run_roasters(
    ctx: RunContext,
    target_roasters: list[RoasterSource],
) -> tuple[list[RoasterScrapeResult], list[str]]:
    if not target_roasters:
        return [], []
    tasks = [
        asyncio.create_task(process_roaster(ctx, roaster))
        for roaster in target_roasters
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    scrape_results: list[RoasterScrapeResult] = []
    failed_names: list[str] = []
    for roaster, result in zip(target_roasters, results):
        if isinstance(result, Exception):
            ctx.logger.exception(
                "Roaster task for %s raised: %s", roaster.name, result
            )
            failed_names.append(roaster.name)
            continue
        scrape_results.append(result)
        if result.status.status in {"failure", "empty"}:
            failed_names.append(roaster.name)
    return scrape_results, failed_names


def _write_run_catalogs(
    settings: Settings,
    run_id: str,
    roaster_catalogs: list[dict[str, Any]],
    failed_roasters: list[str],
    log: logging.Logger,
) -> tuple[int, int]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    combined = build_combined_catalog(
        run_id=run_id,
        generated_at=generated_at,
        roasters_path=str(settings.roasters_path),
        denylist_path=str(settings.denylist_path),
        roaster_catalogs=roaster_catalogs,
        failed_roasters=failed_roasters,
    )
    combined_path = save_json(
        combined_catalog_path(settings.reports_dir, run_id),
        combined,
    )
    new_products = build_new_products_catalog(combined)
    new_path = save_json(
        new_products_catalog_path(settings.reports_dir, run_id),
        new_products,
    )
    log.info("Saved combined catalog to %s", combined_path)
    log.info("Saved new-products catalog to %s", new_path)
    return (
        int(combined["summary"]["products"]),
        int(combined["summary"]["new_products"]),
    )


async def _run_resume(
    ctx: RunContext,
    roasters: list[RoasterSource],
) -> int:
    settings = ctx.settings
    log = ctx.logger
    roasters_to_retry = collect_resume_targets(
        roasters, settings.reports_dir, ctx.run_id, log
    )
    log.info(
        "Resume mode: found %d missing/failed catalogs out of %d configured roasters.",
        len(roasters_to_retry),
        len(roasters),
    )
    _results, failed_roasters = await _run_roasters(ctx, roasters_to_retry)
    catalog_paths = today_roaster_catalog_paths(settings.reports_dir, ctx.run_id)
    catalogs = load_roaster_catalogs(catalog_paths, log) if catalog_paths else []
    if not catalogs:
        log.error("Resume mode: no readable roaster catalogs found for %s.", ctx.run_id)
        return 1
    missing_roasters = collect_missing_roaster_names(
        roasters, settings.reports_dir, ctx.run_id
    )
    if missing_roasters:
        log.warning(
            "Resume mode: catalogs still missing for %d roasters: %s",
            len(missing_roasters),
            ", ".join(missing_roasters),
        )
    failed = merge_failed_roaster_names(failed_roasters, missing_roasters)
    _write_run_catalogs(settings, ctx.run_id, catalogs, failed, log)
    return 0


async def _run_full(
    ctx: RunContext,
    roasters: list[RoasterSource],
) -> int:
    results, failed_roasters = await _run_roasters(ctx, roasters)
    catalogs = [result.catalog for result in results if result.catalog]
    if not catalogs:
        ctx.logger.error("No roaster catalogs were written for %s.", ctx.run_id)
        return 1
    _write_run_catalogs(ctx.settings, ctx.run_id, catalogs, failed_roasters, ctx.logger)
    return 0


async def run(settings: Settings) -> int:
    setup_logging(settings.log_level, settings.log_path, settings.log_format)
    log = logging.getLogger("coffee_watch")

    settings.assets_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Coffee Watch scrape-only run: roasters=%s output=%s fetch_product_pages=%s",
        settings.roasters_path,
        settings.reports_dir,
        settings.fetch_product_pages,
    )

    roasters = load_roasters(settings, log)
    if not roasters:
        log.error("No roasters configured; exiting.")
        return 1

    denylist = load_denylist(settings.denylist_path)
    if denylist:
        log.info("Loaded %d denylisted domains.", len(denylist))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d")
    timeout = httpx.Timeout(settings.http_timeout_s)
    limiter = PerHostLimiter(
        per_host=settings.per_host_concurrency,
        global_cap=settings.http_concurrency,
    )

    t_start = time.monotonic()
    exit_code = 1
    seen_products = SeenProducts(settings.seen_db_path, log)
    try:
        async with httpx.AsyncClient(
            http2=True,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=timeout,
        ) as http_client:
            ctx = RunContext(
                settings=settings,
                logger=log,
                http_client=http_client,
                limiter=limiter,
                seen_products=seen_products,
                run_id=run_id,
                denylist=denylist,
                assets_dir=settings.assets_dir,
            )
            if settings.resume:
                exit_code = await _run_resume(ctx, roasters)
            else:
                exit_code = await _run_full(ctx, roasters)
    finally:
        seen_products.close()

    elapsed = time.monotonic() - t_start
    log.info(
        "Run complete | mode=%s roasters=%d elapsed=%.1fs",
        "resume" if settings.resume else "full",
        len(roasters),
        elapsed,
    )
    return exit_code
