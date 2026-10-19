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
    prune_assets_dir,
    save_json,
    today_roaster_catalog_paths,
)
from .roaster_pipeline import RoasterScrapeResult, process_roaster
from .seen_products import SeenProducts

# A run that collected nothing still writes a valid, empty catalog, so the
# exit code is the only place it can be distinguished from a quiet week. This
# matters most unattended, where nobody reads the log unless something says to.
EMPTY_RUN_EXIT_CODE = 3


def _elapsed_since(started_at: float) -> float:
    return time.monotonic() - started_at if started_at else 0.0


def _report_coverage(
    products: int,
    roaster_count: int,
    failed_roasters: list[str],
    log: logging.Logger,
) -> int:
    """Log what the run actually collected and map it to an exit code."""
    if products == 0:
        log.error(
            "Collected 0 products from %d configured roasters. The catalog is "
            "empty, not quiet: check logs for 'host_not_allowed' (sandbox egress "
            "policy), 429 (bot gating), or robots/DNS failures before treating "
            "this run as evidence.",
            roaster_count,
        )
        return EMPTY_RUN_EXIT_CODE
    if failed_roasters:
        log.warning(
            "Partial coverage: %d of %d roasters returned nothing (%s). Products "
            "collected: %d.",
            len(failed_roasters),
            roaster_count,
            ", ".join(failed_roasters),
            products,
        )
    return 0


async def _run_roasters(
    ctx: RunContext,
    target_roasters: list[RoasterSource],
) -> tuple[list[RoasterScrapeResult], list[str]]:
    if not target_roasters:
        return [], []
    tasks = {
        asyncio.create_task(process_roaster(ctx, roaster)): roaster
        for roaster in target_roasters
    }
    deadline = ctx.settings.run_deadline_s
    # asyncio.wait rather than gather: when the budget runs out, roasters that
    # already finished are kept and only the stragglers are cancelled. An
    # unattended run would rather have a partial catalog than nothing.
    _done, pending = await asyncio.wait(
        tasks.keys(), timeout=deadline if deadline > 0 else None
    )
    if pending:
        ctx.logger.error(
            "Run deadline of %.0fs reached; cancelling %d roaster(s) still in "
            "flight: %s",
            deadline,
            len(pending),
            ", ".join(sorted(tasks[task].name for task in pending)),
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    scrape_results: list[RoasterScrapeResult] = []
    failed_names: list[str] = []
    for task, roaster in tasks.items():
        if task.cancelled():
            failed_names.append(roaster.name)
            continue
        error = task.exception()
        # BaseException, not Exception: a cancelled or hard-failed task must
        # not be mistaken for a scrape result.
        if error is not None:
            ctx.logger.error(
                "Roaster task for %s raised: %s",
                roaster.name,
                error,
                exc_info=error,
            )
            failed_names.append(roaster.name)
            continue
        result = task.result()
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
    elapsed_s: float = 0.0,
) -> tuple[int, int]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    combined = build_combined_catalog(
        run_id=run_id,
        generated_at=generated_at,
        roasters_path=str(settings.roasters_path),
        denylist_path=str(settings.denylist_path),
        roaster_catalogs=roaster_catalogs,
        failed_roasters=failed_roasters,
        elapsed_s=elapsed_s,
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
    started_at: float = 0.0,
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
    products, _new_products = _write_run_catalogs(
        settings, ctx.run_id, catalogs, failed, log, _elapsed_since(started_at)
    )
    return _report_coverage(products, len(roasters), failed, log)


async def _run_full(
    ctx: RunContext,
    roasters: list[RoasterSource],
    started_at: float = 0.0,
) -> int:
    results, failed_roasters = await _run_roasters(ctx, roasters)
    catalogs = [result.catalog for result in results if result.catalog]
    if not catalogs:
        ctx.logger.error("No roaster catalogs were written for %s.", ctx.run_id)
        return 1
    products, _new_products = _write_run_catalogs(
        ctx.settings,
        ctx.run_id,
        catalogs,
        failed_roasters,
        ctx.logger,
        _elapsed_since(started_at),
    )
    return _report_coverage(products, len(roasters), failed_roasters, ctx.logger)


async def run(settings: Settings) -> int:
    setup_logging(settings.log_level, settings.log_path, settings.log_format)
    log = logging.getLogger("coffee_watch")

    settings.assets_dir.mkdir(parents=True, exist_ok=True)
    prune_assets_dir(settings.assets_dir, settings.assets_retention_days, log)
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
                exit_code = await _run_resume(ctx, roasters, t_start)
            else:
                exit_code = await _run_full(ctx, roasters, t_start)
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
