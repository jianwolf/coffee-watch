"""Top-level scrape orchestration."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import httpx

from .atom_feed import ATOM_PRICE_SOURCE
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

# Shopify's catalog-API 429s carry no Retry-After, and which stores are gated
# changes minute to minute (docs/shopify-bot-gating.md), so a store that
# refused the first pass may answer a few minutes later. A full run typically
# finishes in a third of its deadline, and that idle remainder is what pays for
# a second attempt.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# The second pass has to be worth making: a cooldown that eats the whole
# remaining budget would re-request a host that just refused us and then
# cancel mid-flight. Unless this much working time survives the cooldown,
# skipping is the polite answer.
RETRY_MIN_WORK_S = 30.0
# How many times a refusing host may be asked again in one run. Pass 1 has
# already spent its internal attempts, and docs/shopify-bot-gating.md sets
# the house rule at backing off after three consecutive 429s, so the ceiling
# stays low on purpose: three asks per host per run, total.
# One retry pass by default. Each pass spends three catalog requests per host
# (http_max_retries), so the whole chain already asks a refusing host seven
# times: three in the first pass, three here, one in the feed pass. A second
# retry pass costs three more for a gain that measured 0 of 3 in one run and
# 2 of 5 in another, so it is available by flag rather than by default.
RETRY_MAX_PASSES = 1
_CATALOG_STATUS_RE = re.compile(r"catalog API returned HTTP (\d{3})")
_RETRY_AFTER_RE = re.compile(r"Retry-After: (\d+)")


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
            "empty, not quiet. Check the log for 'Egress proxy refused' (sandbox "
            "network policy -- the usual signature is every host failing at "
            "robots.txt, which RFC 9309 makes disallow-all), 429 (roaster bot "
            "gating), or DNS failures, before treating this run as evidence.",
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
    deadline: float | None = None,
) -> tuple[list[RoasterScrapeResult], list[str]]:
    if not target_roasters:
        return [], []
    tasks = {
        asyncio.create_task(process_roaster(ctx, roaster)): roaster
        for roaster in target_roasters
    }
    if deadline is None:
        # In settings, 0 means "no deadline configured"; as an argument it means
        # "the budget is spent". Translate the former into the latter's None so
        # the two cannot be confused: `None` is the only way to say "no limit",
        # and a numeric 0 cancels immediately instead of waiting forever.
        deadline = ctx.settings.run_deadline_s or None
    # asyncio.wait rather than gather: when the budget runs out, roasters that
    # already finished are kept and only the stragglers are cancelled. An
    # unattended run would rather have a partial catalog than nothing.
    _done, pending = await asyncio.wait(tasks.keys(), timeout=deadline)
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


def _rate_limited_roasters(
    results: list[RoasterScrapeResult],
    roasters: list[RoasterSource],
) -> list[RoasterSource]:
    """Roasters worth one more attempt, in configuration order.

    Only a retryable status on the catalog API qualifies. A roaster skipped at
    the robots gate is deliberately excluded: honouring robots.txt is the point,
    not an error to retry. So is a sandbox egress refusal, which will not change
    mid-run.
    """
    by_name = {roaster.name: roaster for roaster in roasters}
    targets: list[RoasterSource] = []
    for result in results:
        if result.status.products_found:
            continue
        # A paginated roaster accumulates several diagnostics, so retryability
        # must not depend on which one happens to come first.
        codes = {
            int(code)
            for code in _CATALOG_STATUS_RE.findall(result.status.note or "")
        }
        if not codes & RETRY_STATUSES:
            continue
        roaster = by_name.get(result.status.roaster)
        if roaster is not None:
            targets.append(roaster)
    return targets


def _requested_backoff(result: RoasterScrapeResult) -> float:
    """The largest Retry-After a roaster asked for, in seconds, or 0."""
    values = _RETRY_AFTER_RE.findall(result.status.note or "")
    return max((float(value) for value in values), default=0.0)


def _requested_backoff_for(
    results: list[RoasterScrapeResult],
    targets: list[RoasterSource],
) -> float:
    """The longest wait any roaster still in play asked for."""
    wanted = {roaster.name for roaster in targets}
    return max(
        (
            _requested_backoff(result)
            for result in results
            if result.status.roaster in wanted
        ),
        default=0.0,
    )


def _remaining_budget(settings: Settings, started_at: float) -> float | None:
    """Seconds left of the run deadline, or None when none was declared."""
    if settings.run_deadline_s <= 0 or not started_at:
        return None
    return settings.run_deadline_s - (time.monotonic() - started_at)


async def _retry_rate_limited(
    ctx: RunContext,
    roasters: list[RoasterSource],
    results: list[RoasterScrapeResult],
    failed_roasters: list[str],
    started_at: float,
) -> tuple[list[RoasterScrapeResult], list[str], dict[str, int]]:
    """Re-ask rate-limited roasters, as long as asking keeps working.

    A pass keeps the normal per-request retries. Cutting them to a single
    attempt was tried and measured worse: two runs recovered 4 of 10 roasters
    with the full three attempts, and one run with a single attempt recovered 0
    of 9. The internal attempts carry their own 30-60s backoffs, so they appear
    to be doing the work the cooldown alone does not.

    The loop stops on the first pass that recovers nothing, which is the signal
    that whoever is left is gated rather than throttled, and re-asking a host
    that has refused every time is what docs/shopify-bot-gating.md says not to
    do. Everything is bounded by what the first pass left of the deadline, so
    with no deadline declared there is no idle remainder to spend and nothing
    runs.
    """
    settings = ctx.settings
    log = ctx.logger
    stats = {"retried": 0, "recovered": 0, "passes": 0}
    if not settings.retry_rate_limited_roasters:
        return results, failed_roasters, stats

    recovered_names: set[str] = set()

    for attempt in range(1, max(1, settings.retry_max_passes) + 1):
        targets = _rate_limited_roasters(results, roasters)
        if not targets:
            break

        budget = _remaining_budget(settings, started_at)
        if budget is None:
            log.info(
                "Second pass skipped for %d rate-limited roaster(s): no "
                "--run-deadline-s budget declared.",
                len(targets),
            )
            break

        # Escalating: a host that has refused twice has earned a longer wait.
        cooldown = max(settings.retry_cooldown_s * attempt, _requested_backoff_for(
            results, targets
        ))
        work = budget - cooldown
        # `not (work >= ...)` rather than `<`, so a NaN budget stops the loop.
        if not (work >= RETRY_MIN_WORK_S):
            log.info(
                "Retry pass %d skipped for %d roaster(s): a %.0fs cooldown leaves "
                "%.0fs of the %.0fs budget, under the %.0fs needed to make the "
                "attempt worth sending.",
                attempt,
                len(targets),
                cooldown,
                work,
                budget,
                RETRY_MIN_WORK_S,
            )
            break

        log.info(
            "Retry pass %d: %d roaster(s) rate-limited at the catalog API (%s); "
            "waiting %.0fs of the %.0fs left in the budget.",
            attempt,
            len(targets),
            ", ".join(roaster.name for roaster in targets),
            cooldown,
            budget,
        )
        await asyncio.sleep(cooldown)
        retry_results, _retry_failed = await _run_roasters(
            ctx, targets, deadline=work
        )
        stats["passes"] = attempt
        stats["retried"] = len({roaster.name for roaster in targets} | recovered_names)

        # Keep a retry only where it produced something: a second refusal must
        # not overwrite the first pass's evidence.
        better = {
            result.status.roaster: result
            for result in retry_results
            if result.status.products_found
        }
        if not better:
            log.info(
                "Retry pass %d recovered nothing; the remaining %d roaster(s) look "
                "gated rather than throttled, so stopping here.",
                attempt,
                len(targets),
            )
            break

        log.info(
            "Retry pass %d recovered %d of %d roaster(s): %s",
            attempt,
            len(better),
            len(targets),
            ", ".join(sorted(better)),
        )
        recovered_names |= set(better)
        # Keyed by position, not by name: two config entries sharing a name
        # would otherwise both be replaced by the same retry result.
        merged = list(results)
        for index, result in enumerate(results):
            replacement = better.pop(result.status.roaster, None)
            if replacement is not None:
                merged[index] = replacement
        results = merged

    stats["recovered"] = len(recovered_names)
    remaining_failed = [
        name for name in failed_roasters if name not in recovered_names
    ]
    return results, remaining_failed, stats



async def _fill_from_atom(
    ctx: RunContext,
    roasters: list[RoasterSource],
    results: list[RoasterScrapeResult],
    failed_roasters: list[str],
    started_at: float,
) -> tuple[list[RoasterScrapeResult], list[str], int]:
    """Last resort for roasters the catalog API never let through.

    Runs after the retry passes, not instead of them: the feed carries no
    per-variant size, price or stock, so a gated roaster should spend its full
    attempts on real catalog data before settling for a thinner source. The
    products that come back are marked, and the brief says what is missing.
    """
    settings = ctx.settings
    log = ctx.logger
    if not settings.atom_fallback:
        return results, failed_roasters, 0
    targets = _rate_limited_roasters(results, roasters)
    if not targets:
        return results, failed_roasters, 0

    budget = _remaining_budget(settings, started_at)
    if budget is not None and budget < RETRY_MIN_WORK_S:
        log.info(
            "Atom fallback skipped for %d roaster(s): %.0fs of budget left.",
            len(targets),
            budget,
        )
        return results, failed_roasters, 0

    log.info(
        "Atom fallback: trying the collection feed for %d roaster(s) the catalog "
        "API never allowed (%s).",
        len(targets),
        ", ".join(roaster.name for roaster in targets),
    )
    # Single catalog attempt here: this pass is for the feed, and a host that
    # has refused every earlier attempt should not get three more.
    atom_ctx = replace(
        ctx,
        atom_fallback=True,
        settings=replace(settings, http_max_retries=0),
    )
    atom_results, _atom_failed = await _run_roasters(
        atom_ctx, targets, deadline=budget
    )
    better = {
        result.status.roaster: result
        for result in atom_results
        if result.status.products_found
    }
    if not better:
        log.info("Atom fallback recovered nothing.")
        return results, failed_roasters, 0

    # This pass re-asks the catalog API before falling back, so some roasters
    # recover with full variant data and only the rest are feed-sourced. Credit
    # them separately: calling an API recovery "from the feed" would overstate
    # how much of the catalog is missing its sizes.
    via_feed = {
        name
        for name, result in better.items()
        if any(
            isinstance(product, dict)
            and product.get("price_source") == ATOM_PRICE_SOURCE
            for product in (result.catalog or {}).get("products", [])
        )
    }
    via_api = sorted(set(better) - via_feed)
    if via_feed:
        log.info(
            "Atom fallback: %d roaster(s) recovered from the feed (%s), with no "
            "per-variant size, price or stock.",
            len(via_feed),
            ", ".join(sorted(via_feed)),
        )
    if via_api:
        log.info(
            "Atom fallback: %d roaster(s) recovered with full catalog data after "
            "all (%s); the catalog API allowed them on this attempt.",
            len(via_api),
            ", ".join(via_api),
        )
    recovered = set(better)
    merged = list(results)
    for index, result in enumerate(results):
        replacement = better.pop(result.status.roaster, None)
        if replacement is not None:
            merged[index] = replacement
    remaining_failed = [name for name in failed_roasters if name not in recovered]
    return merged, remaining_failed, len(via_feed)


def _write_run_catalogs(
    settings: Settings,
    run_id: str,
    roaster_catalogs: list[dict[str, Any]],
    failed_roasters: list[str],
    log: logging.Logger,
    elapsed_s: float = 0.0,
    retry_stats: dict[str, int] | None = None,
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
        retry_stats=retry_stats,
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
    try:
        results, failed_roasters, retry_stats = await _retry_rate_limited(
            ctx, roasters, results, failed_roasters, started_at
        )
    except Exception:
        # The retry sits between collecting results and writing them, so a
        # failure here would otherwise throw away a scrape that succeeded.
        ctx.logger.exception(
            "Second pass failed; keeping the first pass's results."
        )
        retry_stats = {"retried": 0, "recovered": 0, "passes": 0}
    try:
        results, failed_roasters, from_atom = await _fill_from_atom(
            ctx, roasters, results, failed_roasters, started_at
        )
        retry_stats = {**retry_stats, "from_atom": from_atom}
    except Exception:
        ctx.logger.exception(
            "Atom fallback failed; keeping what the catalog passes collected."
        )
        retry_stats = {**retry_stats, "from_atom": 0}
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
        retry_stats,
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
