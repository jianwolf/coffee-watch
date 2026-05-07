"""Top-level run orchestration.

Responsibilities:
- parse settings & set up logging
- load roasters and denylist
- dispatch to one of three modes (digest-only, resume, full)
- finalize each mode by building and writing digest reports
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .classify import (
    NEW_PRODUCTS_WINDOW_DAYS,
    build_filtered_new_items_for_digest,
)
from .config import Settings
from .constants import USER_AGENT
from .context import RunContext
from .http_limits import PerHostLimiter
from .llm_backend import backend_label, create_llm_backend
from .logging_utils import setup_logging
from .models import RoasterSource
from .parsing import load_denylist, load_roasters
from .prompts import normalize_language
from .report_status import (
    collect_missing_roaster_names,
    collect_resume_targets,
    extract_failed_roasters_from_reports,
    merge_failed_roaster_names,
    today_roaster_report_paths,
)
from .reporting import load_items_for_reports, load_reports_for_digest
from .roaster_pipeline import (
    build_digest_jobs,
    generate_digest_reports,
    process_roaster,
    save_digest_prompts,
)
from .seen_products import SeenProducts

logger = logging.getLogger(__name__)


async def _finalize_digests(
    ctx: RunContext,
    reports: list[tuple[str, str]],
    failed_roasters: list[str],
) -> None:
    settings = ctx.settings
    log = ctx.logger
    filtered_new_items: list[dict[str, Any]] = []
    if settings.new_products_digest:
        filtered_new_items = build_filtered_new_items_for_digest(
            reports,
            ctx.run_id,
            settings,
            log,
            "Digest",
            items_loader=lambda r: load_items_for_reports(
                r, settings.reports_dir, ctx.run_id, log
            ),
            window_days=NEW_PRODUCTS_WINDOW_DAYS,
            persist_seen=False,
        )
    digest_jobs = build_digest_jobs(
        reports,
        filtered_new_items,
        ctx.language,
        settings.batch_page_text_max_chars,
        settings.new_products_digest,
        settings.user_ask,
    )
    if not digest_jobs:
        return
    save_digest_prompts(
        digest_jobs,
        ctx.assets_dir,
        settings.reports_dir,
        ctx.run_id,
        settings.save_prompt,
        log,
    )
    if settings.skip_llm:
        log.info("Digest LLM calls skipped by configuration.")
        return
    await generate_digest_reports(
        digest_jobs,
        ctx.llm,
        settings,
        log,
        ctx.run_id,
        failed_roasters=failed_roasters,
    )


async def _run_roasters(
    ctx: RunContext,
    target_roasters: list[RoasterSource],
) -> tuple[list[Path], list[str]]:
    if not target_roasters:
        return [], []
    tasks = [
        asyncio.create_task(process_roaster(ctx, roaster))
        for roaster in target_roasters
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    report_paths: list[Path] = []
    failed_names: list[str] = []
    for roaster, result in zip(target_roasters, results):
        if isinstance(result, Exception):
            ctx.logger.exception(
                "Roaster task for %s raised: %s", roaster.name, result
            )
            failed_names.append(roaster.name)
            continue
        path, _items, status = result
        if path:
            report_paths.append(path)
        if status.status in {"failure", "empty"}:
            failed_names.append(roaster.name)
    return report_paths, failed_names


async def _run_digest_only(ctx: RunContext) -> int:
    settings = ctx.settings
    log = ctx.logger
    report_paths = today_roaster_report_paths(settings.reports_dir, ctx.run_id)
    if not report_paths:
        log.error(
            "Digest-only mode: no reports found for %s in %s",
            ctx.run_id,
            settings.reports_dir,
        )
        return 1
    reports = load_reports_for_digest(report_paths, log)
    if not reports:
        log.error("Digest-only mode: no readable reports found for %s.", ctx.run_id)
        return 1
    failed_roasters = extract_failed_roasters_from_reports(
        reports, settings.reports_dir, ctx.run_id
    )
    await _finalize_digests(ctx, reports, failed_roasters)
    return 0


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
        "Resume mode: found %d missing/failed reports out of %d configured roasters.",
        len(roasters_to_retry),
        len(roasters),
    )
    if roasters_to_retry:
        await _run_roasters(ctx, roasters_to_retry)
    report_paths = today_roaster_report_paths(settings.reports_dir, ctx.run_id)
    reports = load_reports_for_digest(report_paths, log) if report_paths else []
    if not reports:
        log.error("Resume mode: no readable reports found for %s.", ctx.run_id)
        return 1
    missing_roasters = collect_missing_roaster_names(
        roasters, settings.reports_dir, ctx.run_id
    )
    if missing_roasters:
        log.warning(
            "Resume mode: reports still missing for %d roasters: %s",
            len(missing_roasters),
            ", ".join(missing_roasters),
        )
    failed_roasters = merge_failed_roaster_names(
        extract_failed_roasters_from_reports(reports, settings.reports_dir, ctx.run_id),
        missing_roasters,
    )
    if settings.save_report:
        await _finalize_digests(ctx, reports, failed_roasters)
    return 0


async def _run_full(
    ctx: RunContext,
    roasters: list[RoasterSource],
) -> int:
    settings = ctx.settings
    log = ctx.logger
    report_paths, failed_roasters = await _run_roasters(ctx, roasters)
    reports: list[tuple[str, str]] = []
    if report_paths:
        reports = load_reports_for_digest(report_paths, log)
        failed_roasters = merge_failed_roaster_names(
            failed_roasters,
            extract_failed_roasters_from_reports(
                reports, settings.reports_dir, ctx.run_id
            ),
        )
    if settings.save_report:
        await _finalize_digests(ctx, reports, failed_roasters)
    return 0


async def run(settings: Settings) -> int:
    setup_logging(settings.log_level, settings.log_path, settings.log_format)
    log = logging.getLogger("coffee_watch")

    settings.assets_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Using LLM backend: backend=%s roaster=%s digest=%s",
        backend_label(settings),
        settings.model,
        settings.digest_model,
    )
    if settings.llm_backend == "mlx":
        mlx_base_url = (
            f"http://{settings.mlx_host}:{settings.mlx_port}"
            + ("/v1" if settings.mlx_runtime == "lm" else "")
        )
        log.info(
            "MLX server target: runtime=%s model=%s url=%s startup_timeout=%.1fs trust_remote_code=%s",
            settings.mlx_runtime,
            settings.mlx_model,
            mlx_base_url,
            settings.mlx_startup_timeout_s,
            settings.mlx_trust_remote_code,
        )
    log.info(
        "New-products digest: %s",
        "enabled" if settings.new_products_digest else "disabled",
    )
    if settings.user_ask:
        log.info("User ask active: %s", settings.user_ask)

    api_key = os.getenv("GEMINI_API_KEY")
    if settings.llm_backend == "gemini" and not api_key:
        log.warning("No GEMINI_API_KEY found; relying on default SDK auth.")

    roasters = load_roasters(settings, log)
    if not roasters:
        log.error("No roasters configured; exiting.")
        return 1

    raw_language = settings.language
    language = normalize_language(raw_language)
    if raw_language.strip().lower() != language:
        log.info("Normalized language setting from %s to %s", raw_language, language)

    denylist = load_denylist(settings.denylist_path)
    if denylist:
        log.info("Loaded %d denylisted domains.", len(denylist))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d")
    timeout = httpx.Timeout(settings.http_timeout_s)
    limiter = PerHostLimiter(
        per_host=settings.per_host_concurrency,
        global_cap=settings.http_concurrency,
    )

    if settings.digest_only and settings.resume:
        log.info("Both digest-only and resume are enabled; running digest-only mode.")

    t_start = time.monotonic()
    seen_products = SeenProducts(settings.seen_db_path, log)
    llm = create_llm_backend(settings, api_key, log)
    exit_code = 1
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
                llm=llm,
                run_id=run_id,
                language=language,
                denylist=denylist,
                assets_dir=settings.assets_dir,
            )
            if settings.digest_only:
                exit_code = await _run_digest_only(ctx)
            elif settings.resume:
                exit_code = await _run_resume(ctx, roasters)
            else:
                exit_code = await _run_full(ctx, roasters)
    finally:
        try:
            await llm.aclose()
        finally:
            seen_products.close()

    elapsed = time.monotonic() - t_start
    log.info(
        "Run complete | mode=%s roasters=%d elapsed=%.1fs",
        "digest-only" if settings.digest_only else ("resume" if settings.resume else "full"),
        len(roasters),
        elapsed,
    )
    return exit_code
