"""Tests for the hardening an unattended run depends on.

Nobody reads the log of a scheduled run unless something says to, so these
three behaviours are the difference between a degraded run that announces
itself and one that looks like a quiet market: a bounded wall clock, a non-zero
exit when nothing was collected, and an egress refusal that names itself.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx

from coffee_watch import network, runner
from coffee_watch.config import Settings
from coffee_watch.models import RoasterRunStatus, RoasterSource
from coffee_watch.report_status import STATUS_SUCCESS
from coffee_watch.roaster_pipeline import RoasterScrapeResult


def _status(roaster: str, products: int) -> RoasterRunStatus:
    return RoasterRunStatus(
        roaster=roaster,
        run_id="20261018",
        status=STATUS_SUCCESS if products else "empty",
        attempts=1,
        catalog_path=f"20261018-{roaster.lower()}.catalog.json",
        products_found=products,
        new_products=0,
        fetched_pages=0,
        platform="shopify",
    )


def _result(name: str, products: int, tmp_path: Path) -> RoasterScrapeResult:
    return RoasterScrapeResult(
        tmp_path / f"20261018-{name.lower()}.catalog.json",
        {
            "roaster": {"name": name},
            "summary": {"products_found": products, "new_products": 0},
            "products": [{"title": f"{name} coffee"}] * products,
        },
        _status(name, products),
    )


def _ctx(tmp_path: Path, **setting_overrides):
    settings = replace(Settings.defaults(), reports_dir=tmp_path, **setting_overrides)
    return SimpleNamespace(
        settings=settings,
        logger=logging.getLogger("test-unattended"),
        run_id="20261018",
    )


def test_a_run_that_collects_nothing_exits_non_zero(monkeypatch, tmp_path: Path):
    """A fully blocked scrape still writes a valid catalog and must not look fine."""
    ctx = _ctx(tmp_path)

    async def fake_run_roasters(ctx_arg, roasters):
        return [_result("Gated", 0, tmp_path)], ["Gated"]

    monkeypatch.setattr(runner, "_run_roasters", fake_run_roasters)
    exit_code = asyncio.run(runner._run_full(ctx, [RoasterSource("Gated", "https://x")]))

    assert exit_code == runner.EMPTY_RUN_EXIT_CODE
    # The catalog is still written: the evidence of the failure is worth keeping.
    assert (tmp_path / "20261018-catalog.json").exists()


def test_partial_coverage_still_succeeds(monkeypatch, tmp_path: Path):
    ctx = _ctx(tmp_path)

    async def fake_run_roasters(ctx_arg, roasters):
        return [_result("Open", 3, tmp_path), _result("Gated", 0, tmp_path)], ["Gated"]

    monkeypatch.setattr(runner, "_run_roasters", fake_run_roasters)
    exit_code = asyncio.run(
        runner._run_full(
            ctx, [RoasterSource("Open", "https://a"), RoasterSource("Gated", "https://b")]
        )
    )
    # Degradation is the brief's story to tell; only a zero-product run is an error.
    assert exit_code == 0


def test_the_run_deadline_keeps_what_finished_and_cancels_the_rest(tmp_path: Path):
    ctx = _ctx(tmp_path, run_deadline_s=0.2)
    started: list[str] = []

    async def fake_process_roaster(ctx_arg, roaster):
        started.append(roaster.name)
        if roaster.name == "Tarpit":
            await asyncio.sleep(30)
        return _result(roaster.name, 2, tmp_path)

    import coffee_watch.runner as runner_module

    original = runner_module.process_roaster
    runner_module.process_roaster = fake_process_roaster
    try:
        results, failed = asyncio.run(
            runner._run_roasters(
                ctx,
                [RoasterSource("Fast", "https://a"), RoasterSource("Tarpit", "https://b")],
            )
        )
    finally:
        runner_module.process_roaster = original

    assert sorted(started) == ["Fast", "Tarpit"]
    assert [result.status.roaster for result in results] == ["Fast"]
    assert failed == ["Tarpit"]


def test_no_deadline_waits_for_every_roaster(tmp_path: Path):
    ctx = _ctx(tmp_path, run_deadline_s=0.0)

    async def fake_process_roaster(ctx_arg, roaster):
        await asyncio.sleep(0.05)
        return _result(roaster.name, 1, tmp_path)

    import coffee_watch.runner as runner_module

    original = runner_module.process_roaster
    runner_module.process_roaster = fake_process_roaster
    try:
        results, failed = asyncio.run(
            runner._run_roasters(
                ctx, [RoasterSource("A", "https://a"), RoasterSource("B", "https://b")]
            )
        )
    finally:
        runner_module.process_roaster = original

    assert [result.status.roaster for result in results] == ["A", "B"]
    assert failed == []


def test_a_raising_roaster_is_reported_without_killing_the_run(tmp_path: Path):
    ctx = _ctx(tmp_path)

    async def fake_process_roaster(ctx_arg, roaster):
        if roaster.name == "Broken":
            raise RuntimeError("parser blew up")
        return _result(roaster.name, 1, tmp_path)

    import coffee_watch.runner as runner_module

    original = runner_module.process_roaster
    runner_module.process_roaster = fake_process_roaster
    try:
        results, failed = asyncio.run(
            runner._run_roasters(
                ctx,
                [RoasterSource("Fine", "https://a"), RoasterSource("Broken", "https://b")],
            )
        )
    finally:
        runner_module.process_roaster = original

    assert [result.status.roaster for result in results] == ["Fine"]
    assert failed == ["Broken"]


def _response(status: int, headers: dict[str, str]) -> httpx.Response:
    request = httpx.Request("GET", "https://roaster.example/products.json")
    return httpx.Response(status_code=status, headers=headers, request=request)


def test_a_sandbox_egress_refusal_is_named_as_such(caplog):
    log = logging.getLogger("test-proxy")
    with caplog.at_level(logging.ERROR, logger="test-proxy"):
        detected = network.log_proxy_denial(
            _response(403, {network.PROXY_DENY_HEADER: "host_not_allowed"}), log
        )
    assert detected is True
    message = caplog.text
    assert "roaster.example" in message
    assert "host_not_allowed" in message
    assert "sandbox network policy" in message


def test_a_roasters_own_403_is_not_mistaken_for_a_sandbox_block(caplog):
    log = logging.getLogger("test-proxy")
    with caplog.at_level(logging.ERROR, logger="test-proxy"):
        detected = network.log_proxy_denial(_response(403, {}), log)
    assert detected is False
    assert caplog.text == ""


def test_a_refused_connect_is_named_even_without_a_response(caplog):
    """The form that actually fired in a cloud run.

    The proxy rejects the CONNECT, so there is no response and no
    `x-deny-reason` header -- httpx raises ProxyError instead. Logged as a
    generic request failure, this reads as the roaster being unreachable.
    """
    log = logging.getLogger("test-proxy")
    with caplog.at_level(logging.ERROR, logger="test-proxy"):
        network.log_proxy_connect_refusal(
            "https://roaster.example/robots.txt",
            httpx.ProxyError("403 Forbidden"),
            log,
        )
    assert "roaster.example" in caplog.text
    assert "Egress proxy refused" in caplog.text
    assert "network policy" in caplog.text


def test_proxy_error_subclasses_http_error():
    """Why the ProxyError handler has to come first."""
    assert issubclass(httpx.ProxyError, httpx.HTTPError)


def _empty_result(name: str, note: str, tmp_path: Path) -> RoasterScrapeResult:
    status = _status(name, 0)
    status = replace(status, note=note)
    return RoasterScrapeResult(
        tmp_path / f"20261018-{name.lower()}.catalog.json",
        {"roaster": {"name": name}, "summary": {"products_found": 0}, "products": []},
        status,
    )


def test_only_a_retryable_catalog_status_earns_a_second_attempt(tmp_path: Path):
    """Robots and sandbox refusals are excluded on purpose.

    Honouring robots.txt is the point, not an error to retry, and an egress
    policy will not change mid-run.
    """
    roasters = [RoasterSource(n, f"https://{n.lower()}.example") for n in
                ("Gated", "Server", "Robots", "Sandbox", "Notfound", "Fine")]
    results = [
        _empty_result("Gated", "catalog API returned HTTP 429", tmp_path),
        _empty_result("Server", "catalog API returned HTTP 503", tmp_path),
        _empty_result(
            "Robots",
            "robots.txt unreachable or disallows crawling, so no catalog request "
            "was made",
            tmp_path,
        ),
        _empty_result("Sandbox", "blocked by sandbox egress policy", tmp_path),
        _empty_result("Notfound", "catalog API returned HTTP 404", tmp_path),
        _result("Fine", 3, tmp_path),
    ]
    targets = runner._rate_limited_roasters(results, roasters)
    assert [r.name for r in targets] == ["Gated", "Server"]


def test_the_second_pass_needs_a_declared_budget(tmp_path: Path, caplog):
    """No deadline means no idle remainder to spend, so no silent slowdown."""
    ctx = _ctx(tmp_path, run_deadline_s=0.0)
    ctx.logger = logging.getLogger("test-retry-nobudget")
    roasters = [RoasterSource("Gated", "https://gated.example")]
    results = [_empty_result("Gated", "catalog API returned HTTP 429", tmp_path)]

    with caplog.at_level(logging.INFO, logger="test-retry-nobudget"):
        merged, failed, stats = asyncio.run(
            runner._retry_rate_limited(ctx, roasters, results, ["Gated"], 0.0)
        )
    assert stats == {"retried": 0, "recovered": 0}
    assert merged is results and failed == ["Gated"]
    assert "no --run-deadline-s budget declared" in caplog.text


def test_the_second_pass_is_skipped_when_the_budget_is_spent(tmp_path: Path, caplog):
    import time as _time

    ctx = _ctx(tmp_path, run_deadline_s=5.0, retry_cooldown_s=120.0)
    ctx.logger = logging.getLogger("test-retry-notime")
    roasters = [RoasterSource("Gated", "https://gated.example")]
    results = [_empty_result("Gated", "catalog API returned HTTP 429", tmp_path)]
    # Pretend the first pass already used the whole deadline.
    started_at = _time.monotonic() - 10.0

    with caplog.at_level(logging.WARNING, logger="test-retry-notime"):
        _merged, _failed, stats = asyncio.run(
            runner._retry_rate_limited(ctx, roasters, results, ["Gated"], started_at)
        )
    assert stats == {"retried": 0, "recovered": 0}
    assert "not enough for a" in caplog.text


def test_a_recovered_roaster_replaces_its_empty_result(tmp_path: Path):
    ctx = _ctx(tmp_path, run_deadline_s=600.0, retry_cooldown_s=0.0)
    roasters = [
        RoasterSource("Gated", "https://gated.example"),
        RoasterSource("Fine", "https://fine.example"),
    ]
    results = [
        _empty_result("Gated", "catalog API returned HTTP 429", tmp_path),
        _result("Fine", 2, tmp_path),
    ]

    async def fake_process_roaster(ctx_arg, roaster):
        return _result(roaster.name, 4, tmp_path)

    original = runner.process_roaster
    runner.process_roaster = fake_process_roaster
    try:
        merged, failed, stats = asyncio.run(
            runner._retry_rate_limited(
                ctx, roasters, results, ["Gated"], __import__("time").monotonic()
            )
        )
    finally:
        runner.process_roaster = original

    assert stats == {"retried": 1, "recovered": 1}
    assert [r.status.products_found for r in merged] == [4, 2]
    assert failed == [], "a recovered roaster must leave the failed list"


def test_a_second_refusal_does_not_overwrite_the_first_pass(tmp_path: Path):
    """The retry is only ever allowed to improve on what we already had."""
    ctx = _ctx(tmp_path, run_deadline_s=600.0, retry_cooldown_s=0.0)
    roasters = [RoasterSource("Gated", "https://gated.example")]
    results = [_empty_result("Gated", "catalog API returned HTTP 429", tmp_path)]

    async def fake_process_roaster(ctx_arg, roaster):
        return _empty_result(roaster.name, "catalog API returned HTTP 429", tmp_path)

    original = runner.process_roaster
    runner.process_roaster = fake_process_roaster
    try:
        merged, failed, stats = asyncio.run(
            runner._retry_rate_limited(
                ctx, roasters, results, ["Gated"], __import__("time").monotonic()
            )
        )
    finally:
        runner.process_roaster = original

    assert stats == {"retried": 1, "recovered": 0}
    assert merged == results
    assert failed == ["Gated"]


def test_the_second_pass_can_be_turned_off(tmp_path: Path):
    ctx = _ctx(tmp_path, run_deadline_s=600.0, retry_empty_roasters=False)
    roasters = [RoasterSource("Gated", "https://gated.example")]
    results = [_empty_result("Gated", "catalog API returned HTTP 429", tmp_path)]
    _merged, _failed, stats = asyncio.run(
        runner._retry_rate_limited(
            ctx, roasters, results, ["Gated"], __import__("time").monotonic()
        )
    )
    assert stats == {"retried": 0, "recovered": 0}
