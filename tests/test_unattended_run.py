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
