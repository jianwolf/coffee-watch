from __future__ import annotations

import httpx
import pytest

from coffee_watch.http_limits import Host429Gate
from coffee_watch.network import fetch_text_with_jitter

from .conftest import make_logger
from .test_network import _settings


def test_gate_requires_positive_threshold():
    with pytest.raises(ValueError):
        Host429Gate(threshold=0)


def test_gate_trips_after_consecutive_429s():
    gate = Host429Gate(threshold=3)
    url = "https://example.com/products/x"
    for _ in range(2):
        gate.record(url, 429)
    assert gate.is_gated(url) is False
    gate.record(url, 429)
    assert gate.is_gated(url) is True
    # Other hosts are unaffected.
    assert gate.is_gated("https://other.example.com/") is False


def test_gate_resets_on_any_non_429_status():
    gate = Host429Gate(threshold=2)
    url = "https://example.com/products/x"
    gate.record(url, 429)
    gate.record(url, 200)
    gate.record(url, 429)
    assert gate.is_gated(url) is False


def test_gate_is_keyed_by_host_not_path():
    gate = Host429Gate(threshold=2)
    gate.record("https://example.com/products/a", 429)
    gate.record("https://example.com/products/b", 429)
    assert gate.is_gated("https://example.com/products/c") is True


async def test_fetch_records_each_attempt_and_skips_gated_hosts():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(429)

    gate = Host429Gate(threshold=3)
    settings = _settings(http_max_retries=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await fetch_text_with_jitter(
            client,
            "https://example.com/products/a",
            settings,
            make_logger(),
            gate=gate,
        )
        # The first URL's three 429 attempts already trip the gate; the next
        # URL on the same host must be skipped without any request.
        second = await fetch_text_with_jitter(
            client,
            "https://example.com/products/b",
            settings,
            make_logger(),
            gate=gate,
        )

    assert first is not None and first.status_code == 429
    assert second is None
    assert len(calls) == 3
    assert gate.is_gated("https://example.com/anything") is True
