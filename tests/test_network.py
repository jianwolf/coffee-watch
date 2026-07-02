from __future__ import annotations

import gzip
from dataclasses import replace

import httpx

from coffee_watch.config import Settings
from coffee_watch.models import PaginationConfig, RoasterSource
from coffee_watch.network import (
    RETRY_AFTER_MAX_S,
    _parse_sitemap_xml,
    _retry_delay,
    fetch_products_for_roaster,
    fetch_text_with_jitter,
    robots_allows,
)

from .conftest import make_logger


def _settings(**overrides) -> Settings:
    base = replace(
        Settings.defaults(),
        jitter_min_s=0,
        jitter_max_s=0,
        http_max_retries=0,
    )
    return replace(base, **overrides)


async def test_fetch_returns_body_within_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"hello")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await fetch_text_with_jitter(
            client, "https://example.com/products.json", _settings(), make_logger()
        )

    assert response is not None
    assert response.status_code == 200
    assert response.text == "hello"


async def test_fetch_drops_bodies_larger_than_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 2048)

    settings = _settings(http_max_response_bytes=1024)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await fetch_text_with_jitter(
            client, "https://example.com/products.json", settings, make_logger()
        )

    assert response is None


async def test_fetch_refuses_non_http_scheme():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await fetch_text_with_jitter(
            client, "file:///etc/passwd", _settings(), make_logger()
        )

    assert response is None
    assert calls == []


async def test_fetch_refuses_denylisted_host():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await fetch_text_with_jitter(
            client,
            "https://bad.example.com/products.json",
            _settings(),
            make_logger(),
            denylist={"bad.example.com"},
        )

    assert response is None
    assert calls == []


async def test_fetch_drops_redirects_to_denylisted_host():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "good.example.com":
            return httpx.Response(
                302, headers={"location": "https://bad.example.com/page"}
            )
        return httpx.Response(200, content=b"secret")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        response = await fetch_text_with_jitter(
            client,
            "https://good.example.com/page",
            _settings(),
            make_logger(),
            denylist={"bad.example.com"},
        )

    assert response is None


async def test_fetch_decodes_valid_gzip_in_a_single_request():
    """A legitimately compressed response must not be re-fetched.

    Regression test: the streamed body is already decompressed, so rebuilding
    the response with the original content-encoding header made httpx decode
    it a second time and fail — turning every gzip response into a retry
    with an uncompressed transfer.
    """
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.headers.get("accept-encoding", ""))
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-type": "text/html"},
            content=gzip.compress(b"<html>hello coffee</html>"),
        )

    settings = _settings(http_max_retries=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await fetch_text_with_jitter(
            client, "https://example.com/products.json", settings, make_logger()
        )

    assert response is not None
    assert response.text == "<html>hello coffee</html>"
    assert len(requests) == 1
    assert "identity" not in requests[0]


async def test_fetch_does_not_retry_oversized_bodies():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=b"x" * 2048)

    settings = _settings(http_max_retries=3, http_max_response_bytes=1024)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await fetch_text_with_jitter(
            client, "https://example.com/products.json", settings, make_logger()
        )

    assert response is None
    # A size-cap drop is a policy decision, not a transient failure.
    assert len(calls) == 1


async def test_fetch_retries_decoding_errors_with_identity_encoding():
    seen_encodings: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_encodings.append(request.headers.get("accept-encoding", ""))
        if len(seen_encodings) == 1:
            # Corrupted compressed body, as intermittently served by some CDNs.
            return httpx.Response(
                200, headers={"content-encoding": "gzip"}, content=b"not gzip"
            )
        return httpx.Response(200, content=b"plain body")

    settings = _settings(http_max_retries=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await fetch_text_with_jitter(
            client, "https://example.com/robots.txt", settings, make_logger()
        )

    assert response is not None
    assert response.text == "plain body"
    assert len(seen_encodings) == 2
    assert seen_encodings[1] == "identity"


def test_retry_delay_caps_server_provided_retry_after():
    response = httpx.Response(429, headers={"retry-after": "86400"})
    delay = _retry_delay(0, response, _settings(), 1.0)
    assert delay == RETRY_AFTER_MAX_S


def test_retry_delay_uses_small_retry_after_values():
    response = httpx.Response(429, headers={"retry-after": "3"})
    delay = _retry_delay(0, response, _settings(), 1.0)
    assert delay == 3.0


def test_retry_delay_parses_http_date_retry_after():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    response = httpx.Response(429, headers={"retry-after": format_datetime(when)})
    delay = _retry_delay(0, response, _settings(), 1.0)
    # Allow slack for the wall-clock read between header build and parse.
    assert 20.0 <= delay <= RETRY_AFTER_MAX_S


def test_retry_delay_clamps_past_http_date_retry_after():
    response = httpx.Response(
        429, headers={"retry-after": "Wed, 01 Jan 2020 00:00:00 GMT"}
    )
    settings = _settings(jitter_min_s=0.5, jitter_max_s=1.0)
    delay = _retry_delay(0, response, settings, 1.0)
    # A date in the past still yields at least the politeness floor.
    assert delay == 0.5


async def test_robots_disallows_on_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    cache: dict = {}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        allowed = await robots_allows(
            client,
            "https://example.com/products.json",
            _settings(),
            cache,
            make_logger(),
        )

    assert allowed is False
    # The disallow verdict is cached for the rest of the run.
    assert not cache["https://example.com"].can_fetch("any", "https://example.com/x")


async def test_robots_disallows_on_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        allowed = await robots_allows(
            client,
            "https://example.com/products.json",
            _settings(),
            {},
            make_logger(),
        )

    assert allowed is False


async def test_robots_allows_when_robots_txt_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        allowed = await robots_allows(
            client,
            "https://example.com/products.json",
            _settings(),
            {},
            make_logger(),
        )

    assert allowed is True


async def test_fetch_products_for_roaster_dedupes_across_pages(tmp_path):
    page_one = """
    <html><body>
      <a href="/products/ethiopia">Ethiopia</a>
      <a href="/products/kenya">Kenya</a>
    </body></html>
    """
    # Page 2 repeats a featured product from page 1 plus one new item.
    page_two = """
    <html><body>
      <a href="/products/ethiopia">Ethiopia</a>
      <a href="/products/colombia">Colombia</a>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        if request.url.path == "/shop":
            page = request.url.params.get("page", "1")
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=(page_one if page == "1" else page_two).encode(),
            )
        return httpx.Response(404)

    roaster = RoasterSource(
        "Test Roaster",
        "https://example.com",
        products_path="/shop",
        products_type="html",
        pagination=PaginationConfig(max_pages=2),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        products = await fetch_products_for_roaster(
            client,
            roaster,
            _settings(),
            {},
            tmp_path,
            "20260701",
            None,
            make_logger(),
        )

    urls = [product.url for product in products]
    assert urls == [
        "https://example.com/products/ethiopia",
        "https://example.com/products/kenya",
        "https://example.com/products/colombia",
    ]


def test_parse_sitemap_xml_rejects_doctype_and_entities():
    xml = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE urlset [<!ENTITY x "y">]>'
        '<urlset><url><loc>https://example.com/p</loc>'
        "<lastmod>2026-01-01</lastmod></url></urlset>"
    )
    assert _parse_sitemap_xml(xml) == ({}, [])


def test_parse_sitemap_xml_reads_plain_urlset():
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://example.com/product-page/p</loc>"
        "<lastmod>2026-01-01</lastmod></url></urlset>"
    )
    lastmods, children = _parse_sitemap_xml(xml)
    assert lastmods == {"https://example.com/product-page/p": "2026-01-01"}
    assert children == []
