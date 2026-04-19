from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from coffee_watch.classify import (
    NEW_PRODUCTS_WINDOW_DAYS,
    classify_new_products,
    resolve_update_date,
)
from coffee_watch.models import ProductCandidate
from coffee_watch.seen_products import SeenProducts


def _product(
    url: str = "https://example.com/products/x",
    name: str = "x",
    shopify_published_at: str = "",
    shopify_updated_at: str = "",
) -> ProductCandidate:
    return ProductCandidate(
        product_id="pid",
        name=name,
        url=url,
        source="R",
        list_price="",
        list_badge="",
        body_html="",
        variants=(),
        shopify_updated_at=shopify_updated_at,
        shopify_published_at=shopify_published_at,
    )


def _seen(tmp_path: Path) -> SeenProducts:
    return SeenProducts(tmp_path / "seen.db", logging.getLogger("test"))


def test_resolve_update_date_prefers_shopify_published_at():
    product = _product(shopify_published_at="2026-04-10T00:00:00Z")
    when, source = resolve_update_date(product, "", "", "")
    assert source == "shopify_published_at"
    assert when == date(2026, 4, 10)


def test_resolve_update_date_falls_back_to_http_last_modified():
    product = _product()
    when, source = resolve_update_date(
        product, "Wed, 16 Apr 2026 00:00:00 GMT", "", ""
    )
    assert source == "http_last_modified"
    assert when == date(2026, 4, 16)


def test_resolve_update_date_falls_back_to_seen_at():
    product = _product()
    when, source = resolve_update_date(product, "", "", "2026-04-12T00:00:00Z")
    assert source == "seen_at"
    assert when == date(2026, 4, 12)


def test_classify_new_products_marks_inside_window(tmp_path):
    seen = _seen(tmp_path)
    try:
        product = _product(shopify_published_at="2026-04-18T00:00:00Z")
        run_day = date(2026, 4, 18)
        new_urls, by_source, undated, outside = classify_new_products(
            [product],
            run_day,
            seen,
            {product.url: ""},
            {},
            {},
            "shopify",
            window_days=NEW_PRODUCTS_WINDOW_DAYS,
            persist_seen=True,
        )
        assert product.url in new_urls
        assert by_source["shopify_published_at"] == 1
        assert undated == 0
        assert outside == 0
    finally:
        seen.close()


def test_classify_new_products_outside_window(tmp_path):
    seen = _seen(tmp_path)
    try:
        product = _product(shopify_published_at="2026-04-01T00:00:00Z")
        run_day = date(2026, 4, 18)
        new_urls, _, _, outside = classify_new_products(
            [product],
            run_day,
            seen,
            {product.url: ""},
            {},
            {},
            "shopify",
            persist_seen=True,
        )
        assert product.url not in new_urls
        assert outside == 1
    finally:
        seen.close()


def test_description_edit_does_not_reset_seen_at(tmp_path):
    seen = _seen(tmp_path)
    try:
        url = "https://example.com/products/x"
        product_v1 = _product(url=url, name="x")
        run_day = date(2026, 4, 10)
        classify_new_products(
            [product_v1],
            run_day,
            seen,
            {url: "original description"},
            {},
            {},
            "generic",
            persist_seen=True,
        )
        # Next run, same URL, different description: classifier should reuse
        # the earlier first_seen_at (now outside the window).
        later_run_day = date(2026, 4, 30)
        product_v2 = _product(url=url, name="x")
        new_urls, by_source, undated, outside = classify_new_products(
            [product_v2],
            later_run_day,
            seen,
            {url: "reworded description"},
            {},
            {},
            "generic",
            persist_seen=False,
        )
        assert url not in new_urls
        assert outside == 1
    finally:
        seen.close()
