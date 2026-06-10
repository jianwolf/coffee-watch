from __future__ import annotations

from dataclasses import replace

import httpx

from coffee_watch.catalog import STOREFRONT_UNAVAILABLE_STATUS
from coffee_watch.config import Settings
from coffee_watch.context import RunContext
from coffee_watch.http_limits import PerHostLimiter
from coffee_watch.models import ProductCandidate, RoasterSource, VariantInfo
from coffee_watch.roaster_pipeline import (
    _fetch_missing_product_pages,
    _fetch_storefront_product_pages,
    _with_storefront_purchase_details,
)
from coffee_watch.seen_products import SeenProducts

from .conftest import make_logger


async def test_fetch_storefront_product_pages_checks_single_variant_products(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        if request.url.path == "/products/hidden-single":
            return httpx.Response(401, text="Access Limited")
        if request.url.path == "/products/visible-single":
            return httpx.Response(
                200,
                headers={"last-modified": "Mon, 01 Jun 2026 20:00:00 GMT"},
                text="""
                <html>
                  <head>
                    <script type="application/ld+json">
                    {"@type":"Product","description":"Tasting notes: mango, lychee."}
                    </script>
                  </head>
                  <body><div>$24.00 Price</div><button>100 g</button></body>
                </html>
                """,
            )
        return httpx.Response(404)

    settings = replace(Settings.defaults(), jitter_min_s=0, jitter_max_s=0)
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        ctx = RunContext(
            settings=settings,
            logger=make_logger(),
            http_client=client,
            limiter=PerHostLimiter(per_host=1, global_cap=50),
            seen_products=SeenProducts(tmp_path / "seen.db"),
            run_id="20260601",
            denylist=set(),
            assets_dir=tmp_path,
        )
        roaster = RoasterSource(
            "Test Roaster",
            "https://example.com",
            platform="shopify",
            verify_variant_pages=True,
        )
        products = [
            ProductCandidate(
                product_id="hidden",
                name="Hidden Single",
                url="https://example.com/products/hidden-single",
                source="test",
                variants=(VariantInfo("Default Title", "30.00", 0, True),),
            ),
            ProductCandidate(
                product_id="visible",
                name="Visible Single",
                url="https://example.com/products/visible-single",
                source="test",
                variants=(VariantInfo("Default Title", "24.00", 0, True),),
            ),
        ]

        (
            titles_by_url,
            price_by_url,
            storefront_status_by_url,
            page_text_by_url,
            http_last_modified_by_url,
            errors_by_url,
            fetched_pages,
        ) = await _fetch_storefront_product_pages(ctx, roaster, products)

    assert fetched_pages == 2
    assert (
        storefront_status_by_url["https://example.com/products/hidden-single"]
        == STOREFRONT_UNAVAILABLE_STATUS
    )
    assert "not publicly buyable" in errors_by_url[
        "https://example.com/products/hidden-single"
    ][0]
    assert titles_by_url["https://example.com/products/visible-single"] == ("100 g",)
    assert price_by_url["https://example.com/products/visible-single"] == "$24.00"
    assert "Tasting notes: mango, lychee." in page_text_by_url[
        "https://example.com/products/visible-single"
    ]
    assert (
        http_last_modified_by_url["https://example.com/products/visible-single"]
        == "Mon, 01 Jun 2026 20:00:00 GMT"
    )


async def test_fetch_storefront_product_pages_reads_wix_price_and_bag_size(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        if request.url.path == "/product-page/peru-la-margarita":
            return httpx.Response(
                200,
                text="""
                <html>
                  <body>
                    <main>
                      <h1>PERU - LA MARGARITA GESHA RESERVE</h1>
                      <div>$24.00 Price</div>
                      <div>Tasting notes | White floral, Mango, Mandarin</div>
                      <div>Bag Size*</div>
                      <div>100g</div>
                      <div>Profile*</div>
                      <div>Light (Filter)</div>
                      <div>Grind*</div>
                      <div>Whole bean</div>
                    </main>
                  </body>
                </html>
                """,
            )
        return httpx.Response(404)

    settings = replace(Settings.defaults(), jitter_min_s=0, jitter_max_s=0)
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        ctx = RunContext(
            settings=settings,
            logger=make_logger(),
            http_client=client,
            limiter=PerHostLimiter(per_host=1, global_cap=50),
            seen_products=SeenProducts(tmp_path / "seen.db"),
            run_id="20260601",
            denylist=set(),
            assets_dir=tmp_path,
        )
        roaster = RoasterSource(
            "Memli Coffee Lab",
            "https://example.com",
            platform="wix",
            verify_variant_pages=True,
        )
        products = [
            ProductCandidate(
                product_id="peru",
                name="Peru La Margarita",
                url="https://example.com/product-page/peru-la-margarita",
                source="Memli Coffee Lab",
            ),
        ]

        (
            titles_by_url,
            price_by_url,
            storefront_status_by_url,
            page_text_by_url,
            http_last_modified_by_url,
            errors_by_url,
            fetched_pages,
        ) = await _fetch_storefront_product_pages(ctx, roaster, products)

    assert fetched_pages == 1
    assert titles_by_url["https://example.com/product-page/peru-la-margarita"] == (
        "100g",
    )
    assert price_by_url["https://example.com/product-page/peru-la-margarita"] == "$24.00"
    assert "White floral, Mango" in page_text_by_url[
        "https://example.com/product-page/peru-la-margarita"
    ]
    assert storefront_status_by_url == {}
    assert http_last_modified_by_url == {}
    assert errors_by_url == {}


async def test_fetch_missing_product_pages_returns_wix_price_and_bag_size(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        if request.url.path == "/product-page/peru-la-margarita":
            return httpx.Response(
                200,
                headers={"last-modified": "Mon, 01 Jun 2026 20:00:00 GMT"},
                text="""
                <html>
                  <body>
                    <main>
                      <h1>PERU - LA MARGARITA GESHA RESERVE</h1>
                      <div>$24.00 Price</div>
                      <div>Tasting notes | White floral, Mango, Mandarin</div>
                      <div>Bag Size*</div>
                      <div>100g</div>
                    </main>
                  </body>
                </html>
                """,
            )
        return httpx.Response(404)

    settings = replace(Settings.defaults(), jitter_min_s=0, jitter_max_s=0)
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        ctx = RunContext(
            settings=settings,
            logger=make_logger(),
            http_client=client,
            limiter=PerHostLimiter(per_host=1, global_cap=50),
            seen_products=SeenProducts(tmp_path / "seen.db"),
            run_id="20260601",
            denylist=set(),
            assets_dir=tmp_path,
        )
        roaster = RoasterSource(
            "Memli Coffee Lab",
            "https://example.com",
            platform="wix",
        )
        products = [
            ProductCandidate(
                product_id="peru",
                name="Peru La Margarita",
                url="https://example.com/product-page/peru-la-margarita",
                source="Memli Coffee Lab",
            ),
        ]
        page_text_by_id = {product.product_id: "" for product in products}

        (
            http_last_modified_by_url,
            errors_by_url,
            page_price_by_url,
            page_visible_titles_by_url,
            fetched_pages,
        ) = await _fetch_missing_product_pages(
            ctx,
            roaster,
            products,
            page_text_by_id,
        )

    url = "https://example.com/product-page/peru-la-margarita"
    assert fetched_pages == 1
    assert http_last_modified_by_url[url] == "Mon, 01 Jun 2026 20:00:00 GMT"
    assert errors_by_url == {}
    assert page_price_by_url[url] == "$24.00"
    assert page_visible_titles_by_url[url] == ("100g",)
    assert "White floral, Mango" in page_text_by_id["peru"]


def test_with_storefront_purchase_details_builds_page_variant():
    product = ProductCandidate(
        product_id="peru",
        name="Peru La Margarita",
        url="https://example.com/product-page/peru-la-margarita",
        source="Memli Coffee Lab",
    )

    updated = _with_storefront_purchase_details(product, ("100g",), "$24.00", "")

    assert updated.list_price == ""
    assert updated.visible_variant_titles == ("100g",)
    assert updated.variants == (VariantInfo("100g", "24.00", 100, True),)


def test_with_storefront_purchase_details_does_not_price_every_visible_size():
    product = ProductCandidate(
        product_id="peru",
        name="Peru La Margarita",
        url="https://example.com/product-page/peru-la-margarita",
        source="Memli Coffee Lab",
    )

    updated = _with_storefront_purchase_details(
        product,
        ("100g", "250g"),
        "$24.00",
        "",
    )

    assert updated.visible_variant_titles == ("100g", "250g")
    assert updated.variants == (VariantInfo("100g", "24.00", 100, True),)
