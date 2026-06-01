from __future__ import annotations

from dataclasses import replace

import httpx

from coffee_watch.catalog import STOREFRONT_UNAVAILABLE_STATUS
from coffee_watch.config import Settings
from coffee_watch.context import RunContext
from coffee_watch.http_limits import PerHostLimiter
from coffee_watch.models import ProductCandidate, RoasterSource, VariantInfo
from coffee_watch.roaster_pipeline import _fetch_storefront_product_pages
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
                  <body><button>100 g</button></body>
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
    assert "Tasting notes: mango, lychee." in page_text_by_url[
        "https://example.com/products/visible-single"
    ]
    assert (
        http_last_modified_by_url["https://example.com/products/visible-single"]
        == "Mon, 01 Jun 2026 20:00:00 GMT"
    )
