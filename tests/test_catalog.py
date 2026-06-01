from __future__ import annotations

from datetime import date

from coffee_watch.catalog import catalog_product_from_candidate
from coffee_watch.models import ProductCandidate, RoasterSource, VariantInfo


def test_catalog_product_extracts_labeled_fields():
    product = ProductCandidate(
        product_id="p1",
        name="Ethiopia Test",
        url="https://example.com/products/ethiopia-test",
        source="R",
        list_price="",
        variants=(VariantInfo("12oz", "22.00", 340, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text=(
            "Origin: Ethiopia. Process: Washed. "
            "Tasting notes: jasmine, peach, black tea."
        ),
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=True,
        date_source="seen_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price"] == "22.00"
    assert item["availability"] == "available"
    assert item["origin"] == "Ethiopia"
    assert item["process"] == "Washed"
    assert item["tasting_notes"] == ["jasmine", "peach", "black tea"]
