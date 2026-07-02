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
    assert item["price_variant"] == "12oz"
    assert item["price_label"] == "12oz = $22.00"
    assert item["price_source"] == "variant"
    assert item["availability"] == "available"
    assert item["origin"] == "Ethiopia"
    assert item["process"] == "Washed"
    assert item["tasting_notes"] == ["jasmine", "peach", "black tea"]


def test_catalog_product_splits_slash_separated_tasting_notes():
    product = ProductCandidate(
        product_id="p1",
        name="Ethiopia Test",
        url="https://example.com/products/ethiopia-test",
        source="R",
        variants=(VariantInfo("12oz", "22.00", 340, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com"),
        raw_product_text="Tasting notes: jasmine/peach / black tea.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=True,
        date_source="seen_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["tasting_notes"] == ["jasmine", "peach", "black tea"]


def test_catalog_product_prefers_available_variant_over_list_price():
    product = ProductCandidate(
        product_id="p1",
        name="Variant Coffee",
        url="https://example.com/products/variant",
        source="R",
        list_price="$9.00",
        variants=(
            VariantInfo("125 g", "9.00", 125, False),
            VariantInfo("250 g", "34.25", 250, True),
        ),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="Tasting notes: lychee, caramel.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price"] == "34.25"
    assert item["price_variant"] == "250 g"
    assert item["price_label"] == "250 g = $34.25"
    assert item["price_source"] == "variant"


def test_catalog_product_extracts_pipe_separated_singular_tasting_note():
    product = ProductCandidate(
        product_id="p1",
        name="Pipe Notes Coffee",
        url="https://example.com/product-page/pipe-notes",
        source="R",
        variants=(),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="wix"),
        raw_product_text=(
            "Description: Tasting note | Floral, Berries, Plum, Red Grape "
            "This new micro-lot is delicate."
        ),
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="wix_lastmod",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["tasting_notes"] == ["Floral", "Berries", "Plum", "Red Grape"]


def test_catalog_product_formats_list_price_once_when_no_variants():
    product = ProductCandidate(
        product_id="p1",
        name="List Price Coffee",
        url="https://example.com/products/list-price",
        source="R",
        list_price="$22.00",
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="wix"),
        raw_product_text="Tasting notes: orange, honey.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="seen_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price"] == "$22.00"
    assert item["price_label"] == "$22.00"
    assert item["price_source"] == "list_price"


def test_catalog_product_treats_wix_html_product_without_variants_as_available():
    product = ProductCandidate(
        product_id="p1",
        name="Wix Coffee",
        url="https://example.com/product-page/wix-coffee",
        source="R",
        list_price="",
        variants=(),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="wix"),
        raw_product_text="Tasting notes: mango, longan.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="wix_lastmod",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["availability"] == "available"


def test_catalog_product_honors_sold_out_badge_without_variants():
    product = ProductCandidate(
        product_id="p1",
        name="Sold Out Wix Coffee",
        url="https://example.com/product-page/sold-out-wix-coffee",
        source="R",
        list_badge="Sold Out",
        variants=(),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="wix"),
        raw_product_text="Tasting notes: apple.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="wix_lastmod",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["availability"] == "unavailable"


def test_catalog_product_omits_default_title_from_price_label():
    product = ProductCandidate(
        product_id="p1",
        name="Single Variant Coffee",
        url="https://example.com/products/single-variant",
        source="R",
        variants=(VariantInfo("Default Title", "53.00", 0, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="Tasting notes: raspberry, jasmine.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price"] == "53.00"
    assert item["price_variant"] == ""
    assert item["price_label"] == "$53.00"
    assert item["price_source"] == "variant"


def test_catalog_product_extracts_notes_of_pattern():
    product = ProductCandidate(
        product_id="p1",
        name="Notes Of Coffee",
        url="https://example.com/products/notes-of",
        source="R",
        variants=(VariantInfo("250 g", "48.50", 250, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text=(
            "Costa Rica San Isidro $48.50 Notes of Plum, Orange, Jasmine "
            "Washed Process Johel established the farm in 2014."
        ),
        first_seen_at="2026-06-09T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 6, 9),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["tasting_notes"] == ["Plum", "Orange", "Jasmine"]


def test_catalog_product_extracts_profile_of_pattern():
    product = ProductCandidate(
        product_id="p1",
        name="Profile Coffee",
        url="https://example.com/products/profile-of",
        source="R",
        variants=(VariantInfo("100 g", "54.00", 100, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text=(
            "Processed using the traditional washed method, to produce that "
            "classic profile of citrus, florals, and elegance, that we all expect."
        ),
        first_seen_at="2026-06-09T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 6, 9),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["tasting_notes"] == ["citrus", "florals", "elegance"]


def test_catalog_product_ignores_single_note_of_phrase():
    product = ProductCandidate(
        product_id="p1",
        name="Story Coffee",
        url="https://example.com/products/story",
        source="R",
        variants=(VariantInfo("250 g", "20.00", 250, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="He took careful notes of the harvest that season.",
        first_seen_at="2026-06-09T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 6, 9),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["tasting_notes"] == []


def test_catalog_product_uses_api_grams_for_default_title_variant():
    product = ProductCandidate(
        product_id="p1",
        name="Guest Roaster Coffee",
        url="https://example.com/products/guest-roaster",
        source="R",
        variants=(VariantInfo("Default Title", "56.00", 250, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="Tasting notes: lychee, jasmine.",
        first_seen_at="2026-06-09T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 6, 9),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price"] == "56.00"
    assert item["price_variant"] == "250 g"
    assert item["price_label"] == "250 g = $56.00"
    assert item["price_source"] == "variant_api_grams"


def test_catalog_product_ignores_implausible_api_grams_on_default_title():
    product = ProductCandidate(
        product_id="p1",
        name="Odd Grams Coffee",
        url="https://example.com/products/odd-grams",
        source="R",
        variants=(VariantInfo("Default Title", "24.00", 5, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="Tasting notes: cherry.",
        first_seen_at="2026-06-09T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 6, 9),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price_variant"] == ""
    assert item["price_label"] == "$24.00"
    assert item["price_source"] == "variant"


def test_catalog_product_does_not_price_storefront_unavailable_product():
    product = ProductCandidate(
        product_id="p1",
        name="Hidden Coffee",
        url="https://example.com/products/hidden",
        source="R",
        variants=(
            VariantInfo("125 g", "12.00", 125, True),
            VariantInfo("250 g", "36.00", 250, True),
        ),
        storefront_status="storefront_unavailable",
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="Tasting notes: mango, lychee.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=["variant page returned status 401"],
    )

    assert item["price"] == ""
    assert item["price_variant"] == ""
    assert item["price_label"] == ""
    assert item["price_source"] == "storefront_unavailable"
    assert item["availability"] == "unavailable"
    assert item["storefront_status"] == "storefront_unavailable"


def test_catalog_product_uses_smallest_storefront_visible_variant():
    product = ProductCandidate(
        product_id="p1",
        name="Reserve Coffee",
        url="https://example.com/products/reserve",
        source="R",
        variants=(
            VariantInfo("125 g", "9.00", 125, True),
            VariantInfo("250 g", "34.25", 250, True),
            VariantInfo("2 lb", "92.25", 907, True),
        ),
        visible_variant_titles=("250 g", "2 lb"),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="Tasting notes: lychee, green apple, caramel.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price"] == "34.25"
    assert item["price_variant"] == "250 g"
    assert item["price_label"] == "250 g = $34.25"
    assert item["price_source"] == "visible_variant"


def test_catalog_product_preserves_visible_premium_small_format():
    product = ProductCandidate(
        product_id="p1",
        name="Premium Coffee",
        url="https://example.com/products/premium",
        source="R",
        variants=(
            VariantInfo("100 g", "38.00", 100, True),
            VariantInfo("250 g", "88.00", 250, True),
            VariantInfo("2lb", "68.00", 907, True),
        ),
        visible_variant_titles=("100 g", "250 g"),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="Tasting notes: mango, jasmine.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price"] == "38.00"
    assert item["price_variant"] == "100 g"
    assert item["price_label"] == "100 g = $38.00"
    assert item["price_source"] == "visible_variant"


def test_catalog_product_skips_unavailable_visible_small_format():
    product = ProductCandidate(
        product_id="p1",
        name="Premium Coffee",
        url="https://example.com/products/premium",
        source="R",
        variants=(
            VariantInfo("100 g", "38.00", 100, False),
            VariantInfo("250 g", "88.00", 250, True),
            VariantInfo("2lb", "248.00", 907, True),
        ),
        visible_variant_titles=("100 g", "250 g", "2lb"),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="Tasting notes: mango, jasmine.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price"] == "88.00"
    assert item["price_variant"] == "250 g"
    assert item["price_label"] == "250 g = $88.00"
    assert item["price_source"] == "visible_variant"


def test_catalog_product_prefers_whole_bean_for_same_weight():
    product = ProductCandidate(
        product_id="p1",
        name="Whole Bean Coffee",
        url="https://example.com/products/whole-bean",
        source="R",
        variants=(
            VariantInfo("250g / Ground for Aeropress", "24.00", 250, True),
            VariantInfo("250g / Drip", "24.00", 250, True),
            VariantInfo("250g / Whole Bean", "24.00", 250, True),
            VariantInfo("1kg / Whole Bean", "80.00", 1000, True),
        ),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com", platform="shopify"),
        raw_product_text="Tasting notes: apple, honey.",
        first_seen_at="2026-04-18T00:00:00+00:00",
        is_new=False,
        date_source="shopify_published_at",
        update_date=date(2026, 4, 18),
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price"] == "24.00"
    assert item["price_variant"] == "250g / Whole Bean"
    assert item["price_label"] == "250g / Whole Bean = $24.00"
    assert item["price_source"] == "variant"


def test_catalog_product_keeps_existing_currency_symbol_in_price_label():
    product = ProductCandidate(
        product_id="p1",
        name="Kieni",
        url="https://example.com/products/kieni",
        source="R",
        variants=(VariantInfo("250 g", "€18.50", 250, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com"),
        raw_product_text="",
        first_seen_at="",
        is_new=False,
        date_source="unknown",
        update_date=None,
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    # A price already carrying a currency marker must not gain a second one.
    assert item["price_label"] == "250 g = €18.50"


def test_catalog_product_labels_bare_numeric_price_as_usd():
    product = ProductCandidate(
        product_id="p1",
        name="Auction Lot",
        url="https://example.com/products/auction-lot",
        source="R",
        variants=(VariantInfo("250 g", "1050.00", 250, True),),
    )
    item = catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com"),
        raw_product_text="",
        first_seen_at="",
        is_new=False,
        date_source="unknown",
        update_date=None,
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )

    assert item["price_label"] == "250 g = $1050.00"


def _catalog_item(raw_product_text: str) -> dict:
    product = ProductCandidate(
        product_id="p1",
        name="Spec Sheet Coffee",
        url="https://example.com/products/spec-sheet-coffee",
        source="R",
        variants=(VariantInfo("250 g", "20.00", 250, True),),
    )
    return catalog_product_from_candidate(
        product=product,
        roaster=RoasterSource("R", "https://example.com"),
        raw_product_text=raw_product_text,
        first_seen_at="",
        is_new=False,
        date_source="unknown",
        update_date=None,
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )


def test_catalog_product_stops_notes_at_next_spec_field():
    # Spec-sheet pages run notes straight into the next labeled field with no
    # punctuation; the field text must not leak into tasting notes.
    item = _catalog_item(
        "Tasting Notes: Raspberries, Peach, Key Lime Origin: El Paraiso, Cauca, "
        "Colombia Variety: Castillo Producer: Samuel Diego Bermudez Elevation: 1960 masl"
    )

    assert item["tasting_notes"] == ["Raspberries", "Peach", "Key Lime"]
    assert item["origin"] == "El Paraiso, Cauca, Colombia"


def test_catalog_product_stops_notes_at_components_field():
    item = _catalog_item(
        "Tasting Notes: Berry, Chocolate, Creamy Current Components: 60% Washed Colombia"
    )

    assert item["tasting_notes"] == ["Berry", "Chocolate", "Creamy"]


def test_catalog_product_rejects_prose_behind_notes_label():
    # A generic "Notes:" label on a subscription page is not flavor data.
    item = _catalog_item(
        "Notes: Please purchase your subscription separately from other products"
    )

    assert item["tasting_notes"] == []


def test_catalog_product_trims_notes_at_prose_sentence():
    item = _catalog_item(
        "Tasting notes | Strawberry syrup, Honeydew, Pomegranate, Violet, Dates "
        "We are honored and excited to reintroduce you to a very special farm"
    )

    assert item["tasting_notes"] == [
        "Strawberry syrup",
        "Honeydew",
        "Pomegranate",
        "Violet",
        "Dates",
    ]


def test_catalog_product_stops_process_at_next_spec_field():
    item = _catalog_item(
        "Process: Double fermentation thermal shock Elevation: 1930 masl"
    )

    assert item["process"] == "Double fermentation thermal shock"
