"""Tests for the Atom feed fallback.

The feed is a degraded source: it carries no per-variant size, price or stock
and is capped at a roaster's newest 25 products. These tests pin both the
parsing and the honesty of the marking, because a feed row presented as a
purchasable size is worse than no row at all.
"""

from __future__ import annotations

from coffee_watch import atom_feed
from coffee_watch.atom_feed import ATOM_PRICE_SOURCE, atom_url_for, parse_atom_feed
from coffee_watch.catalog import catalog_product_from_candidate
from coffee_watch.models import RoasterSource

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:s="http://jadedpixel.com/-/spec/shopify">
  <title>Coffee</title>
  <entry>
    <id>https://example.com/products/9161215017137</id>
    <published>2026-10-07T10:37:52-04:00</published>
    <updated>2026-10-08T11:00:00-04:00</updated>
    <link rel="alternate" type="text/html" href="https://example.com/products/patio-bonito"/>
    <title>Patio Bonito &amp; Friends</title>
    <s:vendor>Test Roaster</s:vendor>
    <summary type="html">
      <![CDATA[<p><strong>Price: </strong> 24.00 - 180.00 <a href="x">(3 variants)</a></p>
<p>Tasting Notes: <strong>Cherry, Lychee, Complex</strong><br>Origin: Colombia</p>]]>
    </summary>
  </entry>
  <entry>
    <id>https://example.com/products/555</id>
    <published>2026-09-29T09:00:00-04:00</published>
    <link rel="alternate" type="text/html" href="https://example.com/products/tegu"/>
    <title>Tegu</title>
    <summary type="html"><![CDATA[<p>No price line here</p>]]></summary>
  </entry>
</feed>
"""


def test_url_is_derived_only_for_shopify_collections():
    assert (
        atom_url_for("https://littlewolf.coffee", "/collections/coffee/products.json")
        == "https://littlewolf.coffee/collections/coffee.atom"
    )
    assert (
        atom_url_for("https://dayglow.coffee", "/collections/craft-coffee-1/products.json")
        == "https://dayglow.coffee/collections/craft-coffee-1.atom"
    )
    # A non-collection path must never be turned into a guessed feed URL.
    assert atom_url_for("https://www.memlicoffee.com", "/shop") == ""
    assert atom_url_for("https://x.com", "/collections/a/b/products.json") == ""
    assert atom_url_for("", "/collections/coffee/products.json") == ""
    assert atom_url_for("https://x.com", "") == ""


def test_parsing_keeps_what_the_feed_actually_has():
    products = parse_atom_feed(FEED, "Test Roaster")
    assert len(products) == 2
    first = products[0]
    assert first.name == "Patio Bonito & Friends", "titles arrive XML-escaped"
    assert first.url == "https://example.com/products/patio-bonito"
    assert first.product_id == "Test Roaster:9161215017137"
    # The lower bound of the range, which is a real quoted price but of an
    # unknown size.
    assert first.list_price == "24.00"
    assert first.shopify_published_at == "2026-10-07T10:37:52-04:00"
    assert first.shopify_updated_at == "2026-10-08T11:00:00-04:00"
    assert "Cherry, Lychee, Complex" in first.body_html
    # The thing the feed cannot give us.
    assert first.variants == ()
    assert products[1].list_price == "", "no price line means no price"


def test_parsing_is_ordered_newest_first_like_the_feed():
    products = parse_atom_feed(FEED, "Test Roaster")
    assert [p.shopify_published_at[:10] for p in products] == ["2026-10-07", "2026-09-29"]


def test_parsing_survives_junk():
    assert parse_atom_feed("", "R") == []
    assert parse_atom_feed("<feed></feed>", "R") == []
    # An entry with no usable link or title is dropped rather than guessed at.
    assert parse_atom_feed("<feed><entry><id>x</id></entry></feed>", "R") == []
    assert (
        parse_atom_feed(
            '<feed><entry><link rel="alternate" href="https://e.com/p/1"/>'
            "<title></title></entry></feed>",
            "R",
        )
        == []
    )


def test_duplicate_urls_collapse():
    doubled = FEED.replace("</feed>", "") + FEED[FEED.index("<entry>") :]
    products = parse_atom_feed(doubled, "Test Roaster")
    assert len({p.url for p in products}) == len(products)


def test_the_catalog_marks_a_feed_product_as_sizeless_and_unknown_stock():
    """The whole point of the marking: no invented size, no invented stock."""
    from dataclasses import replace

    candidate = replace(
        parse_atom_feed(FEED, "Test Roaster")[0], catalog_source=ATOM_PRICE_SOURCE
    )
    item = catalog_product_from_candidate(
        product=candidate,
        roaster=RoasterSource("Test Roaster", "https://example.com"),
        raw_product_text="Tasting Notes: Cherry, Lychee, Complex",
        first_seen_at="",
        is_new=True,
        date_source="shopify_published_at",
        update_date=None,
        http_last_modified="",
        wix_lastmod="",
        errors=[],
    )
    assert item["price_source"] == ATOM_PRICE_SOURCE
    assert item["price"] == "24.00"
    assert item["price_label"] == "$24.00", "a bare price, never '<size> = <price>'"
    assert item["price_variant"] == "", "no size was established"
    assert item["variants"] == []
    # A Shopify product with no variants is unknown stock, not in stock.
    assert item["availability"] == "unknown"


def test_the_brief_flags_a_feed_row_loudly():
    from coffee_watch import brief

    product = {
        "roaster": "Test Roaster",
        "title": "Patio Bonito",
        "product_url": "https://example.com/products/patio-bonito",
        "price": "24.00",
        "price_variant": "",
        "price_label": "$24.00",
        "price_source": ATOM_PRICE_SOURCE,
        "availability": "unknown",
        "storefront_status": "",
        "tasting_notes": ["Cherry"],
        "variants": [],
        "is_new": True,
        "errors": [],
        "first_seen_at": "2026-10-19T00:00:00+00:00",
        "raw_product_text": "",
        "shopify_published_at": "2026-10-07T10:37:52-04:00",
    }
    flags = brief.product_flags(product)
    assert "ATOM-ONLY-no-size-or-stock" in flags
    assert "size-verified" not in flags
    # No size means no $/100g invented from a range's lower bound.
    assert brief.per_100g(product) == ""

    catalog = {
        "run_id": "20261019",
        "generated_at": "2026-10-19T00:00:00+00:00",
        "summary": {"roasters": 1, "products": 1, "new_products": 1, "failed_roasters": []},
        "roasters": [
            {
                "name": "Test Roaster",
                "status": {},
                "summary": {"products_found": 1, "new_products": 1},
                "errors": [],
            }
        ],
        "products": [product],
    }
    header = "\n".join(brief.coverage(catalog))
    assert "atom fallback: 1 row(s)" in header
    assert "NO per-variant size, price or stock" in header
    assert "newest 25" in header, "the cap has to reach the reader too"


def test_the_cap_is_recorded_as_shopifys_limit():
    """?page is ignored by the feed, so there is nothing for us to paginate."""
    assert atom_feed.ATOM_MAX_ENTRIES == 25
    assert "capped" in atom_feed.__doc__
