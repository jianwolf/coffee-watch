from __future__ import annotations

import logging

from coffee_watch.catalog_parsers import _first_occurrence_index, parse_wix_shop_catalog
from coffee_watch.models import RoasterSource


def test_first_occurrence_index_matches_str_find():
    html = (
        "intro /product-page/foo-bar middle /product-page/foo "
        "again /product-page/foo-bar tail /product-page/baz"
    )
    needles = {"/product-page/foo", "/product-page/foo-bar", "/product-page/baz"}

    index = _first_occurrence_index(html, needles)

    for needle in needles:
        assert index[needle] == html.find(needle)


def test_first_occurrence_index_handles_missing_and_empty_needles():
    index = _first_occurrence_index("nothing here", {"", "/product-page/gone"})

    assert index == {}


def test_parse_wix_shop_catalog_extracts_plain_dollar_price():
    html = """
    <a href="/product-page/test-coffee">
      <span data-hook="product-item-title">Test Coffee</span>
      <span>$22.00</span>
    </a>
    """

    products = parse_wix_shop_catalog(
        html,
        "https://example.com",
        RoasterSource("Wix", "https://example.com"),
        10,
        logging.getLogger("test"),
    )

    assert products[0].name == "Test Coffee"
    assert products[0].list_price == "$22.00"
