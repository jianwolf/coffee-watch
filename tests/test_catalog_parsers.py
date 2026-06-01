from __future__ import annotations

import logging

from coffee_watch.catalog_parsers import parse_wix_shop_catalog
from coffee_watch.models import RoasterSource


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
