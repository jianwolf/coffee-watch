from __future__ import annotations

import logging

from coffee_watch.reporting import extract_coffee_list_items


def test_extract_coffee_list_items_handles_wrapped_multiline_description():
    report = (
        "# Coffee Watch Report\n\n"
        "Roaster: R\n"
        "Generated: ...\n"
        "Run: 20260418\n\n"
        "## Recommendations\n\n"
        "## Coffee list\n\n"
        "- product_id: pid\n"
        "  name: Coffee\n"
        "  url: https://example.com/products/coffee\n"
        "  description:\n"
        "  <UNTRUSTED_SCRAPED_TEXT>\n"
        "Bright mango acidity\n"
        "Floral finish\n"
        "</UNTRUSTED_SCRAPED_TEXT>\n\n"
    )

    items = extract_coffee_list_items([("20260418-r.md", report)], logging.getLogger("t"))

    assert len(items) == 1
    assert items[0]["product_id"] == "pid"
    assert items[0]["description"] == "Bright mango acidity\nFloral finish"
