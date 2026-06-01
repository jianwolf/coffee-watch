from __future__ import annotations

from coffee_watch.text_utils import (
    extract_product_page_price,
    extract_product_page_size_labels,
    extract_size_button_labels,
    grams_from_size_label,
)


def test_extract_size_button_labels_reads_visible_size_buttons_only():
    html = """
    <script>{"optionValues":[{"name":"125 g"},{"name":"5 lb"}]}</script>
    <button class="active">250 g</button>
    <button><span>2 lb</span></button>
    <button>Whole Bean</button>
    <button>1</button>
    """

    assert extract_size_button_labels(html) == ("250 g", "2 lb")


def test_extract_size_button_labels_keeps_visible_small_formats():
    html = """
    <button>100 g</button>
    <button>4 oz</button>
    <button>250 g</button>
    """

    assert extract_size_button_labels(html) == ("100 g", "4 oz", "250 g")


def test_extract_size_button_labels_skips_disabled_size_buttons():
    html = """
    <button disabled>100 g</button>
    <button aria-disabled="true">125 g</button>
    <button>250 g</button>
    """

    assert extract_size_button_labels(html) == ("250 g",)


def test_extract_size_button_labels_skips_hidden_size_buttons():
    html = """
    <button hidden>100 g</button>
    <button aria-hidden="true">125 g</button>
    <button style="display: none">4 oz</button>
    <button class="hidden">2 lb</button>
    <button>250 g</button>
    """

    assert extract_size_button_labels(html) == ("250 g",)


def test_extract_product_page_price_reads_wix_price_text():
    html = """
    <main>
      <h1>PERU - LA MARGARITA GESHA RESERVE</h1>
      <div>$24.00 Price</div>
      <div>Tasting notes | White floral, Mango</div>
    </main>
    """

    assert extract_product_page_price(html) == "$24.00"


def test_extract_product_page_size_labels_reads_wix_bag_size_text():
    html = """
    <main>
      <div>Bag Size*</div>
      <div>100g</div>
      <div>Profile*</div>
      <div>Light (Filter)</div>
      <div>Grind*</div>
      <div>Whole bean</div>
    </main>
    """

    assert extract_product_page_size_labels(html) == ("100g",)


def test_grams_from_size_label_converts_common_formats():
    assert grams_from_size_label("100g") == 100
    assert grams_from_size_label("4 oz") == 113
    assert grams_from_size_label("2 lb") == 907
