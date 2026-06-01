from __future__ import annotations

from coffee_watch.text_utils import extract_size_button_labels


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
