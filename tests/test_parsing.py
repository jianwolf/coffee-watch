from __future__ import annotations

import logging
from dataclasses import replace

from coffee_watch.config import Settings
from coffee_watch.parsing import (
    bool_from_config,
    load_roasters,
    parse_items_path,
    parse_pagination,
    parse_variants,
    product_matches_filters,
    to_str_tuple,
)


def test_bool_from_config_handles_common_strings():
    assert bool_from_config("false", default=True) is False
    assert bool_from_config("YES") is True
    assert bool_from_config(None, default=True) is True
    assert bool_from_config("maybe", default=False) is False


def test_parse_pagination_handles_string_stop_on_empty():
    pagination = parse_pagination({"stop_on_empty": "false"})

    assert pagination is not None
    assert pagination.stop_on_empty is False


def test_to_str_tuple_skips_none_and_blank_items():
    assert to_str_tuple([" a ", None, "", "b"], ()) == ("a", "b")


def test_parse_items_path_skips_blank_list_items():
    assert parse_items_path([" data ", "", None, "products"]) == ("data", "products")


def test_product_matches_filters_excludes_title_keywords_case_insensitively():
    item = {"title": "Digital Gift Card", "tags": [], "product_type": "coffee"}

    assert not product_matches_filters(
        item,
        include_tags=set(),
        exclude_tags=set(),
        include_types=set(),
        exclude_types=set(),
        exclude_title_keywords=("gift card",),
    )


def test_parse_variants_accepts_string_grams_and_available_values():
    variants = parse_variants(
        [
            {"title": "12oz", "price": "22.00", "grams": "340", "available": "true"},
            {"title": "Sold Out", "price": "22.00", "grams": "bad", "available": "false"},
        ]
    )

    assert variants[0].grams == 340
    assert variants[0].available is True
    assert variants[1].grams == 0
    assert variants[1].available is False


def test_load_roasters_treats_string_false_enabled_as_disabled(tmp_path):
    path = tmp_path / "roasters.json"
    path.write_text(
        '[{"name": "Off", "base_url": "https://off.test", "enabled": "false"}]',
        encoding="utf-8",
    )

    settings = replace(Settings.defaults(), roasters_path=path)

    assert load_roasters(settings, logging.getLogger("test")) == []
