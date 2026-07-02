from __future__ import annotations

import json
import logging
from dataclasses import replace

import pytest

from coffee_watch.config import ConfigError, Settings
from coffee_watch.parsing import (
    bool_from_config,
    load_roasters,
    parse_items_path,
    parse_pagination,
    parse_variants,
    product_matches_filters,
    resolve_product_url,
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


def test_parse_variants_prefers_title_weight_when_shopify_grams_is_wrong():
    variants = parse_variants(
        [
            {"title": "8oz", "price": "28.00", "grams": 227, "available": True},
            {"title": "2lb", "price": "90.00", "grams": 907, "available": True},
            {"title": "5lb", "price": "195.00", "grams": 227, "available": True},
        ]
    )

    assert [variant.grams for variant in variants] == [227, 907, 2268]


def test_load_roasters_treats_string_false_enabled_as_disabled(tmp_path):
    path = tmp_path / "roasters.json"
    path.write_text(
        '[{"name": "Off", "base_url": "https://off.test", "enabled": "false"}]',
        encoding="utf-8",
    )

    settings = replace(Settings.defaults(), roasters_path=path)

    assert load_roasters(settings, logging.getLogger("test")) == []


def test_resolve_product_url_anchors_schemeless_relative_urls():
    resolved = resolve_product_url(
        "https://example.com", "products/foo", "", "", None
    )

    assert resolved == "https://example.com/products/foo"


def test_resolve_product_url_passes_absolute_urls_through():
    resolved = resolve_product_url(
        "https://example.com", "https://other.example.com/products/foo", "", "", None
    )

    assert resolved == "https://other.example.com/products/foo"


def test_load_roasters_rejects_unknown_keys(tmp_path):
    roasters_path = tmp_path / "roasters.json"
    roasters_path.write_text(
        json.dumps(
            [
                {
                    "name": "Typo Roaster",
                    "base_url": "https://example.com",
                    "verify_varient_pages": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    settings = replace(Settings.defaults(), roasters_path=roasters_path)

    with pytest.raises(ConfigError, match=r"Typo Roaster.*verify_varient_pages"):
        load_roasters(settings, logging.getLogger("test"))


def test_parse_pagination_rejects_unknown_keys():
    with pytest.raises(ConfigError, match=r"pagination.*max_page"):
        parse_pagination({"max_page": 3}, "Some Roaster")


def test_load_roasters_reports_roaster_name_for_bad_numbers(tmp_path):
    roasters_path = tmp_path / "roasters.json"
    roasters_path.write_text(
        json.dumps(
            [
                {
                    "name": "Bad Jitter",
                    "base_url": "https://example.com",
                    "jitter_multiplier": "fast",
                }
            ]
        ),
        encoding="utf-8",
    )
    settings = replace(Settings.defaults(), roasters_path=roasters_path)

    with pytest.raises(ConfigError, match="jitter_multiplier for Bad Jitter"):
        load_roasters(settings, logging.getLogger("test"))
