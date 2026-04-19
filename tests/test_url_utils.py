from __future__ import annotations

from coffee_watch.url_utils import (
    build_url_with_params,
    canonicalize_url,
    matches_patterns,
    normalize_base_url,
    safe_slug,
)


def test_normalize_base_url_adds_scheme():
    assert normalize_base_url("example.com") == "https://example.com"


def test_normalize_base_url_strips_trailing_slash():
    assert normalize_base_url("https://example.com/") == "https://example.com"


def test_canonicalize_url_drops_query_and_fragment():
    assert (
        canonicalize_url("https://example.com/p?x=1#frag")
        == "https://example.com/p"
    )


def test_build_url_with_params_appends_to_existing_query():
    url = build_url_with_params(
        "https://example.com", "/products.json?page=1", {"limit": "10"}
    )
    assert "limit=10" in url
    assert url.startswith("https://example.com/products.json?")


def test_matches_patterns_include_and_exclude():
    assert matches_patterns(
        "https://x.com/products/a",
        ("/products/",),
        (),
    )
    assert not matches_patterns(
        "https://x.com/about",
        ("/products/",),
        (),
    )
    assert not matches_patterns(
        "https://x.com/products/gift-card",
        ("/products/",),
        ("gift-card",),
    )


def test_matches_patterns_regex_prefix():
    assert matches_patterns(
        "https://x.com/p/123",
        ("re:/p/\\d+",),
        (),
    )


def test_safe_slug_lowercases_and_hyphenates():
    assert safe_slug("Black & White Roasters") == "black-white-roasters"
    assert safe_slug("   ") == "roaster"
