from __future__ import annotations

import argparse

import pytest

from coffee_watch.config import ConfigError, Settings, build_settings


def _args(**overrides):
    ns = argparse.Namespace()
    for field in Settings.__dataclass_fields__:
        setattr(ns, field, None)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_defaults_pass_validation():
    settings = build_settings(_args(), {})
    assert settings.fetch_product_pages is True
    assert settings.resume is False
    assert settings.log_format == "text"


def test_cli_overrides_config_and_defaults():
    settings = build_settings(
        _args(http_concurrency=3),
        {"http_concurrency": 1},
    )
    assert settings.http_concurrency == 3


def test_config_alias_for_output_dir():
    settings = build_settings(_args(), {"output_dir": "out"})
    assert str(settings.reports_dir) == "out"


def test_string_bool_config_values():
    settings = build_settings(
        _args(),
        {"fetch_product_pages": "false", "resume": "true"},
    )
    assert settings.fetch_product_pages is False
    assert settings.resume is True


def test_invalid_log_format_raises():
    with pytest.raises(ConfigError):
        build_settings(_args(log_format="xml"), {})


def test_jitter_range_validation():
    with pytest.raises(ConfigError):
        build_settings(_args(jitter_min_s=5.0, jitter_max_s=1.0), {})


def test_invalid_http_concurrency_raises():
    with pytest.raises(ConfigError):
        build_settings(_args(http_concurrency=0), {})
