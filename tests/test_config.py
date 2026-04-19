from __future__ import annotations

import argparse

import pytest

from coffee_watch.config import ConfigError, Settings, build_settings


def _args(**overrides):
    ns = argparse.Namespace()
    # Required args: parse_args sets None for everything not provided.
    for field in Settings.__dataclass_fields__:
        setattr(ns, field, None)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_defaults_pass_validation():
    settings = build_settings(_args(), {})
    assert settings.llm_backend == "gemini"
    assert settings.skip_llm is False
    assert settings.log_format == "text"


def test_cli_overrides_config_and_defaults():
    settings = build_settings(
        _args(llm_backend="mlx", model="x"),
        {"llm_backend": "gemini", "model": "y"},
    )
    assert settings.llm_backend == "mlx"
    assert settings.model == "x"


def test_config_alias_for_skip_llm():
    settings = build_settings(_args(), {"skip_gemini": True})
    assert settings.skip_llm is True


def test_config_alias_for_llm_timeout():
    settings = build_settings(_args(), {"gemini_timeout_s": 123.0})
    assert settings.llm_timeout_s == 123.0


def test_user_ask_list_with_one_item_ok():
    settings = build_settings(_args(), {"user_asks": ["decaf"]})
    assert settings.user_ask == "decaf"


def test_user_ask_list_with_multiple_rejected():
    with pytest.raises(ConfigError):
        build_settings(_args(), {"user_asks": ["a", "b"]})


def test_invalid_llm_backend_raises():
    with pytest.raises(ConfigError):
        build_settings(_args(llm_backend="claude"), {})


def test_invalid_log_format_raises():
    with pytest.raises(ConfigError):
        build_settings(_args(log_format="xml"), {})


def test_jitter_range_validation():
    with pytest.raises(ConfigError):
        build_settings(_args(jitter_min_s=5.0, jitter_max_s=1.0), {})


def test_mlx_backend_inherits_mlx_model():
    settings = build_settings(
        _args(llm_backend="mlx", mlx_model="local/model"),
        {},
    )
    assert settings.model == "local/model"
    assert settings.digest_model == "local/model"
