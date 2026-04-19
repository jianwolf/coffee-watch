"""Backwards-compatible re-exports.

Historically ``coffee_watch.llm`` owned the ``CoffeeWatchLLM`` switch class.
That responsibility now lives in :mod:`coffee_watch.llm_backend` (Protocol +
``GeminiBackend`` / ``MLXBackend``). This module re-exports the public
surface callers still use, plus a ``CoffeeWatchLLM`` alias that delegates to
the Protocol-based factory so any external code keeps working.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import Settings
from .gemini import format_grounding_metadata
from .llm_backend import (
    LLMBackend,
    MLXServerError,
    backend_label,
    create_llm_backend,
)
from .mlx_text import looks_like_reasoning_only, sanitize_mlx_text

# Legacy underscore aliases preserved for any external callers.
_looks_like_reasoning_only = looks_like_reasoning_only
_sanitize_mlx_text = sanitize_mlx_text


def CoffeeWatchLLM(  # noqa: N802 - preserved name for back-compat
    settings: Settings,
    api_key: Optional[str],
    logger: logging.Logger,
) -> LLMBackend:
    return create_llm_backend(settings, api_key, logger)


__all__ = [
    "CoffeeWatchLLM",
    "LLMBackend",
    "MLXServerError",
    "backend_label",
    "create_llm_backend",
    "format_grounding_metadata",
    "_looks_like_reasoning_only",
    "_sanitize_mlx_text",
]
