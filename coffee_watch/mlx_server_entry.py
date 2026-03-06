#!/usr/bin/env python3
"""MLX server wrapper to normalize tokenizer outputs for chat templates.

Some mlx_lm/transformers combos return BatchEncoding from apply_chat_template
when tokenize=True. The server expects a list[int], so normalize here before
starting the server process.
"""

from __future__ import annotations

import sys


def _normalize_tokens(tokens):
    if tokens is None:
        return tokens
    try:
        from transformers.tokenization_utils_base import BatchEncoding  # type: ignore
    except Exception:
        BatchEncoding = None  # type: ignore

    if BatchEncoding is not None and isinstance(tokens, BatchEncoding):
        if "input_ids" in tokens:
            tokens = tokens["input_ids"]
        else:
            tokens = getattr(tokens, "input_ids", tokens)
    elif isinstance(tokens, dict) and "input_ids" in tokens:
        tokens = tokens["input_ids"]

    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()

    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        tokens = tokens[0]

    return tokens


def _patch_mlx_server():
    from mlx_lm import server as mlx_server

    original_tokenize = mlx_server.ResponseGenerator._tokenize

    def _tokenize(self, tokenizer, request):
        tokens = original_tokenize(self, tokenizer, request)
        return _normalize_tokens(tokens)

    mlx_server.ResponseGenerator._tokenize = _tokenize
    return mlx_server


def main() -> int:
    mlx_server = _patch_mlx_server()
    return mlx_server.main()


if __name__ == "__main__":
    sys.exit(main())
