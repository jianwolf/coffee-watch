from __future__ import annotations

from coffee_watch.mlx_text import looks_like_reasoning_only, sanitize_mlx_text


def test_strip_think_tags():
    assert (
        sanitize_mlx_text("<think>internal</think>\n# Report\nHello")
        == "# Report\nHello"
    )


def test_trailing_think_close_removes_preamble():
    assert (
        sanitize_mlx_text("some reasoning\n</think>\n# Report\nBody")
        == "# Report\nBody"
    )


def test_reasoning_only_preamble_recovers_markdown():
    text = "Thinking: I should consider ...\n\n# Report\nBody"
    assert sanitize_mlx_text(text).startswith("# Report")


def test_reasoning_only_without_markdown_returns_empty():
    text = "Thinking process: no real answer here"
    assert sanitize_mlx_text(text) == ""


def test_looks_like_reasoning_only_detects_analyze_variants():
    assert looks_like_reasoning_only("Let me Analyze the request first")


def test_looks_like_reasoning_only_returns_false_for_plain_content():
    assert not looks_like_reasoning_only("# Report\nBody text")
