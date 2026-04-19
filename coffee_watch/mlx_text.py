"""Text-sanitization helpers for MLX model output.

Factored out of :mod:`coffee_watch.llm_backend` so tests and other callers can
import these helpers without pulling in ``google.genai`` or ``httpx``.
"""

from __future__ import annotations

import re
from typing import Any

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_CLOSE_TAG_RE = re.compile(r"</think>", re.IGNORECASE)
LEADING_REASONING_LABEL_RE = re.compile(
    r"^\s*(thinking\s*process|thinking|reasoning|analysis)\s*:\s*",
    re.IGNORECASE,
)
ANALYZE_REQUEST_RE = re.compile(r"analy[sz]e\s*the\s*request", re.IGNORECASE)


def flatten_chat_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text")
        return text.strip() if isinstance(text, str) else ""
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                chunks.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks).strip()


def extract_stream_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            chunks.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def trim_to_markdown_answer(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if index == 0:
            continue
        if line.lstrip().startswith("#"):
            candidate = "\n".join(lines[index:]).strip()
            if candidate:
                return candidate
    return ""


def looks_like_reasoning_only(text: str) -> bool:
    compact = re.sub(r"\s+", "", text[:2000]).lower()
    if compact.startswith(
        ("thinkingprocess:", "thinking:", "reasoning:", "analysis:")
    ):
        return True
    return "analyzetherequest" in compact


def sanitize_mlx_text(text: str) -> str:
    cleaned = text
    if THINK_CLOSE_TAG_RE.search(cleaned):
        cleaned = THINK_CLOSE_TAG_RE.split(cleaned)[-1]
    cleaned = THINK_TAG_RE.sub("", cleaned).strip()
    if not cleaned:
        return ""
    if LEADING_REASONING_LABEL_RE.match(cleaned) or ANALYZE_REQUEST_RE.search(
        cleaned[:500]
    ):
        recovered = trim_to_markdown_answer(cleaned)
        if recovered:
            cleaned = recovered
        else:
            return ""
    return cleaned.strip()


def extract_stream_delta_text(choice: dict[str, Any]) -> str:
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        text = extract_stream_content_text(content)
        if text:
            return text
    message = choice.get("message")
    if isinstance(message, dict):
        return extract_stream_content_text(message.get("content"))
    return ""


__all__ = [
    "flatten_chat_message_content",
    "extract_stream_content_text",
    "extract_stream_delta_text",
    "looks_like_reasoning_only",
    "sanitize_mlx_text",
    "trim_to_markdown_answer",
]
