from __future__ import annotations

from pathlib import Path

FORBIDDEN_RUNTIME_MARKERS = (
    "google.generativeai",
    "google-genai",
    "google-generativeai",
    "anthropic",
    "openai",
    "mlx_lm",
    "mlx-lm",
)


def test_scraper_runtime_has_no_model_api_dependencies():
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "coffee_watch").glob("*.py")),
        root / "main.py",
        root / "scrape_coffee.py",
        root / "requirements.txt",
        root / "pyproject.toml",
    ]
    haystack = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)

    for marker in FORBIDDEN_RUNTIME_MARKERS:
        assert marker not in haystack
