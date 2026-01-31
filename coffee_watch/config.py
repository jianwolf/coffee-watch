from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Settings:
    model: str
    gemini_timeout_s: float
    http_timeout_s: float
    jitter_min_s: float
    jitter_max_s: float
    http_concurrency: int
    max_products_per_source: int
    page_text_max_chars: int
    batch_page_text_max_chars: int
    log_json_max_chars: int
    fetch_only: bool
    skip_gemini: bool
    save_prompt: bool
    save_pretty_products_json: bool
    save_raw_products_json: bool
    save_report: bool
    roasters_path: Path
    denylist_path: Path
    reports_dir: Path
    log_path: Path
    log_level: str
    language: str

    @staticmethod
    def defaults() -> "Settings":
        return Settings(
            model="gemini-3-pro-preview",
            gemini_timeout_s=600.0,
            http_timeout_s=20.0,
            jitter_min_s=0.7,
            jitter_max_s=2.0,
            http_concurrency=4,
            max_products_per_source=200,
            page_text_max_chars=0,
            batch_page_text_max_chars=0,
            log_json_max_chars=0,
            fetch_only=False,
            skip_gemini=False,
            save_prompt=False,
            save_pretty_products_json=False,
            save_raw_products_json=False,
            save_report=True,
            roasters_path=Path("config/roasters.json"),
            denylist_path=Path("config/denylist.txt"),
            reports_dir=Path("reports"),
            log_path=Path("logs/coffee_watch.log"),
            log_level="INFO",
            language="zh",
        )


def add_bool_flag(
    parser: argparse.ArgumentParser, name: str, help_text: str, default: Optional[bool]
) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text}"
    )
    parser.set_defaults(**{dest: default})


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coffee Watch monitoring agent")
    parser.add_argument("--config", type=Path, help="Path to JSON config file")
    parser.add_argument("--model", type=str, help="Gemini model ID")
    parser.add_argument(
        "--gemini-timeout-s",
        type=float,
        help="Gemini request timeout in seconds (0 = no timeout)",
    )
    parser.add_argument("--http-timeout-s", type=float, help="HTTP timeout in seconds")
    parser.add_argument("--jitter-min-s", type=float, help="Minimum jitter sleep (s)")
    parser.add_argument("--jitter-max-s", type=float, help="Maximum jitter sleep (s)")
    parser.add_argument(
        "--http-concurrency", type=int, help="Max concurrent HTTP requests"
    )
    parser.add_argument(
        "--max-products-per-source",
        type=int,
        help="Max products to process per roaster",
    )
    parser.add_argument(
        "--page-text-max-chars",
        type=int,
        help="Max chars for per-page text (0 = no limit)",
    )
    parser.add_argument(
        "--batch-page-text-max-chars",
        type=int,
        help="Max chars for batch page text (0 = no limit)",
    )
    parser.add_argument(
        "--log-json-max-chars",
        type=int,
        help="Max chars when logging raw products JSON (0 = no limit)",
    )
    add_bool_flag(parser, "fetch-only", "fetch only (no Gemini calls)", None)
    add_bool_flag(parser, "skip-gemini", "skip Gemini calls", None)
    add_bool_flag(parser, "save-prompt", "save Gemini prompt files", None)
    add_bool_flag(
        parser, "save-pretty-products-json", "save pretty products JSON", None
    )
    add_bool_flag(parser, "save-raw-products-json", "save raw products JSON", None)
    add_bool_flag(parser, "save-report", "save Gemini reports", None)
    parser.add_argument("--roasters-path", type=Path, help="Path to roasters JSON")
    parser.add_argument("--denylist-path", type=Path, help="Path to denylist file")
    parser.add_argument("--reports-dir", type=Path, help="Reports output directory")
    parser.add_argument("--log-path", type=Path, help="Log file path")
    parser.add_argument("--log-level", type=str, help="Log level (e.g. INFO)")
    parser.add_argument(
        "--language", type=str, help="Report language: en or zh (简体中文)"
    )
    return parser.parse_args(argv)


def load_config_file(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        print(f"Config file not found: {path}", file=sys.stderr)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid config JSON ({path}): {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"Config file must contain a JSON object: {path}", file=sys.stderr)
        return {}
    return data


def build_settings(args: argparse.Namespace, config: dict[str, Any]) -> Settings:
    defaults = Settings.defaults()

    def pick_value(field: str) -> Any:
        value = getattr(args, field, None)
        if value is not None:
            return value
        if field in config and config[field] is not None:
            return config[field]
        return getattr(defaults, field)

    def pick_path(field: str) -> Path:
        value = pick_value(field)
        return value if isinstance(value, Path) else Path(str(value))

    return Settings(
        model=str(pick_value("model")),
        gemini_timeout_s=float(pick_value("gemini_timeout_s")),
        http_timeout_s=float(pick_value("http_timeout_s")),
        jitter_min_s=float(pick_value("jitter_min_s")),
        jitter_max_s=float(pick_value("jitter_max_s")),
        http_concurrency=int(pick_value("http_concurrency")),
        max_products_per_source=int(pick_value("max_products_per_source")),
        page_text_max_chars=int(pick_value("page_text_max_chars")),
        batch_page_text_max_chars=int(pick_value("batch_page_text_max_chars")),
        log_json_max_chars=int(pick_value("log_json_max_chars")),
        fetch_only=bool(pick_value("fetch_only")),
        skip_gemini=bool(pick_value("skip_gemini")),
        save_prompt=bool(pick_value("save_prompt")),
        save_pretty_products_json=bool(pick_value("save_pretty_products_json")),
        save_raw_products_json=bool(pick_value("save_raw_products_json")),
        save_report=bool(pick_value("save_report")),
        roasters_path=pick_path("roasters_path"),
        denylist_path=pick_path("denylist_path"),
        reports_dir=pick_path("reports_dir"),
        log_path=pick_path("log_path"),
        log_level=str(pick_value("log_level")),
        language=str(pick_value("language")),
    )
