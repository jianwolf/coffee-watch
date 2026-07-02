from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG_FORMATS = frozenset({"text", "json"})
LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"})


class ConfigError(ValueError):
    """Raised when Settings cannot be constructed from CLI/config inputs."""


@dataclass(frozen=True)
class Settings:
    http_timeout_s: float
    http_total_timeout_s: float
    http_max_response_bytes: int
    http_max_retries: int
    jitter_min_s: float
    jitter_max_s: float
    http_concurrency: int
    per_host_concurrency: int
    sitemap_max_pages: int
    max_products_per_source: int
    new_window_days: int
    page_text_max_chars: int
    log_json_max_chars: int
    fetch_product_pages: bool
    resume: bool
    save_pretty_products_json: bool
    save_raw_products_json: bool
    seen_db_path: Path
    roasters_path: Path
    denylist_path: Path
    reports_dir: Path
    assets_dir: Path
    assets_retention_days: int
    log_path: Path
    log_level: str
    log_format: str

    def __post_init__(self) -> None:
        issues: list[str] = []
        if self.log_format not in LOG_FORMATS:
            issues.append(
                f"log_format must be one of {sorted(LOG_FORMATS)}; got {self.log_format!r}"
            )
        if self.log_level.upper() not in LOG_LEVELS:
            issues.append(
                f"log_level must be one of {sorted(LOG_LEVELS)}; got {self.log_level!r}"
            )
        if self.http_timeout_s <= 0:
            issues.append("http_timeout_s must be > 0")
        if self.http_total_timeout_s < 0:
            issues.append("http_total_timeout_s must be >= 0 (0 disables the deadline)")
        if self.http_max_response_bytes < 0:
            issues.append("http_max_response_bytes must be >= 0 (0 disables the cap)")
        if self.http_max_retries < 0:
            issues.append("http_max_retries must be >= 0")
        if self.jitter_min_s < 0 or self.jitter_max_s < 0:
            issues.append("jitter_min_s and jitter_max_s must be non-negative")
        if self.jitter_min_s > self.jitter_max_s:
            issues.append(
                f"jitter_min_s ({self.jitter_min_s}) must be <= jitter_max_s ({self.jitter_max_s})"
            )
        if self.http_concurrency < 1:
            issues.append("http_concurrency must be >= 1")
        if self.per_host_concurrency < 1:
            issues.append("per_host_concurrency must be >= 1")
        if self.sitemap_max_pages < 1:
            issues.append("sitemap_max_pages must be >= 1")
        if self.new_window_days < 1:
            issues.append("new_window_days must be >= 1")
        if self.max_products_per_source < 1:
            issues.append("max_products_per_source must be >= 1")
        if self.page_text_max_chars < 0:
            issues.append("page_text_max_chars must be >= 0")
        if self.log_json_max_chars < 0:
            issues.append("log_json_max_chars must be >= 0")
        if self.assets_retention_days < 0:
            issues.append("assets_retention_days must be >= 0 (0 disables pruning)")
        if issues:
            raise ConfigError("; ".join(issues))

    @staticmethod
    def defaults() -> Settings:
        return Settings(
            http_timeout_s=20.0,
            http_total_timeout_s=120.0,
            http_max_response_bytes=10_000_000,
            http_max_retries=2,
            jitter_min_s=0.7,
            jitter_max_s=2.0,
            http_concurrency=50,
            per_host_concurrency=1,
            sitemap_max_pages=8,
            max_products_per_source=200,
            new_window_days=7,
            page_text_max_chars=0,
            log_json_max_chars=0,
            fetch_product_pages=True,
            resume=False,
            save_pretty_products_json=False,
            save_raw_products_json=False,
            seen_db_path=Path("logs/seen_products.db"),
            roasters_path=Path("config/roasters.json"),
            denylist_path=Path("config/denylist.txt"),
            reports_dir=Path("reports"),
            assets_dir=Path("logs/assets"),
            assets_retention_days=30,
            log_path=Path("logs/coffee_watch.log"),
            log_level="INFO",
            log_format="text",
        )


def add_bool_flag(
    parser: argparse.ArgumentParser,
    name: str,
    help_text: str,
    default: bool | None,
) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text}"
    )
    parser.set_defaults(**{dest: default})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape specialty coffee roaster catalogs into normalized JSON"
    )
    parser.add_argument("--config", type=Path, help="Path to JSON config file")
    parser.add_argument("--http-timeout-s", type=float, help="HTTP timeout in seconds")
    parser.add_argument(
        "--http-total-timeout-s",
        type=float,
        help="Hard deadline per HTTP request in seconds, including all reads (0 disables)",
    )
    parser.add_argument(
        "--http-max-response-bytes",
        type=int,
        help="Max HTTP response body size in bytes (0 disables the cap)",
    )
    parser.add_argument(
        "--http-max-retries",
        type=int,
        help="Max HTTP retries for transient failures (429/5xx)",
    )
    parser.add_argument("--jitter-min-s", type=float, help="Minimum jitter sleep (s)")
    parser.add_argument("--jitter-max-s", type=float, help="Maximum jitter sleep (s)")
    parser.add_argument(
        "--http-concurrency", type=int, help="Max concurrent HTTP requests (global)"
    )
    parser.add_argument(
        "--per-host-concurrency",
        type=int,
        help="Max concurrent HTTP requests per host",
    )
    parser.add_argument(
        "--sitemap-max-pages",
        type=int,
        help="Max sitemap pages to inspect per roaster",
    )
    parser.add_argument(
        "--max-products-per-source",
        type=int,
        help="Max products to process per roaster",
    )
    parser.add_argument(
        "--new-window-days",
        type=int,
        help="New-product window in days (default 7); raise to match a longer buying cadence, e.g. 30",
    )
    parser.add_argument(
        "--page-text-max-chars",
        type=int,
        help="Max chars for per-page text (0 = no limit)",
    )
    parser.add_argument(
        "--log-json-max-chars",
        type=int,
        help="Max chars when logging products JSON snippets on errors (0 = disable)",
    )
    add_bool_flag(
        parser,
        "fetch-product-pages",
        "fetch product pages when catalog data lacks descriptions",
        None,
    )
    add_bool_flag(
        parser,
        "resume",
        "retry missing/failed roaster catalog outputs for the current UTC day",
        None,
    )
    add_bool_flag(
        parser, "save-pretty-products-json", "save pretty products JSON", None
    )
    add_bool_flag(parser, "save-raw-products-json", "save raw products JSON", None)
    parser.add_argument("--seen-db-path", type=Path, help="Path to SQLite seen DB")
    parser.add_argument("--roasters-path", type=Path, help="Path to roasters JSON")
    parser.add_argument("--denylist-path", type=Path, help="Path to denylist file")
    parser.add_argument("--reports-dir", type=Path, help="Catalog output directory")
    parser.add_argument(
        "--assets-dir", type=Path, help="Directory for raw/pretty scrape assets"
    )
    parser.add_argument(
        "--assets-retention-days",
        type=int,
        help="Days to keep raw payloads in the assets dir (0 disables pruning)",
    )
    parser.add_argument("--log-path", type=Path, help="Log file path")
    parser.add_argument("--log-level", type=str, help="Log level (e.g. INFO)")
    parser.add_argument(
        "--log-format",
        type=str,
        choices=sorted(LOG_FORMATS),
        help="Log format: text (default) or json",
    )
    return parser.parse_args(argv)


def load_config_file(path: Path | None) -> dict[str, Any]:
    """Load the JSON config file, failing closed on any problem.

    An explicitly requested config that is missing or malformed must abort the
    run rather than silently fall back to defaults.
    """
    if path is None:
        return {}
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid config JSON ({path}): {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a JSON object: {path}")
    return data


CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "fetch_product_pages": ("fetch_pages",),
    "reports_dir": ("output_dir",),
}


def _validate_config_keys(config: dict[str, Any]) -> None:
    allowed = set(Settings.__dataclass_fields__) | {
        alias for aliases in CONFIG_ALIASES.values() for alias in aliases
    }
    unknown = sorted(key for key in config if key not in allowed)
    if unknown:
        raise ConfigError(f"Unknown config key(s): {', '.join(unknown)}")


def _as_bool(field_name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(
            f"{field_name} must be a boolean or one of true/false/yes/no/on/off/1/0; "
            f"got {value!r}"
        )
    return bool(value)


def _as_int(field_name: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer; got {value!r}") from exc


def _as_float(field_name: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number; got {value!r}") from exc


def build_settings(args: argparse.Namespace, config: dict[str, Any]) -> Settings:
    _validate_config_keys(config)
    defaults = Settings.defaults()

    def get_config_value(field_name: str) -> Any:
        if field_name in config and config[field_name] is not None:
            return config[field_name]
        for alias in CONFIG_ALIASES.get(field_name, ()):
            if alias in config and config[alias] is not None:
                return config[alias]
        return None

    def pick_value(field_name: str) -> Any:
        value = getattr(args, field_name, None)
        if value is not None:
            return value
        config_value = get_config_value(field_name)
        if config_value is not None:
            return config_value
        return getattr(defaults, field_name)

    def pick_path(field_name: str) -> Path:
        value = pick_value(field_name)
        return value if isinstance(value, Path) else Path(str(value))

    log_level = str(pick_value("log_level")).strip().upper() or defaults.log_level
    log_format = str(pick_value("log_format")).strip().lower() or defaults.log_format

    return Settings(
        http_timeout_s=_as_float("http_timeout_s", pick_value("http_timeout_s")),
        http_total_timeout_s=_as_float(
            "http_total_timeout_s", pick_value("http_total_timeout_s")
        ),
        http_max_response_bytes=_as_int(
            "http_max_response_bytes", pick_value("http_max_response_bytes")
        ),
        http_max_retries=_as_int("http_max_retries", pick_value("http_max_retries")),
        jitter_min_s=_as_float("jitter_min_s", pick_value("jitter_min_s")),
        jitter_max_s=_as_float("jitter_max_s", pick_value("jitter_max_s")),
        http_concurrency=_as_int("http_concurrency", pick_value("http_concurrency")),
        per_host_concurrency=_as_int(
            "per_host_concurrency", pick_value("per_host_concurrency")
        ),
        sitemap_max_pages=_as_int("sitemap_max_pages", pick_value("sitemap_max_pages")),
        max_products_per_source=_as_int(
            "max_products_per_source", pick_value("max_products_per_source")
        ),
        new_window_days=_as_int("new_window_days", pick_value("new_window_days")),
        page_text_max_chars=_as_int(
            "page_text_max_chars", pick_value("page_text_max_chars")
        ),
        log_json_max_chars=_as_int(
            "log_json_max_chars", pick_value("log_json_max_chars")
        ),
        fetch_product_pages=_as_bool(
            "fetch_product_pages", pick_value("fetch_product_pages")
        ),
        resume=_as_bool("resume", pick_value("resume")),
        save_pretty_products_json=_as_bool(
            "save_pretty_products_json", pick_value("save_pretty_products_json")
        ),
        save_raw_products_json=_as_bool(
            "save_raw_products_json", pick_value("save_raw_products_json")
        ),
        seen_db_path=pick_path("seen_db_path"),
        roasters_path=pick_path("roasters_path"),
        denylist_path=pick_path("denylist_path"),
        reports_dir=pick_path("reports_dir"),
        assets_dir=pick_path("assets_dir"),
        assets_retention_days=_as_int(
            "assets_retention_days", pick_value("assets_retention_days")
        ),
        log_path=pick_path("log_path"),
        log_level=log_level,
        log_format=log_format,
    )
