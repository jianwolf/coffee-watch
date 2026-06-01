"""Run-scoped context bundle for scraping."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx

from .config import Settings
from .http_limits import PerHostLimiter
from .seen_products import SeenProducts


@dataclass
class RunContext:
    settings: Settings
    logger: logging.Logger
    http_client: httpx.AsyncClient
    limiter: PerHostLimiter
    seen_products: SeenProducts
    run_id: str
    denylist: set[str]
    assets_dir: Path
    robots_cache: dict[str, RobotFileParser] = field(default_factory=dict)
