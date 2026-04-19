"""Run-scoped context bundle.

Collects the objects that travel together through the orchestration code so
individual signatures don't have to pass a dozen parameters each. Not
``frozen=True`` because ``robots_cache`` accumulates entries as the run
progresses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx

from .config import Settings
from .http_limits import PerHostLimiter
from .llm_backend import LLMBackend
from .seen_products import SeenProducts


@dataclass
class RunContext:
    settings: Settings
    logger: logging.Logger
    http_client: httpx.AsyncClient
    limiter: PerHostLimiter
    seen_products: SeenProducts
    llm: LLMBackend
    run_id: str
    language: str
    denylist: set[str]
    assets_dir: Path
    robots_cache: dict[str, RobotFileParser] = field(default_factory=dict)
