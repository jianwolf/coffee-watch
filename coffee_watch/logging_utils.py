"""Logging setup and formatters.

Both the stream handler and the rotating file handler are configured here so
the entry point just calls :func:`setup_logging` once. Supports ``text``
(default) or ``json`` log format.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path

TEXT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class JsonFormatter(logging.Formatter):
    converter = staticmethod(time.gmtime)

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        return json.dumps(payload, ensure_ascii=True)


def _make_formatter(fmt: str) -> logging.Formatter:
    if fmt == "json":
        return JsonFormatter()
    return logging.Formatter(TEXT_FORMAT)


def setup_logging(
    level: str,
    log_path: Path | None = None,
    log_format: str = "text",
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = _make_formatter(log_format)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


__all__ = ["JsonFormatter", "setup_logging"]
