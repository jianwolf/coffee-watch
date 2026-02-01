from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CachedPage:
    url: str
    fetched_at: str
    last_modified: str
    etag: str
    status_code: int
    scraped_text: str


class PageCache:
    def __init__(self, path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.path = path
        self._logger = logger
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS page_cache (
                url TEXT PRIMARY KEY,
                fetched_at TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                etag TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                scraped_text TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_page_cache_fetched_at ON page_cache(fetched_at)"
        )
        self._conn.commit()

    def get(self, url: str) -> Optional[CachedPage]:
        cursor = self._conn.execute(
            "SELECT url, fetched_at, last_modified, etag, status_code, scraped_text "
            "FROM page_cache WHERE url = ?",
            (url,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return CachedPage(
            url=row[0],
            fetched_at=row[1],
            last_modified=row[2],
            etag=row[3],
            status_code=int(row[4]),
            scraped_text=row[5],
        )

    def upsert(
        self,
        url: str,
        fetched_at: str,
        last_modified: str,
        etag: str,
        status_code: int,
        scraped_text: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO page_cache
                (url, fetched_at, last_modified, etag, status_code, scraped_text)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                fetched_at=excluded.fetched_at,
                last_modified=excluded.last_modified,
                etag=excluded.etag,
                status_code=excluded.status_code,
                scraped_text=excluded.scraped_text
            """,
            (url, fetched_at, last_modified, etag, int(status_code), scraped_text),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error as exc:
            if self._logger:
                self._logger.warning("Failed to close cache DB: %s", exc)
