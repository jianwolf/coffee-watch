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
    shopify_updated_at: str
    html_last_modified: str
    cached_at: str


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
                scraped_text TEXT NOT NULL,
                shopify_updated_at TEXT NOT NULL DEFAULT '',
                html_last_modified TEXT NOT NULL DEFAULT '',
                cached_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_page_cache_fetched_at ON page_cache(fetched_at)"
        )
        self._ensure_columns()
        self._conn.commit()

    def _ensure_columns(self) -> None:
        try:
            rows = self._conn.execute("PRAGMA table_info(page_cache)").fetchall()
        except sqlite3.Error as exc:
            if self._logger:
                self._logger.warning("Failed to inspect cache DB schema: %s", exc)
            return
        existing = {row[1] for row in rows}
        migrations = [
            ("shopify_updated_at", "shopify_updated_at TEXT NOT NULL DEFAULT ''"),
            ("html_last_modified", "html_last_modified TEXT NOT NULL DEFAULT ''"),
            ("cached_at", "cached_at TEXT NOT NULL DEFAULT ''"),
        ]
        for name, ddl in migrations:
            if name in existing:
                continue
            try:
                self._conn.execute(f"ALTER TABLE page_cache ADD COLUMN {ddl}")
            except sqlite3.Error as exc:
                if self._logger:
                    self._logger.warning("Failed to add cache column %s: %s", name, exc)
        try:
            self._conn.execute(
                "UPDATE page_cache "
                "SET html_last_modified = last_modified "
                "WHERE html_last_modified = '' AND last_modified != ''"
            )
            self._conn.execute(
                "UPDATE page_cache "
                "SET cached_at = fetched_at "
                "WHERE cached_at = '' AND fetched_at != ''"
            )
        except sqlite3.Error as exc:
            if self._logger:
                self._logger.warning("Failed to backfill cache metadata: %s", exc)
        self._conn.commit()

    def get(self, url: str) -> Optional[CachedPage]:
        cursor = self._conn.execute(
            "SELECT url, fetched_at, last_modified, etag, status_code, scraped_text, "
            "shopify_updated_at, html_last_modified, cached_at "
            "FROM page_cache WHERE url = ?",
            (url,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        html_last_modified = row[7] or row[2]
        cached_at = row[8] or row[1]
        return CachedPage(
            url=row[0],
            fetched_at=row[1],
            last_modified=row[2],
            etag=row[3],
            status_code=int(row[4]),
            scraped_text=row[5],
            shopify_updated_at=row[6],
            html_last_modified=html_last_modified,
            cached_at=cached_at,
        )

    def upsert(
        self,
        url: str,
        fetched_at: str,
        last_modified: str,
        etag: str,
        status_code: int,
        scraped_text: str,
        shopify_updated_at: str = "",
        html_last_modified: str = "",
        cached_at: Optional[str] = None,
    ) -> None:
        cached_at_value = cached_at or fetched_at
        self._conn.execute(
            """
            INSERT INTO page_cache
                (
                    url,
                    fetched_at,
                    last_modified,
                    etag,
                    status_code,
                    scraped_text,
                    shopify_updated_at,
                    html_last_modified,
                    cached_at
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                fetched_at=excluded.fetched_at,
                last_modified=excluded.last_modified,
                etag=excluded.etag,
                status_code=excluded.status_code,
                scraped_text=excluded.scraped_text,
                shopify_updated_at=CASE
                    WHEN excluded.shopify_updated_at != ''
                    THEN excluded.shopify_updated_at
                    ELSE page_cache.shopify_updated_at
                END,
                html_last_modified=CASE
                    WHEN excluded.html_last_modified != ''
                    THEN excluded.html_last_modified
                    ELSE page_cache.html_last_modified
                END,
                cached_at=CASE
                    WHEN page_cache.cached_at != ''
                    THEN page_cache.cached_at
                    ELSE excluded.cached_at
                END
            """,
            (
                url,
                fetched_at,
                last_modified,
                etag,
                int(status_code),
                scraped_text,
                shopify_updated_at,
                html_last_modified,
                cached_at_value,
            ),
        )
        self._conn.commit()

    def update_shopify_updated_at(self, url: str, shopify_updated_at: str) -> None:
        if not shopify_updated_at:
            return
        try:
            self._conn.execute(
                "UPDATE page_cache SET shopify_updated_at = ? WHERE url = ?",
                (shopify_updated_at, url),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            if self._logger:
                self._logger.warning("Failed to update shopify_updated_at for %s: %s", url, exc)

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error as exc:
            if self._logger:
                self._logger.warning("Failed to close cache DB: %s", exc)
