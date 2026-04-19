from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SeenProduct:
    hash: str
    url: str
    title: str
    description: str
    first_seen_at: str
    shopify_updated_at: str
    roaster: str
    platform: str


class SeenProducts:
    """SQLite-backed `first-seen` tracker.

    Writes and reads are synchronous; wrap calls from asyncio code via
    :meth:`aget` / :meth:`arecord` which dispatch through ``asyncio.to_thread``
    so the event loop isn't stalled by disk IO.
    """

    def __init__(self, path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.path = path
        self._logger = logger
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_products (
                hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                shopify_updated_at TEXT NOT NULL DEFAULT '',
                roaster TEXT NOT NULL DEFAULT '',
                platform TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._ensure_columns()
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_seen_products_url ON seen_products(url)"
        )
        self._conn.commit()

    def _ensure_columns(self) -> None:
        columns = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(seen_products)")
        }
        additions = {
            "roaster": "TEXT NOT NULL DEFAULT ''",
            "platform": "TEXT NOT NULL DEFAULT ''",
        }
        for name, ddl in additions.items():
            if name in columns:
                continue
            self._conn.execute(
                f"ALTER TABLE seen_products ADD COLUMN {name} {ddl}"
            )

    @staticmethod
    def compute_hash(url: str, title: str, description: str) -> str:
        payload = f"{url.strip()}\n{title.strip()}\n{description.strip()}"
        return hashlib.sha3_512(payload.encode("utf-8")).hexdigest()

    def get(self, hash_value: str) -> Optional[SeenProduct]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT hash, url, title, description, first_seen_at, "
                "shopify_updated_at, roaster, platform "
                "FROM seen_products WHERE hash = ?",
                (hash_value,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return SeenProduct(
            hash=row[0],
            url=row[1],
            title=row[2],
            description=row[3],
            first_seen_at=row[4],
            shopify_updated_at=row[5],
            roaster=row[6],
            platform=row[7],
        )

    def first_seen_for_url(self, url: str) -> str:
        """Return the earliest ``first_seen_at`` for any row matching ``url``.

        This keeps the seen-product semantics stable under description edits:
        a reworded description produces a new hash but we still recognize the
        product by URL.
        """
        cleaned = url.strip()
        if not cleaned:
            return ""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT MIN(first_seen_at) FROM seen_products WHERE url = ? AND first_seen_at != ''",
                (cleaned,),
            )
            row = cursor.fetchone()
        if not row or row[0] is None:
            return ""
        return str(row[0])

    def record(
        self,
        hash_value: str,
        url: str,
        title: str,
        description: str,
        first_seen_at: str,
        shopify_updated_at: str = "",
        roaster: str = "",
        platform: str = "",
    ) -> None:
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO seen_products
                        (hash, url, title, description, first_seen_at, shopify_updated_at, roaster, platform)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(hash) DO UPDATE SET
                        url=excluded.url,
                        title=excluded.title,
                        description=excluded.description,
                        shopify_updated_at=CASE
                            WHEN excluded.shopify_updated_at != ''
                            THEN excluded.shopify_updated_at
                            ELSE seen_products.shopify_updated_at
                        END,
                        roaster=CASE
                            WHEN excluded.roaster != ''
                            THEN excluded.roaster
                            ELSE seen_products.roaster
                        END,
                        platform=CASE
                            WHEN excluded.platform != ''
                            THEN excluded.platform
                            ELSE seen_products.platform
                        END
                    """,
                    (
                        hash_value,
                        url,
                        title,
                        description,
                        first_seen_at,
                        shopify_updated_at,
                        roaster,
                        platform,
                    ),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            if self._logger:
                self._logger.warning(
                    "Failed to record seen product %s: %s", url, exc
                )

    async def aget(self, hash_value: str) -> Optional[SeenProduct]:
        return await asyncio.to_thread(self.get, hash_value)

    async def afirst_seen_for_url(self, url: str) -> str:
        return await asyncio.to_thread(self.first_seen_for_url, url)

    async def arecord(
        self,
        hash_value: str,
        url: str,
        title: str,
        description: str,
        first_seen_at: str,
        shopify_updated_at: str = "",
        roaster: str = "",
        platform: str = "",
    ) -> None:
        await asyncio.to_thread(
            self.record,
            hash_value,
            url,
            title,
            description,
            first_seen_at,
            shopify_updated_at,
            roaster,
            platform,
        )

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except sqlite3.Error as exc:
            if self._logger:
                self._logger.warning("Failed to close seen products DB: %s", exc)
