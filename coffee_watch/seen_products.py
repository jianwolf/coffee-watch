from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from pathlib import Path


class SeenProducts:
    """SQLite-backed `first-seen` tracker.

    All methods are synchronous; asyncio code should call into this class
    from a worker thread (``asyncio.to_thread``). The connection permits
    cross-thread use and an internal lock serializes access.
    """

    def __init__(self, path: Path, logger: logging.Logger | None = None) -> None:
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

    def first_seen_for_urls(self, urls: list[str]) -> dict[str, str]:
        """Return ``url -> earliest first_seen_at`` for the given URLs.

        Matching by URL keeps the seen-product semantics stable under
        description edits: a reworded description produces a new hash but the
        product is still recognized. Only URLs with at least one non-empty
        ``first_seen_at`` row appear in the result; absent URLs are unknown —
        callers should treat them as missing.
        """
        cleaned = [u.strip() for u in urls if u and u.strip()]
        if not cleaned:
            return {}
        unique = list(dict.fromkeys(cleaned))
        out: dict[str, str] = {}
        # SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999 (32766 on newer
        # builds); 500 is a safe chunk that keeps the placeholder list short.
        chunk_size = 500
        with self._lock:
            for start in range(0, len(unique), chunk_size):
                chunk = unique[start : start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                cursor = self._conn.execute(
                    f"SELECT url, MIN(first_seen_at) FROM seen_products "
                    f"WHERE url IN ({placeholders}) AND first_seen_at != '' "
                    f"GROUP BY url",
                    chunk,
                )
                for row in cursor.fetchall():
                    if row[1] is not None:
                        out[str(row[0])] = str(row[1])
        return out

    def first_seen_for_hashes(self, hashes: list[str]) -> dict[str, str]:
        """Return ``hash -> first_seen_at`` for every provided hash that exists."""
        cleaned = [h for h in hashes if h]
        if not cleaned:
            return {}
        unique = list(dict.fromkeys(cleaned))
        out: dict[str, str] = {}
        chunk_size = 500
        with self._lock:
            for start in range(0, len(unique), chunk_size):
                chunk = unique[start : start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                cursor = self._conn.execute(
                    f"SELECT hash, first_seen_at FROM seen_products "
                    f"WHERE hash IN ({placeholders})",
                    chunk,
                )
                for row in cursor.fetchall():
                    out[str(row[0])] = str(row[1] or "")
        return out

    _UPSERT_SQL = """
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
        """

    def record_many(self, rows: list[tuple]) -> None:
        """Upsert seen-product rows. Each ``rows`` entry is the positional
        tuple ``(hash, url, title, description, first_seen_at,
        shopify_updated_at, roaster, platform)``.
        """
        if not rows:
            return
        try:
            with self._lock:
                self._conn.executemany(self._UPSERT_SQL, rows)
                self._conn.commit()
        except sqlite3.Error as exc:
            if self._logger:
                self._logger.warning(
                    "Failed to bulk-record %d seen products: %s", len(rows), exc
                )

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except sqlite3.Error as exc:
            if self._logger:
                self._logger.warning("Failed to close seen products DB: %s", exc)
