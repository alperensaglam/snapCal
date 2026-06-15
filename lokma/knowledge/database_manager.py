"""DatabaseManager — owns the SQLite connection and the in-memory record cache.

This replaces ``get_nutrition_info()``, which opened and closed a fresh
connection *for every detection in every frame* (the source of the "silent DB
lock" pain point). The manager opens one connection and, on the inference path,
loads every row once into an in-memory dict so per-frame lookups are O(1) and
touch no I/O.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from lokma.core.exceptions import DatabaseError
from lokma.core.models import FoodRecord
from lokma.knowledge.schema import (
    CREATE_NUTRITION_TABLE,
    DROP_NUTRITION_TABLE,
    NUTRITION_COLUMNS,
    insert_sql,
    select_sql,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Read/write gateway to ``lokma_local.db``.

    Use ``read_only=True`` on the inference path (opens a read-only URI and
    cannot accidentally mutate the DB) and ``read_only=False`` for the builder.
    """

    def __init__(self, db_path: Path | str, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None
        self._cache: dict[str, FoodRecord] | None = None

    # --- connection lifecycle ----------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        try:
            if self.read_only:
                if not self.db_path.exists():
                    raise DatabaseError(
                        f"Knowledge base not found at {self.db_path}. "
                        "Run `python scripts/build_knowledge_base.py` first."
                    )
                uri = f"file:{self.db_path.as_posix()}?mode=ro"
                self._conn = sqlite3.connect(uri, uri=True)
            else:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise DatabaseError(f"Failed to open database {self.db_path}: {exc}") from exc
        return self._conn

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DatabaseManager":
        self._connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        self.close()

    # --- write API (builder) -----------------------------------------------

    def create_schema(self, drop_existing: bool = False) -> None:
        """(Re)create the ``nutrition`` table."""
        conn = self._connect()
        cur = conn.cursor()
        if drop_existing:
            cur.execute(DROP_NUTRITION_TABLE)
        cur.executescript(CREATE_NUTRITION_TABLE)
        conn.commit()

    def upsert_records(self, records: list[dict]) -> int:
        """Insert/replace rows. Each dict must contain every NUTRITION_COLUMNS key.

        Values are bound by explicit column name, so an 11-vs-9 style mismatch is
        structurally impossible.
        """
        if not records:
            return 0
        conn = self._connect()
        try:
            rows = [[record[col] for col in NUTRITION_COLUMNS] for record in records]
        except KeyError as exc:
            raise DatabaseError(f"Record missing required column: {exc}") from exc
        try:
            conn.executemany(insert_sql(), rows)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to upsert records: {exc}") from exc
        return len(rows)

    def update_density(self, class_name: str, rho: float, source: str = "nutrition5k_depth") -> None:
        """Update a class's stored density (used by the offline density-mining seam)."""
        conn = self._connect()
        conn.execute(
            f"UPDATE {self._table} SET density = ? WHERE LOWER(class_name) = ?",
            (rho, class_name.lower().strip()),
        )
        logger.debug("Updated density for %s -> %.4f (%s)", class_name, rho, source)

    # --- read API (inference) ----------------------------------------------

    def load(self) -> dict[str, FoodRecord]:
        """Load every row into the in-memory cache and return it."""
        conn = self._connect()
        try:
            cursor = conn.execute(select_sql())
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to read nutrition table: {exc}") from exc
        cache: dict[str, FoodRecord] = {}
        for row in cursor.fetchall():
            record = FoodRecord.from_row(row)
            cache[record.class_name.lower().strip()] = record
        self._cache = cache
        logger.info("Loaded %d food records from %s", len(cache), self.db_path)
        return cache

    def get_food_record(self, class_name: str) -> FoodRecord | None:
        """O(1) cached lookup by (case-insensitive) class name."""
        if self._cache is None:
            self.load()
        assert self._cache is not None
        return self._cache.get(class_name.lower().strip())

    def all_records(self) -> list[FoodRecord]:
        if self._cache is None:
            self.load()
        assert self._cache is not None
        return list(self._cache.values())

    @property
    def _table(self) -> str:
        from lokma.knowledge.schema import NUTRITION_TABLE

        return NUTRITION_TABLE
