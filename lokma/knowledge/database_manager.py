"""DatabaseManager — connection + cache gateway to the normalized KB (Phase 3).

Owns one SQLite connection. On the inference path it loads the whole knowledge
base once (through the ``nutrition`` compatibility view) into an in-memory
``{class_name: FoodRecord}`` cache, so per-frame lookups are O(1) and never embed
or query at runtime. The builder uses the write API (``create_schema`` +
``insert_many`` + ``set_meta``).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from lokma.core.exceptions import DatabaseError
from lokma.core.models import FoodRecord
from lokma.knowledge.schema import (
    CREATE_INDICES,
    CREATE_NUTRITION_VIEW,
    CREATE_TABLES,
    DROP_OBJECTS,
    insert_sql,
    select_sql,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
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
                self._conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
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
        conn = self._connect()
        cur = conn.cursor()
        if drop_existing:
            # `nutrition` may be a legacy Phase-1/2 TABLE or our Phase-3 VIEW —
            # drop whichever actually exists (SQLite refuses a type mismatch).
            existing = cur.execute(
                "SELECT type FROM sqlite_master WHERE name = 'nutrition'"
            ).fetchone()
            if existing is not None:
                cur.execute(f"DROP {existing['type'].upper()} IF EXISTS nutrition")
            for stmt in DROP_OBJECTS:
                cur.execute(stmt)
        for stmt in CREATE_TABLES:
            cur.execute(stmt)
        for stmt in CREATE_INDICES:
            cur.execute(stmt)
        cur.execute(CREATE_NUTRITION_VIEW)
        conn.commit()

    def insert_many(self, table: str, columns: list[str], rows: list[dict]) -> int:
        """Insert/replace rows (bound by explicit column name; count can't drift)."""
        if not rows:
            return 0
        conn = self._connect()
        try:
            data = [[row[col] for col in columns] for row in rows]
        except KeyError as exc:
            raise DatabaseError(f"Row missing required column: {exc}") from exc
        try:
            conn.executemany(insert_sql(table, columns), data)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to insert into {table}: {exc}") from exc
        return len(data)

    def set_meta(self, key: str, value: str) -> None:
        self._connect().execute(
            "INSERT OR REPLACE INTO kb_meta (key, value) VALUES (?, ?)", (key, str(value))
        )

    def update_density(self, slug: str, rho: float, source: str = "nutrition5k_depth") -> None:
        """Update a food's authoritative density (offline density-mining seam)."""
        self._connect().execute("UPDATE food SET density = ? WHERE slug = ?", (rho, slug.lower().strip()))
        logger.debug("Updated density for %s -> %.4f (%s)", slug, rho, source)

    # --- read API (inference) ----------------------------------------------

    def load(self) -> dict[str, FoodRecord]:
        """Load every food (via the compat view) into the in-memory cache."""
        conn = self._connect()
        try:
            cursor = conn.execute(select_sql())
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to read knowledge base: {exc}") from exc
        cache: dict[str, FoodRecord] = {}
        for row in cursor.fetchall():
            record = FoodRecord.from_row(row)
            cache[record.class_name.lower().strip()] = record
        self._cache = cache
        logger.info("Loaded %d food records from %s", len(cache), self.db_path)
        return cache

    def get_food_record(self, class_name: str) -> FoodRecord | None:
        if self._cache is None:
            self.load()
        assert self._cache is not None
        return self._cache.get(class_name.lower().strip())

    def all_records(self) -> list[FoodRecord]:
        if self._cache is None:
            self.load()
        assert self._cache is not None
        return list(self._cache.values())

    # --- multilingual entity lookup ----------------------------------------

    def get_food_id(self, text: str, lang: str | None = None) -> int | None:
        """Resolve a (possibly Turkish) name/alias to a food_id, case-insensitively."""
        conn = self._connect()
        needle = text.lower().strip()
        if lang:
            row = conn.execute(
                "SELECT food_id FROM food_alias WHERE lower(text) = ? AND lang = ? LIMIT 1",
                (needle, lang),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT food_id FROM food_alias WHERE lower(text) = ? LIMIT 1", (needle,)
            ).fetchone()
        return row["food_id"] if row else None

    def get_food(self, food_id: int) -> dict | None:
        row = self._connect().execute("SELECT * FROM food WHERE food_id = ?", (food_id,)).fetchone()
        return dict(row) if row else None
