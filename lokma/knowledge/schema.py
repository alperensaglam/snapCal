"""Canonical ``nutrition`` schema — the single source of truth.

The old builder defined an 11-column table but inserted 9 positional values,
which raised ``table nutrition has 11 columns but 9 values were supplied`` on a
fresh build. Here the column list, the DDL, and the parameterized INSERT are all
derived from one constant (:data:`NUTRITION_COLUMNS`) and the INSERT names its
columns explicitly, so the count can never drift out of sync again.
"""

from __future__ import annotations

NUTRITION_TABLE = "nutrition"

#: Authoritative ordered column list. Everything else is derived from this.
NUTRITION_COLUMNS: list[str] = [
    "class_name",
    "source",
    "usda_desc",
    "calories",
    "protein",
    "fat",
    "carbs",
    "portion_g",
    "ref_area",
    "density",
    "geometric_shape",
]

CREATE_NUTRITION_TABLE = f"""
CREATE TABLE IF NOT EXISTS {NUTRITION_TABLE} (
    class_name      TEXT PRIMARY KEY,
    source          TEXT,
    usda_desc       TEXT,
    calories        REAL,
    protein         REAL,
    fat             REAL,
    carbs           REAL,
    portion_g       REAL,
    ref_area        REAL,
    density         REAL,
    geometric_shape TEXT
)
"""

DROP_NUTRITION_TABLE = f"DROP TABLE IF EXISTS {NUTRITION_TABLE}"


def insert_sql() -> str:
    """``INSERT OR REPLACE`` naming all columns (always 11 values for 11 columns)."""
    columns = ", ".join(NUTRITION_COLUMNS)
    placeholders = ", ".join("?" for _ in NUTRITION_COLUMNS)
    return f"INSERT OR REPLACE INTO {NUTRITION_TABLE} ({columns}) VALUES ({placeholders})"


def select_sql() -> str:
    """``SELECT`` of all canonical columns in order."""
    columns = ", ".join(NUTRITION_COLUMNS)
    return f"SELECT {columns} FROM {NUTRITION_TABLE}"
