"""Normalized, multilingual, multi-source knowledge schema (Phase 3).

The flat Phase-1 ``nutrition`` table (keyed by ``class_name``) becomes a normalized
entity model whose stable spine is ``food_id``, decoupling visual class ↔ canonical
food ↔ language ↔ nutrition source:

  food            canonical entity + authoritative physical attributes
  food_alias      multilingual names / synonyms / matched source descriptions
  nutrition_facts per-source macros (USDA / TürKomp / curated), pick by priority
  class_map       visual model (versioned) -> food
  kb_meta         provenance (schema version, model version, embedding model, ...)

A compatibility **view** named ``nutrition`` re-exposes the exact Phase-1/2
``FoodRecord`` columns (keyed by ``class_name`` = ``food.slug``), so the cached
``DatabaseManager.load()`` and the whole inference pipeline keep working unchanged.

Every table derives its INSERT from one column-list constant (the Phase-1
"single source of truth" discipline), so positional drift is impossible.
"""

from __future__ import annotations

SCHEMA_VERSION = "3.0"

# --- table column constants (authoritative) ---------------------------------

FOOD_COLUMNS = [
    "food_id", "slug", "canonical_name", "cuisine", "category",
    "density", "geometric_shape", "height_cm", "default_portion_g",
]
FOOD_ALIAS_COLUMNS = ["food_id", "lang", "text", "kind"]
NUTRITION_FACTS_COLUMNS = [
    "food_id", "source", "source_ref", "calories", "protein", "fat", "carbs", "priority",
]
CLASS_MAP_COLUMNS = ["model_version", "class_id", "food_id", "ref_area"]
KB_META_COLUMNS = ["key", "value"]

#: Columns of the Phase-1/2 compatibility view (consumed by FoodRecord.from_row).
NUTRITION_COLUMNS = [
    "class_name", "source", "usda_desc", "calories", "protein", "fat",
    "carbs", "portion_g", "ref_area", "density", "geometric_shape",
]

# --- DDL --------------------------------------------------------------------

CREATE_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS food (
        food_id           INTEGER PRIMARY KEY,
        slug              TEXT UNIQUE NOT NULL,
        canonical_name    TEXT NOT NULL,
        cuisine           TEXT,
        category          TEXT,
        density           REAL,
        geometric_shape   TEXT,
        height_cm         REAL,
        default_portion_g REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS food_alias (
        alias_id INTEGER PRIMARY KEY,
        food_id  INTEGER NOT NULL REFERENCES food(food_id),
        lang     TEXT NOT NULL,
        text     TEXT NOT NULL,
        kind     TEXT NOT NULL DEFAULT 'synonym'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS nutrition_facts (
        fact_id    INTEGER PRIMARY KEY,
        food_id    INTEGER NOT NULL REFERENCES food(food_id),
        source     TEXT NOT NULL,
        source_ref TEXT,
        calories   REAL,
        protein    REAL,
        fat        REAL,
        carbs      REAL,
        priority   INTEGER NOT NULL DEFAULT 100,
        UNIQUE(food_id, source)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS class_map (
        model_version TEXT NOT NULL,
        class_id      INTEGER NOT NULL,
        food_id       INTEGER NOT NULL REFERENCES food(food_id),
        ref_area      REAL,
        PRIMARY KEY (model_version, class_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kb_meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
]

CREATE_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_food_alias_lang_text ON food_alias(lang, text)",
    "CREATE INDEX IF NOT EXISTS idx_food_alias_food ON food_alias(food_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_food ON nutrition_facts(food_id)",
    "CREATE INDEX IF NOT EXISTS idx_class_map_food ON class_map(food_id)",
]

#: Re-exposes the Phase-1/2 FoodRecord shape. ``class_name`` = slug; macros come
#: from the lowest-priority (preferred) nutrition_facts row; ref_area from the
#: active model's class_map.
CREATE_NUTRITION_VIEW = """
CREATE VIEW IF NOT EXISTS nutrition AS
SELECT
    f.slug                       AS class_name,
    nf.source                    AS source,
    f.canonical_name             AS usda_desc,
    COALESCE(nf.calories, 0)     AS calories,
    COALESCE(nf.protein, 0)      AS protein,
    COALESCE(nf.fat, 0)          AS fat,
    COALESCE(nf.carbs, 0)        AS carbs,
    f.default_portion_g          AS portion_g,
    cm.ref_area                  AS ref_area,
    f.density                    AS density,
    f.geometric_shape            AS geometric_shape
FROM food f
LEFT JOIN nutrition_facts nf
    ON nf.food_id = f.food_id
    AND nf.priority = (SELECT MIN(priority) FROM nutrition_facts n2 WHERE n2.food_id = f.food_id)
LEFT JOIN class_map cm
    ON cm.food_id = f.food_id
    AND cm.model_version = (SELECT value FROM kb_meta WHERE key = 'active_model_version')
"""

# Note: ``nutrition`` is dropped type-aware in DatabaseManager.create_schema
# (it may be a legacy Phase-1/2 TABLE or our Phase-3 VIEW).
DROP_OBJECTS = [
    "DROP TABLE IF EXISTS class_map",
    "DROP TABLE IF EXISTS nutrition_facts",
    "DROP TABLE IF EXISTS food_alias",
    "DROP TABLE IF EXISTS food",
    "DROP TABLE IF EXISTS kb_meta",
]


def insert_sql(table: str, columns: list[str]) -> str:
    """``INSERT OR REPLACE`` naming every column (count can never drift)."""
    cols = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"


def select_sql() -> str:
    """SELECT the FoodRecord columns from the compatibility view."""
    return f"SELECT {', '.join(NUTRITION_COLUMNS)} FROM nutrition"
