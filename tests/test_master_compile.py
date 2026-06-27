"""Master compilation enhancements — bilingual view, db_version, priority, deploy.

Builds the knowledge base into a temp DB (curated-only, no ML deps) and asserts
the production-readiness additions land in the SQLite sink, the parity-locked
FoodRecord path still loads, and the frictionless deploy copies the bundle.
"""

from __future__ import annotations

from dataclasses import replace

from lokma.config import AppConfig
from lokma.knowledge import taxonomy
from lokma.knowledge.builder import KnowledgeBaseBuilder, deploy_database
from lokma.knowledge.database_manager import DatabaseManager
from lokma.knowledge.schema import DB_VERSION


def _build(tmp_path) -> AppConfig:
    cfg = replace(
        AppConfig(), db_path=tmp_path / "kb.db",
        val_labels_dir=tmp_path / "none", enable_embedding_resolver=False,
    )
    KnowledgeBaseBuilder(cfg).build()
    return cfg


def test_nutrition_view_exposes_bilingual_names(tmp_path):
    cfg = _build(tmp_path)
    db = DatabaseManager(cfg.db_path, read_only=True)
    conn = db._connect()
    row = conn.execute(
        "SELECT name_en, name_tr FROM nutrition WHERE class_name = ?", ("mercimek_corbasi",)
    ).fetchone()
    assert row["name_en"] == "Lentil soup"
    assert row["name_tr"] == "Mercimek çorbası"
    # And a food whose en/tr are identical still resolves both columns.
    lah = conn.execute(
        "SELECT name_en, name_tr FROM nutrition WHERE class_name = ?", ("lahmacun",)
    ).fetchone()
    assert lah["name_en"] == "Lahmacun" and lah["name_tr"] == "Lahmacun"
    db.close()


def test_kb_meta_has_db_version_and_timestamp(tmp_path):
    cfg = _build(tmp_path)
    db = DatabaseManager(cfg.db_path, read_only=True)
    conn = db._connect()
    meta = dict(conn.execute("SELECT key, value FROM kb_meta").fetchall())
    assert meta["db_version"] == DB_VERSION == "1.0.0"
    assert meta["built_at"]  # ISO timestamp present
    assert meta["schema_version"] == "3.0"  # table structure unchanged
    db.close()


def test_curated_facts_get_authoritative_priority(tmp_path):
    cfg = _build(tmp_path)
    db = DatabaseManager(cfg.db_path, read_only=True)
    conn = db._connect()
    # Every curated fact's source is its authoritative source -> priority 0.
    rows = conn.execute(
        "SELECT f.slug, nf.source, nf.priority FROM nutrition_facts nf "
        "JOIN food f ON f.food_id = nf.food_id"
    ).fetchall()
    assert rows  # sanity
    for r in rows:
        assert r["priority"] == 0, f"{r['slug']} ({r['source']}) priority={r['priority']}"
    db.close()


def test_parity_foodrecord_path_still_loads(tmp_path):
    """The additive view columns must not break the FoodRecord (parity) read path."""
    cfg = _build(tmp_path)
    db = DatabaseManager(cfg.db_path, read_only=True)
    rec = db.get_food_record("lahmacun")
    assert rec is not None
    assert rec.class_name == "lahmacun"
    assert rec.calories_per_100g == 230.0  # curated taxonomy value, per-100g intact
    db.close()


def test_deploy_database_copies_bundle(tmp_path):
    res_dir = tmp_path / "ios_resources"
    cfg = replace(
        AppConfig(), db_path=tmp_path / "kb.db", ios_resources_dir=res_dir,
        val_labels_dir=tmp_path / "none", enable_embedding_resolver=False,
    )
    KnowledgeBaseBuilder(cfg).build(deploy=True)
    dest = res_dir / "lokma_local.db"
    assert dest.exists()
    assert dest.read_bytes() == (tmp_path / "kb.db").read_bytes()
