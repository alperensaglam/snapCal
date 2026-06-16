"""Phase 3b: Altın Liste integrity, data.yaml generation, and v2 class_map.

Runs on the base interpreter — no ultralytics needed (auto-labeling is exercised
separately with the ML env in the runbook).
"""

from dataclasses import replace

from lokma.config import AppConfig
from lokma.knowledge import taxonomy
from lokma.knowledge.builder import KnowledgeBaseBuilder
from lokma.knowledge.database_manager import DatabaseManager
from lokma.training.dataset_builder import write_data_yaml


def test_golden_list_integrity():
    index = taxonomy.by_slug()
    assert len(taxonomy.GOLDEN_LIST) == 14
    assert len(set(taxonomy.GOLDEN_LIST)) == len(taxonomy.GOLDEN_LIST)  # no dups
    for slug in taxonomy.GOLDEN_LIST:
        assert slug in index, f"GOLDEN_LIST slug not in taxonomy: {slug}"
    assert [f.slug for f in taxonomy.golden_foods()] == list(taxonomy.GOLDEN_LIST)


def test_write_data_yaml_uses_golden_order(tmp_path):
    path = write_data_yaml(tmp_path / "yolo_v2", taxonomy.GOLDEN_LIST)
    text = path.read_text()
    assert "nc: 14" in text
    assert "train: images/train" in text
    assert "0: baklava" in text          # class_id 0
    assert "13: breakfast_burrito" in text  # class_id 13


def test_builder_emits_v2_class_map(tmp_path):
    cfg = replace(
        AppConfig(), db_path=tmp_path / "kb.db",
        val_labels_dir=tmp_path / "none", enable_embedding_resolver=False,
    )
    KnowledgeBaseBuilder(cfg).build()

    db = DatabaseManager(cfg.db_path, read_only=True)
    conn = db._connect()
    rows = conn.execute(
        "SELECT class_id, food_id FROM class_map WHERE model_version = ? ORDER BY class_id",
        (taxonomy.V2_MODEL_VERSION,),
    ).fetchall()
    assert len(rows) == len(taxonomy.GOLDEN_LIST)
    assert [r["class_id"] for r in rows] == list(range(len(taxonomy.GOLDEN_LIST)))
    assert rows[0]["food_id"] == db.get_food_id("baklava")  # GOLDEN_LIST[0]

    active = conn.execute("SELECT value FROM kb_meta WHERE key = 'active_model_version'").fetchone()
    assert active["value"] == cfg.model_version  # config-driven
    db.close()
