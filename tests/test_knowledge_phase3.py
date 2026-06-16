"""Phase 3 integration: taxonomy build, cross-lingual resolution, resolver core.

Runs on the base interpreter — the build is deterministic and download-free with
``enable_embedding_resolver=False`` (no torch / sentence-transformers needed).
"""

from dataclasses import replace

import numpy as np

from lokma.config import AppConfig
from lokma.knowledge import taxonomy
from lokma.knowledge.builder import KnowledgeBaseBuilder
from lokma.knowledge.database_manager import DatabaseManager
from lokma.knowledge.entity_resolver import EntityResolver


def _cfg(tmp_path):
    return replace(
        AppConfig(),
        db_path=tmp_path / "kb.db",
        val_labels_dir=tmp_path / "nolabels",   # hermetic: default ref areas
        enable_embedding_resolver=False,
    )


def test_build_populates_normalized_schema(tmp_path):
    cfg = _cfg(tmp_path)
    n = KnowledgeBaseBuilder(cfg).build()
    assert n == len(taxonomy.FOODS)

    db = DatabaseManager(cfg.db_path, read_only=True)
    assert len(db.load()) == len(taxonomy.FOODS)

    baklava = db.get_food_record("baklava")  # Turkish, via compat view
    assert baklava.density == 1.2
    assert baklava.calories_per_100g == 440
    assert baklava.source == "turkomp"

    lahmacun = db.get_food_record("lahmacun")
    assert lahmacun.geometric_shape == "flat"
    assert abs(lahmacun.density - 0.55) < 1e-9

    assert db.get_food_record("chicken_breast").calories_per_100g == 165  # essential
    db.close()


def test_cross_lingual_alias_resolution(tmp_path):
    cfg = _cfg(tmp_path)
    KnowledgeBaseBuilder(cfg).build()
    db = DatabaseManager(cfg.db_path, read_only=True)

    fid_tr = db.get_food_id("Lahmacun", "tr")
    assert fid_tr is not None and db.get_food(fid_tr)["slug"] == "lahmacun"

    # 'muz' (tr) and 'banana' (en) point at the same food.
    assert db.get_food_id("muz") == db.get_food_id("banana")

    # English synonym resolves the Turkish dish.
    assert db.get_food(db.get_food_id("lentil soup"))["slug"] == "mercimek_corbasi"
    db.close()


def test_resolver_match_in_corpus_is_pure_numpy():
    corpus = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    q = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    q /= np.linalg.norm(q)
    idx, score = EntityResolver.match_in_corpus(q, corpus, threshold=0.5)
    assert idx == 0 and score > 0.9

    # Below threshold -> no match.
    miss, _ = EntityResolver.match_in_corpus(np.array([0, 0, 1], np.float32), corpus, threshold=0.5)
    assert miss is None
