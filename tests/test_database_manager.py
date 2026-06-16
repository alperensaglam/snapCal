"""DatabaseManager on the normalized Phase-3 schema + the FoodRecord compat view."""

import pytest

from lokma.core.exceptions import DatabaseError
from lokma.knowledge.database_manager import DatabaseManager
from lokma.knowledge.schema import (
    CLASS_MAP_COLUMNS,
    FOOD_ALIAS_COLUMNS,
    FOOD_COLUMNS,
    NUTRITION_FACTS_COLUMNS,
)


def _build(tmp_path) -> DatabaseManager:
    db = DatabaseManager(tmp_path / "t.db", read_only=False)
    db.create_schema(drop_existing=True)
    db.insert_many("food", FOOD_COLUMNS, [{
        "food_id": 1, "slug": "baklava", "canonical_name": "Baklava", "cuisine": "turkish",
        "category": "syrup_pastry", "density": 1.2, "geometric_shape": "prism",
        "height_cm": 4.0, "default_portion_g": 60,
    }])
    db.insert_many("food_alias", FOOD_ALIAS_COLUMNS, [
        {"food_id": 1, "lang": "en", "text": "Baklava", "kind": "primary"},
        {"food_id": 1, "lang": "tr", "text": "baklava", "kind": "primary"},
    ])
    db.insert_many("nutrition_facts", NUTRITION_FACTS_COLUMNS, [{
        "food_id": 1, "source": "turkomp", "source_ref": None,
        "calories": 440, "protein": 6.6, "fat": 29, "carbs": 38, "priority": 0,
    }])
    db.insert_many("class_map", CLASS_MAP_COLUMNS, [
        {"model_version": "foodyolo_v1", "class_id": 2, "food_id": 1, "ref_area": 87543.0},
    ])
    db.set_meta("active_model_version", "foodyolo_v1")
    db.commit()
    return db


def test_normalized_build_and_compat_view(tmp_path):
    db = _build(tmp_path)
    assert "baklava" in db.load()
    rec = db.get_food_record("BAKLAVA")  # case-insensitive
    assert rec.density == 1.2
    assert rec.calories_per_100g == 440
    assert rec.geometric_shape == "prism"
    assert rec.source == "turkomp"
    assert rec.ref_area == 87543.0  # joined via class_map + active_model_version
    db.close()


def test_multilingual_alias_lookup(tmp_path):
    db = _build(tmp_path)
    assert db.get_food_id("baklava", "tr") == 1
    assert db.get_food_id("Baklava") == 1  # case-insensitive
    assert db.get_food_id("nonexistent") is None
    assert db.get_food(1)["slug"] == "baklava"
    db.close()


def test_insert_many_missing_column_raises(tmp_path):
    db = DatabaseManager(tmp_path / "t.db", read_only=False)
    db.create_schema(drop_existing=True)
    with pytest.raises(DatabaseError):
        db.insert_many("food", FOOD_COLUMNS, [{"food_id": 1}])  # missing columns
    db.close()
