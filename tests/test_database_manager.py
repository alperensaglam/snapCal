"""Tests for DatabaseManager — including the 11-column INSERT regression."""

import pytest

from lokma.core.exceptions import DatabaseError
from lokma.knowledge.database_manager import DatabaseManager
from lokma.knowledge.schema import NUTRITION_COLUMNS


def _record(name: str) -> dict:
    values = [name, "survey", "Apple pie", 100.0, 1.0, 2.0, 3.0, 150.0, 40000.0, 0.6, "prism"]
    return dict(zip(NUTRITION_COLUMNS, values))


def test_schema_roundtrip_and_11_column_insert(tmp_path):
    db = DatabaseManager(tmp_path / "t.db", read_only=False)
    db.create_schema(drop_existing=True)

    # Regression: 11 columns / 11 values must not raise "X columns but Y values".
    written = db.upsert_records([_record("apple_pie"), _record("baklava")])
    db.commit()
    assert written == 2

    cache = db.load()
    assert set(cache) == {"apple_pie", "baklava"}

    record = db.get_food_record("  APPLE_PIE ")  # normalization
    assert record is not None
    assert record.density == 0.6
    assert record.geometric_shape == "prism"
    assert record.calories_per_100g == 100.0
    db.close()


def test_read_only_missing_db_raises(tmp_path):
    db = DatabaseManager(tmp_path / "nope.db", read_only=True)
    with pytest.raises(DatabaseError):
        db.load()
