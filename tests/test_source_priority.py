"""SourcePriorityPolicy — the cultural/raw → TurKomp/USDA resolution matrix.

Pure-Python, no DB. Locks the business rule that turns cuisine into the
authoritative source and the integer ``nutrition_facts.priority`` the schema
resolves with.
"""

from __future__ import annotations

import pytest

from lokma.core.exceptions import KnowledgeBaseBuildError
from lokma.knowledge import taxonomy
from lokma.knowledge.source_priority import (
    AUTHORITATIVE_PRIORITY,
    SECONDARY_PRIORITY,
    SourcePriorityPolicy,
)
from lokma.knowledge.taxonomy import (
    CUISINE_TURKISH,
    SOURCE_TURKOMP,
    SOURCE_USDA,
    TaxonomyFood,
)

POLICY = SourcePriorityPolicy()
BY_SLUG = taxonomy.by_slug()

CULTURAL = ["lahmacun", "mercimek_corbasi", "pilav", "doner"]
RAW = ["chicken_breast", "cooked_rice", "egg", "banana"]


@pytest.mark.parametrize("slug", CULTURAL)
def test_cultural_foods_are_turkomp_authoritative(slug):
    food = BY_SLUG[slug]
    assert POLICY.classify(food) == "cultural"
    assert POLICY.authoritative_source(food) == SOURCE_TURKOMP


@pytest.mark.parametrize("slug", RAW)
def test_raw_foods_are_usda_authoritative(slug):
    food = BY_SLUG[slug]
    assert POLICY.classify(food) == "raw"
    assert POLICY.authoritative_source(food) == SOURCE_USDA


def test_priority_ordering():
    cultural = BY_SLUG["lahmacun"]   # authoritative = turkomp
    assert POLICY.priority_for(SOURCE_TURKOMP, cultural, default=100) == AUTHORITATIVE_PRIORITY
    assert POLICY.priority_for(SOURCE_USDA, cultural, default=100) == SECONDARY_PRIORITY
    # A non-real source (e.g. LLM fallback) keeps its own priority, untouched.
    assert POLICY.priority_for("llm", cultural, default=50) == 50
    assert AUTHORITATIVE_PRIORITY < SECONDARY_PRIORITY < 50


def test_validate_passes_on_real_taxonomy():
    POLICY.validate(taxonomy.FOODS)  # must not raise — curated taxonomy is consistent


def test_validate_raises_on_inconsistent_food():
    # Turkish-cuisine dish wrongly curated against USDA -> policy must reject it.
    bad = TaxonomyFood(
        "fake_dish", "Fake", "Sahte", CUISINE_TURKISH, "stew",
        1.0, "cylinder", 4.0, 200, 100, 5.0, 3.0, 10.0, SOURCE_USDA,
    )
    with pytest.raises(KnowledgeBaseBuildError, match="cultural/raw matrix"):
        POLICY.validate([bad])


def test_override_wins_over_cuisine():
    policy = SourcePriorityPolicy(overrides={"banana": SOURCE_TURKOMP})
    assert policy.authoritative_source(BY_SLUG["banana"]) == SOURCE_TURKOMP  # forced
    assert policy.authoritative_source(BY_SLUG["lahmacun"]) == SOURCE_TURKOMP  # cuisine still applies
