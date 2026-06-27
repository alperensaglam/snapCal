"""Source-priority policy — the cultural/raw resolution matrix for nutrition facts.

LOKMA's macros come from two conceptual sources (USDA, TürKomp). Which one is
*authoritative* for a given food is a business rule, not a per-row accident:

  * **Cultural / local** dishes (Turkish cuisine: lahmacun, mercimek_corbasi,
    pilav, doner, …) → **TürKomp** wins, to preserve regional culinary reality.
  * **Raw / universal** ingredients & global standards (chicken_breast, cooked_rice,
    egg, banana, …) → **USDA** wins, for laboratory-grade fidelity.

This policy turns that rule into the integer ``nutrition_facts.priority`` the
schema already resolves with (``MIN(priority)`` in the ``nutrition`` view). Today
each food carries a single curated fact whose ``source`` already equals the
authoritative source, so priorities are unchanged (the authoritative fact gets
``0``); the policy's job is to (a) **validate** that the curated taxonomy is
internally consistent and (b) deterministically pick the winner the day a food
carries competing facts from both sources. Per-100g units and the SQLite sink are
unchanged — this only sets priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from lokma.core.exceptions import KnowledgeBaseBuildError
from lokma.knowledge.taxonomy import (
    CUISINE_TURKISH,
    SOURCE_TURKOMP,
    SOURCE_USDA,
    TaxonomyFood,
)

#: Priority of a food's authoritative source (lowest = preferred by the view).
AUTHORITATIVE_PRIORITY = 0
#: Priority of the other *real* source when it also has a competing fact.
SECONDARY_PRIORITY = 10

#: The two real nutrition sources this matrix arbitrates between. Anything else
#: (e.g. an LLM fallback fact) keeps its own priority and is never promoted here.
_REAL_SOURCES = frozenset({SOURCE_USDA, SOURCE_TURKOMP})


@dataclass(frozen=True)
class SourcePriorityPolicy:
    """Maps each food to its authoritative source and ranks competing facts.

    ``overrides`` (slug → source) is an explicit escape hatch for foods the
    cuisine heuristic would misclassify (e.g. a Turkish-cuisine item that should
    defer to USDA, or vice versa); it wins over the cuisine rule.
    """

    overrides: Mapping[str, str] = field(default_factory=dict)

    def classify(self, food: TaxonomyFood) -> str:
        """``"cultural"`` for Turkish-cuisine dishes, ``"raw"`` otherwise."""
        return "cultural" if food.cuisine == CUISINE_TURKISH else "raw"

    def authoritative_source(self, food: TaxonomyFood) -> str:
        """The source whose metrics should win for this food (override-aware)."""
        if (forced := self.overrides.get(food.slug)) is not None:
            return forced
        return SOURCE_TURKOMP if self.classify(food) == "cultural" else SOURCE_USDA

    def priority_for(self, source: str, food: TaxonomyFood, *, default: int) -> int:
        """Priority for a fact: 0 if authoritative, 10 if the other real source.

        A non-real source (e.g. an LLM fallback) keeps ``default`` (its own
        priority), so the cultural/raw matrix never disturbs the fallback tiers.
        """
        if source == self.authoritative_source(food):
            return AUTHORITATIVE_PRIORITY
        if source in _REAL_SOURCES:
            return SECONDARY_PRIORITY
        return default

    def validate(self, foods: list[TaxonomyFood]) -> None:
        """Fail the build if any curated food disagrees with the matrix.

        Guards against a mistagged taxonomy entry (e.g. a Turkish dish whose
        curated ``source`` is USDA), which would otherwise silently ship the
        wrong authoritative macros.
        """
        bad = [
            f for f in foods
            if f.source in _REAL_SOURCES and f.source != self.authoritative_source(f)
        ]
        if bad:
            detail = ", ".join(
                f"{f.slug}(source={f.source}, expected={self.authoritative_source(f)})"
                for f in bad
            )
            raise KnowledgeBaseBuildError(
                f"source-priority policy: {len(bad)} food(s) disagree with the "
                f"cultural/raw matrix: {detail}"
            )
