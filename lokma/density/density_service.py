"""DensityService — the hierarchical density resolver (the "DensityManager").

Resolves ``ρ`` (g/cm³) for a food via a fallback hierarchy:

  * **Level 1 — Nutrition5k depth:** the authoritative density stored in the DB
    ``density`` column. Seeded today from category averages at build time and
    overwritten by the offline depth-mining seam (see :meth:`update_density` on
    the DatabaseManager).
  * **Level 2 — USDA cup→gram:** ``ρ = grams / volume_cm³``. Implemented as
    :meth:`density_from_usda_cup`; used by the offline build path when a cup
    portion is available (a measure unit is not carried on the runtime record).
  * **Level 3 — Category average:** :func:`lokma.density.categories.density_for`.
  * **Level 4 — Water:** ``1.0`` terminal fallback.
"""

from __future__ import annotations

from enum import Enum

from lokma.core.models import FoodRecord
from lokma.density.categories import WATER_DENSITY, density_for

#: Volume of one US legal cup in cm³ (used for the Level-2 conversion).
USDA_CUP_VOLUME_CM3: float = 236.588


class DensitySource(str, Enum):
    """Which level of the hierarchy produced a density value."""

    NUTRITION5K = "nutrition5k_depth"
    DATABASE = "database"
    USDA_PORTION = "usda_portion"
    CATEGORY = "category"
    WATER = "water"


class DensityService:
    """Resolves food density through the 4-level fallback hierarchy."""

    def __init__(self, water_density: float = WATER_DENSITY) -> None:
        self.water_density = water_density

    def resolve(self, food: FoodRecord) -> tuple[float, DensitySource]:
        """Return ``(density_g_per_cm3, source)`` for a food record."""
        # Level 1: authoritative density stored in the DB.
        if food.density is not None and food.density > 0:
            return food.density, DensitySource.DATABASE

        # Level 2 (USDA cup→gram) is an offline path; no cup volume is carried on
        # the runtime record, so it is intentionally skipped here.

        # Level 3: categorical average.
        category_rho = density_for(food.class_name)
        if category_rho is not None and category_rho > 0:
            return category_rho, DensitySource.CATEGORY

        # Level 4: water.
        return self.water_density, DensitySource.WATER

    @staticmethod
    def density_from_usda_cup(
        grams: float, cups: float = 1.0, cup_volume_cm3: float = USDA_CUP_VOLUME_CM3
    ) -> float:
        """Level-2 conversion: density (g/cm³) from a USDA cup portion weight."""
        volume_cm3 = cups * cup_volume_cm3
        if volume_cm3 <= 0:
            raise ValueError("cup volume must be positive")
        return grams / volume_cm3
