"""Tests for the 4-level density fallback hierarchy."""

from lokma.core.models import FoodRecord
from lokma.density.density_service import DensityService, DensitySource


def _food(**overrides) -> FoodRecord:
    base = dict(
        class_name="apple_pie",
        source="survey",
        usda_desc="x",
        calories_per_100g=0.0,
        protein_per_100g=0.0,
        fat_per_100g=0.0,
        carbs_per_100g=0.0,
        portion_g=100.0,
        ref_area=None,
        density=None,
        geometric_shape=None,
    )
    base.update(overrides)
    return FoodRecord(**base)


def test_level1_database_density_wins():
    rho, source = DensityService().resolve(_food(density=1.3))
    assert rho == 1.3
    assert source == DensitySource.DATABASE


def test_level3_category_average():
    rho, source = DensityService().resolve(_food(class_name="baklava", density=None))
    assert rho == 1.2  # syrup_pastry
    assert source == DensitySource.CATEGORY


def test_level4_water_fallback_for_unknown_class():
    rho, source = DensityService().resolve(_food(class_name="totally_unknown", density=None))
    assert rho == 1.0
    assert source == DensitySource.WATER


def test_level2_usda_cup_conversion():
    # 236.588 g in one cup -> exactly water density.
    assert abs(DensityService.density_from_usda_cup(236.588) - 1.0) < 1e-9
