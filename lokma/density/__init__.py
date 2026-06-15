"""Density layer: the 4-level density resolver and shared category maps."""

from lokma.density.categories import (
    CATEGORY_DENSITY,
    CATEGORY_HEIGHT_CM,
    CLASS_CATEGORY,
    CLASS_SHAPE,
    WATER_DENSITY,
    density_for,
    height_for,
    shape_for,
)
from lokma.density.density_service import DensityService, DensitySource

__all__ = [
    "CATEGORY_DENSITY",
    "CATEGORY_HEIGHT_CM",
    "CLASS_CATEGORY",
    "CLASS_SHAPE",
    "DensityService",
    "DensitySource",
    "WATER_DENSITY",
    "density_for",
    "height_for",
    "shape_for",
]
