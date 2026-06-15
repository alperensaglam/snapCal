"""Shared category -> density and class -> shape maps (Level-3 density data).

These are the categorical averages used both by the builder (to seed the DB
``density``/``geometric_shape`` columns) and by :class:`DensityService` (as the
Level-3 fallback). Keeping them in one module guarantees the build-time and
run-time defaults never diverge.

Densities are approximate g/cm³ for the *prepared dish as plated* (airy pastries
< 1.0, dense syrup-soaked or meat dishes ~ 1.0–1.2).
"""

from __future__ import annotations

#: Level-4 terminal fallback.
WATER_DENSITY: float = 1.0

#: Food-101 class -> coarse density category.
CLASS_CATEGORY: dict[str, str] = {
    "apple_pie": "pastry",
    "bread_pudding": "pastry",
    "baklava": "syrup_pastry",
    "baby_back_ribs": "meat",
    "beef_carpaccio": "meat",
    "beef_tartare": "meat",
    "beet_salad": "salad",
    "beignets": "fried_dough",
    "bibimbap": "rice_bowl",
    "breakfast_burrito": "wrap",
}

#: Category -> representative density (g/cm³).
CATEGORY_DENSITY: dict[str, float] = {
    "pastry": 0.60,
    "syrup_pastry": 1.20,
    "meat": 1.05,
    "salad": 0.50,
    "fried_dough": 0.35,
    "rice_bowl": 0.85,
    "wrap": 0.90,
}

#: Class -> geometric shape prior used by the volume engine.
CLASS_SHAPE: dict[str, str] = {
    "apple_pie": "prism",
    "bread_pudding": "prism",
    "baklava": "prism",
    "baby_back_ribs": "prism",
    "beef_carpaccio": "flat",
    "beef_tartare": "paraboloid",
    "beet_salad": "paraboloid",
    "beignets": "prism",
    "bibimbap": "cylinder",
    "breakfast_burrito": "cylinder",
}


def _norm(class_name: str) -> str:
    return class_name.lower().strip()


def density_for(class_name: str) -> float | None:
    """Category-average density for a class, or ``None`` if unknown."""
    category = CLASS_CATEGORY.get(_norm(class_name))
    if category is None:
        return None
    return CATEGORY_DENSITY.get(category)


def shape_for(class_name: str) -> str | None:
    """Geometric-shape prior for a class, or ``None`` if unknown."""
    return CLASS_SHAPE.get(_norm(class_name))
