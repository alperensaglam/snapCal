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

#: Category -> representative density (g/cm³). Turkish categories added in Phase 3.
CATEGORY_DENSITY: dict[str, float] = {
    "pastry": 0.60,
    "syrup_pastry": 1.20,
    "meat": 1.05,
    "salad": 0.50,
    "fried_dough": 0.35,
    "rice_bowl": 0.85,
    "wrap": 0.90,
    # Turkish + everyday-essentials categories
    "flatbread": 0.55,      # lahmacun — thin
    "doner_meat": 1.05,     # döner — dense mound
    "porous_dough": 0.35,   # pide — airy
    "stew": 1.00,           # kuru fasulye
    "soup": 1.00,           # mercimek çorbası
    "rice": 0.85,           # pilav / cooked rice
    "egg_dish": 0.90,
    "poultry": 1.05,
    "fruit": 0.94,
    "dairy": 1.03,
    "grain": 1.00,
    "bread": 0.30,
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


#: Category -> representative plated height (cm). Once the footprint area is
#: metric (Phase 2 calibration), height dominates volume error — so it is a
#: per-class prior rather than a single global constant.
CATEGORY_HEIGHT_CM: dict[str, float] = {
    "pastry": 3.5,
    "syrup_pastry": 4.0,
    "meat": 3.0,
    "salad": 4.0,
    "fried_dough": 4.0,
    "rice_bowl": 5.0,
    "wrap": 5.0,
    # Turkish + everyday-essentials categories
    "flatbread": 0.8,
    "doner_meat": 6.0,
    "porous_dough": 3.0,
    "stew": 4.0,
    "soup": 4.0,
    "rice": 4.0,
    "egg_dish": 2.5,
    "poultry": 2.5,
    "fruit": 3.5,
    "dairy": 3.0,
    "grain": 2.5,
    "bread": 4.0,
}

DEFAULT_HEIGHT_CM: float = 2.5


def height_for(class_name: str) -> float:
    """Category-average plated height (cm); falls back to a global default."""
    category = CLASS_CATEGORY.get(_norm(class_name))
    if category is not None and category in CATEGORY_HEIGHT_CM:
        return CATEGORY_HEIGHT_CM[category]
    return DEFAULT_HEIGHT_CM
