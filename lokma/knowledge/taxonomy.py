"""Canonical food taxonomy — the single source of truth for Phase 3.

Each :class:`TaxonomyFood` is a language-agnostic food *concept* (stable ``slug``)
carrying its multilingual names, authoritative physical attributes (density /
shape / height for the volume engine), a curated per-100 g macro seed, and — for
the foods the current Food-101 model already detects — its legacy ``class_id``.

The builder turns this into the normalized DB (``food`` / ``food_alias`` /
``nutrition_facts`` / ``class_map``). Curated macros are reasonable approximations
sourced conceptually from USDA (global) / TürKomp (Turkish), pending full
ingestion; they make the knowledge base deterministic and download-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CUISINE_GLOBAL = "global"
CUISINE_TURKISH = "turkish"

SOURCE_USDA = "usda"
SOURCE_TURKOMP = "turkomp"


@dataclass(frozen=True)
class TaxonomyFood:
    slug: str
    name_en: str
    name_tr: str
    cuisine: str
    category: str
    # authoritative physical attributes (per food, not per category)
    density: float            # g/cm³
    geometric_shape: str      # prism | cylinder | paraboloid | flat
    height_cm: float
    default_portion_g: float
    # curated per-100 g macros + their conceptual source
    calories: float
    protein: float
    fat: float
    carbs: float
    source: str
    aliases_en: tuple[str, ...] = ()
    aliases_tr: tuple[str, ...] = ()
    legacy_class_id: int | None = None     # Food-101 v1 class index, if detectable today
    usda_query: str | None = None          # optional override term for embedding/USDA match

    @property
    def primary_facts(self) -> dict[str, float]:
        return {"calories": self.calories, "protein": self.protein, "fat": self.fat, "carbs": self.carbs}


#: The active model version whose class ids map into ``class_map``.
LEGACY_MODEL_VERSION = "foodyolo_v1"

FOODS: list[TaxonomyFood] = [
    # --- Legacy Food-101 (kept for v1-model compatibility; no regression) -----
    TaxonomyFood("apple_pie", "Apple pie", "Elmalı turta", CUISINE_GLOBAL, "pastry",
                 0.60, "prism", 3.5, 125, 237, 2.4, 11.0, 34.0, SOURCE_USDA,
                 aliases_en=("apple pie",), aliases_tr=("elmalı turta",),
                 legacy_class_id=0, usda_query="apple pie"),
    TaxonomyFood("baby_back_ribs", "Baby back ribs", "Domuz kaburga", CUISINE_GLOBAL, "meat",
                 1.05, "prism", 4.0, 250, 277, 25.0, 18.0, 0.0, SOURCE_USDA,
                 aliases_en=("pork ribs", "bbq ribs"), legacy_class_id=1, usda_query="pork ribs cooked"),
    TaxonomyFood("baklava", "Baklava", "Baklava", CUISINE_TURKISH, "syrup_pastry",
                 1.20, "prism", 4.0, 60, 440, 6.6, 29.0, 38.0, SOURCE_TURKOMP,
                 aliases_en=("baklava", "pistachio baklava"), aliases_tr=("baklava", "fıstıklı baklava"),
                 legacy_class_id=2, usda_query="baklava"),
    TaxonomyFood("beef_carpaccio", "Beef carpaccio", "Dana karpaçyo", CUISINE_GLOBAL, "meat",
                 1.05, "flat", 0.8, 90, 190, 20.0, 12.0, 0.0, SOURCE_USDA,
                 aliases_en=("beef carpaccio", "raw beef"), legacy_class_id=3, usda_query="beef raw lean"),
    TaxonomyFood("beef_tartare", "Beef tartare", "Çiğ köfte (et)", CUISINE_GLOBAL, "meat",
                 1.05, "paraboloid", 3.5, 150, 196, 19.0, 12.0, 1.0, SOURCE_USDA,
                 aliases_en=("steak tartare",), legacy_class_id=4, usda_query="steak tartare raw"),
    TaxonomyFood("beet_salad", "Beet salad", "Pancar salatası", CUISINE_GLOBAL, "salad",
                 0.50, "paraboloid", 4.0, 150, 90, 1.5, 5.0, 10.0, SOURCE_USDA,
                 aliases_en=("beet salad", "beetroot salad"), aliases_tr=("pancar salatası",),
                 legacy_class_id=5, usda_query="beets salad"),
    TaxonomyFood("beignets", "Beignets", "Beignet", CUISINE_GLOBAL, "fried_dough",
                 0.35, "prism", 4.0, 90, 417, 6.0, 20.0, 52.0, SOURCE_USDA,
                 aliases_en=("beignet", "fried dough"), legacy_class_id=6, usda_query="beignet"),
    TaxonomyFood("bibimbap", "Bibimbap", "Bibimbap", CUISINE_GLOBAL, "rice_bowl",
                 0.85, "cylinder", 5.0, 400, 130, 6.0, 4.0, 18.0, SOURCE_USDA,
                 aliases_en=("bibimbap", "korean rice bowl"), legacy_class_id=7, usda_query="bibimbap korean"),
    TaxonomyFood("bread_pudding", "Bread pudding", "Ekmek puddingi", CUISINE_GLOBAL, "pastry",
                 0.60, "prism", 3.5, 150, 156, 5.0, 5.0, 23.0, SOURCE_USDA,
                 aliases_en=("bread pudding",), legacy_class_id=8, usda_query="pudding bread"),
    TaxonomyFood("breakfast_burrito", "Breakfast burrito", "Kahvaltı dürümü", CUISINE_GLOBAL, "wrap",
                 0.90, "cylinder", 5.0, 220, 210, 9.0, 9.0, 24.0, SOURCE_USDA,
                 aliases_en=("breakfast burrito", "egg burrito"), legacy_class_id=9, usda_query="egg burrito"),

    # --- Turkish cuisine (new) ------------------------------------------------
    TaxonomyFood("lahmacun", "Lahmacun", "Lahmacun", CUISINE_TURKISH, "flatbread",
                 0.55, "flat", 0.8, 180, 230, 11.0, 8.0, 30.0, SOURCE_TURKOMP,
                 aliases_en=("turkish pizza", "minced meat flatbread"), aliases_tr=("lahmacun",),
                 usda_query="flatbread minced meat"),
    TaxonomyFood("doner", "Döner kebab", "Döner", CUISINE_TURKISH, "doner_meat",
                 1.05, "paraboloid", 6.0, 250, 215, 19.0, 14.0, 3.0, SOURCE_TURKOMP,
                 aliases_en=("doner kebab", "shawarma"), aliases_tr=("döner", "döner kebap", "et döner"),
                 usda_query="gyro doner meat"),
    TaxonomyFood("pide", "Pide", "Pide", CUISINE_TURKISH, "porous_dough",
                 0.35, "prism", 3.0, 300, 270, 11.0, 9.0, 36.0, SOURCE_TURKOMP,
                 aliases_en=("turkish flatbread", "turkish pizza boat"), aliases_tr=("pide", "kıymalı pide"),
                 usda_query="turkish flatbread"),
    TaxonomyFood("kuru_fasulye", "White bean stew", "Kuru fasulye", CUISINE_TURKISH, "stew",
                 1.00, "cylinder", 4.0, 250, 130, 7.0, 4.0, 17.0, SOURCE_TURKOMP,
                 aliases_en=("white bean stew", "beans in tomato sauce"), aliases_tr=("kuru fasulye",),
                 usda_query="white beans cooked tomato"),
    TaxonomyFood("mercimek_corbasi", "Lentil soup", "Mercimek çorbası", CUISINE_TURKISH, "soup",
                 1.00, "cylinder", 4.0, 300, 55, 3.0, 1.5, 8.0, SOURCE_TURKOMP,
                 aliases_en=("lentil soup", "red lentil soup"),
                 aliases_tr=("mercimek çorbası", "mercimek corbasi"), usda_query="lentil soup"),
    TaxonomyFood("pilav", "Rice pilaf", "Pilav", CUISINE_TURKISH, "rice",
                 0.85, "paraboloid", 4.0, 180, 150, 3.0, 3.5, 27.0, SOURCE_TURKOMP,
                 aliases_en=("rice pilaf", "turkish rice"), aliases_tr=("pilav", "pirinç pilavı"),
                 usda_query="rice pilaf cooked"),
    TaxonomyFood("kofte", "Köfte", "Köfte", CUISINE_TURKISH, "meat",
                 1.05, "paraboloid", 4.0, 150, 250, 18.0, 18.0, 3.0, SOURCE_TURKOMP,
                 aliases_en=("meatballs", "turkish meatballs"), aliases_tr=("köfte", "izgara köfte"),
                 usda_query="meatballs beef"),

    # --- Everyday essentials (new) -------------------------------------------
    TaxonomyFood("egg", "Egg", "Yumurta", CUISINE_GLOBAL, "egg_dish",
                 0.90, "prism", 2.5, 50, 155, 13.0, 11.0, 1.1, SOURCE_USDA,
                 aliases_en=("egg", "boiled egg"), aliases_tr=("yumurta", "haşlanmış yumurta"),
                 usda_query="egg cooked"),
    TaxonomyFood("chicken_breast", "Chicken breast", "Tavuk göğsü", CUISINE_GLOBAL, "poultry",
                 1.05, "prism", 2.5, 120, 165, 31.0, 3.6, 0.0, SOURCE_USDA,
                 aliases_en=("chicken breast", "grilled chicken"), aliases_tr=("tavuk göğsü", "tavuk gogsu"),
                 usda_query="chicken breast cooked"),
    TaxonomyFood("cooked_rice", "Cooked rice", "Pirinç", CUISINE_GLOBAL, "rice",
                 0.85, "paraboloid", 4.0, 150, 130, 2.7, 0.3, 28.0, SOURCE_USDA,
                 aliases_en=("rice", "white rice"), aliases_tr=("pirinç", "beyaz pirinç"),
                 usda_query="white rice cooked"),
    TaxonomyFood("banana", "Banana", "Muz", CUISINE_GLOBAL, "fruit",
                 0.94, "cylinder", 3.5, 120, 89, 1.1, 0.3, 23.0, SOURCE_USDA,
                 aliases_en=("banana",), aliases_tr=("muz",), usda_query="banana raw"),
]


def by_slug() -> dict[str, TaxonomyFood]:
    return {f.slug: f for f in FOODS}


def legacy_foods() -> list[TaxonomyFood]:
    """Foods detectable by the current Food-101 model, ordered by class id."""
    return sorted((f for f in FOODS if f.legacy_class_id is not None), key=lambda f: f.legacy_class_id)


#: The model version trained on the Altın Liste (Phase 3b).
V2_MODEL_VERSION = "foodyolo_v2"

#: The **Altın Liste** — the v2 training class set. Tuple order == ``class_id``,
#: so this single constant defines the data.yaml class order, the trained model's
#: ``names``, and the v2 ``class_map`` ordering.
GOLDEN_LIST: tuple[str, ...] = (
    # Turkish cuisine (8)
    "baklava", "lahmacun", "doner", "pide",
    "kuru_fasulye", "mercimek_corbasi", "pilav", "kofte",
    # Everyday essentials (4)
    "egg", "chicken_breast", "cooked_rice", "banana",
    # Western keepers (2)
    "apple_pie", "breakfast_burrito",
)


def golden_foods() -> list[TaxonomyFood]:
    """Altın Liste foods in class-id order (the v2 training set)."""
    index = by_slug()
    return [index[slug] for slug in GOLDEN_LIST]
