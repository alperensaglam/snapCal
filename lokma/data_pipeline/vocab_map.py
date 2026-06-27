"""Default raw-label → canonical-slug aliases for the Vocabulary Alignment Bridge.

External datasets label the *same* food under many names: ``turkish_lahmacun``,
``lahmacun_pide`` and ``minced_meat_flatbread`` are all the canonical taxonomy slug
``lahmacun``. This module is the single, code-reviewable place that records those
equivalences so the pipeline can normalize any incoming dataset label to a slug the
knowledge base — and therefore the runtime macro lookup — actually resolves.

Mirrors the project convention of keeping authoritative data as Python constants
(see ``taxonomy.FOODS`` / ``taxonomy.GOLDEN_LIST``). Keys are matched
case-insensitively (the bridge lower-cases both sides). **Values must be real
taxonomy slugs** — :class:`~lokma.data_pipeline.orchestrator.VocabularyBridge`
fails fast at construction if any value is not in the taxonomy, because a dangling
target would silently mint a detection class the runtime can never price.

This complements — it does not replace — the knowledge-base ``food_alias`` table /
``EntityResolver`` (which resolve human-facing multilingual *names* at DB-build
time). The bridge operates earlier, on dataset *labels*.
"""

from __future__ import annotations

#: raw external dataset label  ->  canonical taxonomy slug (see taxonomy.FOODS).
DEFAULT_LABEL_ALIASES: dict[str, str] = {
    # --- Turkish cuisine -----------------------------------------------------
    "turkish_lahmacun": "lahmacun",
    "lahmacun_pide": "lahmacun",
    "minced_meat_flatbread": "lahmacun",
    "turkish_pizza": "lahmacun",
    "doner_kebab": "doner",
    "donair": "doner",
    "shawarma": "doner",
    "gyro": "doner",
    "turkish_pide": "pide",
    "pide_bread": "pide",
    "white_bean_stew": "kuru_fasulye",
    "kuru_fasulye_stew": "kuru_fasulye",
    "lentil_soup": "mercimek_corbasi",
    "red_lentil_soup": "mercimek_corbasi",
    "rice_pilaf": "pilav",
    "turkish_rice": "pilav",
    "turkish_meatballs": "kofte",
    "izgara_kofte": "kofte",
    "meatballs": "kofte",
    # --- Everyday essentials -------------------------------------------------
    "boiled_egg": "egg",
    "fried_egg": "egg",
    "grilled_chicken": "chicken_breast",
    "chicken_fillet": "chicken_breast",
    "white_rice": "cooked_rice",
    "steamed_rice": "cooked_rice",
    "plain_rice": "cooked_rice",
    # --- Western keepers -----------------------------------------------------
    "apple_pie_slice": "apple_pie",
    "egg_burrito": "breakfast_burrito",
}
