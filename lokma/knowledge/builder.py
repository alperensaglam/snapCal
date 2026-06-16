"""KnowledgeBaseBuilder — taxonomy + resolver -> normalized SQLite (Phase 3).

Populates ``food`` / ``food_alias`` / ``nutrition_facts`` / ``class_map`` / ``kb_meta``
from the canonical taxonomy via the :class:`EntityResolver`. The build is
deterministic and download-free with ``config.enable_embedding_resolver = False``;
enabling it adds matched USDA ``source_ref``s + ``source_desc`` aliases.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from lokma.config import AppConfig
from lokma.core.exceptions import KnowledgeBaseBuildError
from lokma.knowledge.database_manager import DatabaseManager
from lokma.knowledge.entity_resolver import EntityResolver
from lokma.knowledge.schema import (
    CLASS_MAP_COLUMNS,
    FOOD_ALIAS_COLUMNS,
    FOOD_COLUMNS,
    NUTRITION_FACTS_COLUMNS,
    SCHEMA_VERSION,
)
from lokma.knowledge import taxonomy

logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()

    def build(self) -> int:
        foods = taxonomy.FOODS
        if not foods:
            raise KnowledgeBaseBuildError("Empty taxonomy")
        logger.info("Building knowledge base: %d foods -> %s", len(foods), self.config.db_path)

        ref_areas = self._ref_areas(self.config.val_labels_dir)
        resolver = EntityResolver(self.config)
        if self.config.enable_embedding_resolver:
            try:
                descs, refs = self._build_usda_corpus()
                if descs:
                    resolver.prepare_corpus(descs, refs)
            except Exception as exc:  # enrichment is optional; never block the build
                logger.warning("Embedding enrichment unavailable (%s); curated-only build", exc)

        food_rows: list[dict] = []
        alias_rows: list[dict] = []
        fact_rows: list[dict] = []
        class_rows: list[dict] = []
        seen_aliases: set[tuple[int, str, str]] = set()
        slug_to_id: dict[str, int] = {}

        def add_alias(food_id: int, lang: str, text: str, kind: str) -> None:
            text = (text or "").strip()
            if not text:
                return
            key = (food_id, lang, text.lower())
            if key in seen_aliases:
                return
            seen_aliases.add(key)
            alias_rows.append({"food_id": food_id, "lang": lang, "text": text, "kind": kind})

        for food_id, food in enumerate(foods, start=1):
            food_rows.append({
                "food_id": food_id, "slug": food.slug, "canonical_name": food.name_en,
                "cuisine": food.cuisine, "category": food.category, "density": food.density,
                "geometric_shape": food.geometric_shape, "height_cm": food.height_cm,
                "default_portion_g": food.default_portion_g,
            })
            slug_to_id[food.slug] = food_id
            add_alias(food_id, "en", food.name_en, "primary")
            add_alias(food_id, "tr", food.name_tr, "primary")
            for a in food.aliases_en:
                add_alias(food_id, "en", a, "synonym")
            for a in food.aliases_tr:
                add_alias(food_id, "tr", a, "synonym")

            resolved = resolver.resolve(food)
            for fact in resolved.facts:
                fact_rows.append({"food_id": food_id, **fact})
            for lang, text in resolved.source_aliases:
                add_alias(food_id, lang, text, "source_desc")

            if food.legacy_class_id is not None:
                class_rows.append({
                    "model_version": taxonomy.LEGACY_MODEL_VERSION,
                    "class_id": food.legacy_class_id, "food_id": food_id,
                    "ref_area": ref_areas.get(food.legacy_class_id, self.config.default_ref_area),
                })

        # v2 (Altın Liste) class map — class order == taxonomy.GOLDEN_LIST.
        v2_ref = self._ref_areas(self.config.yolo_v2_dataset_dir / "labels" / "val")
        for class_id, slug in enumerate(taxonomy.GOLDEN_LIST):
            food_id = slug_to_id.get(slug)
            if food_id is not None:
                class_rows.append({
                    "model_version": taxonomy.V2_MODEL_VERSION, "class_id": class_id,
                    "food_id": food_id,
                    "ref_area": v2_ref.get(class_id, self.config.default_ref_area),
                })

        db = DatabaseManager(self.config.db_path, read_only=False)
        try:
            db.create_schema(drop_existing=True)
            db.insert_many("food", FOOD_COLUMNS, food_rows)
            db.insert_many("food_alias", FOOD_ALIAS_COLUMNS, alias_rows)
            db.insert_many("nutrition_facts", NUTRITION_FACTS_COLUMNS, fact_rows)
            db.insert_many("class_map", CLASS_MAP_COLUMNS, class_rows)
            db.set_meta("schema_version", SCHEMA_VERSION)
            db.set_meta("active_model_version", self.config.model_version)
            db.set_meta("embedding_model", self.config.embedding_model)
            db.set_meta("built_at", datetime.now(timezone.utc).isoformat())
            db.commit()
        finally:
            db.close()

        logger.info(
            "Built: %d foods, %d aliases, %d facts, %d class maps",
            len(food_rows), len(alias_rows), len(fact_rows), len(class_rows),
        )
        return len(food_rows)

    # --- helpers ------------------------------------------------------------

    def _ref_areas(self, labels_dir) -> dict[int, float]:
        """Per-class average mask pixel area from val labels (shoelace -> 640 grid)."""
        default = self.config.default_ref_area
        if not labels_dir.exists():
            logger.warning("Val labels dir %s missing; default ref areas", labels_dir)
            return {}
        grid = self.config.mask_resolution * self.config.mask_resolution
        areas: dict[int, list[float]] = {}
        for label_file in labels_dir.glob("*.txt"):
            for line in label_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                xs, ys = coords[0::2], coords[1::2]
                area = 0.5 * abs(
                    sum(xs[i] * ys[i + 1] - xs[i + 1] * ys[i] for i in range(-1, len(xs) - 1))
                )
                areas.setdefault(cls_id, []).append(area * grid)
        return {cid: (sum(v) / len(v) if v else default) for cid, v in areas.items()}

    def _build_usda_corpus(self) -> tuple[list[str], list[str]]:
        """Descriptions + fdc_ids from USDA food.csv (survey + foundation)."""
        import pandas as pd

        descriptions: list[str] = []
        refs: list[str] = []
        for folder in (self.config.usda_survey_dir, self.config.usda_foundation_dir):
            food_csv = folder / "food.csv"
            if not food_csv.exists():
                continue
            df = pd.read_csv(food_csv, low_memory=False)
            for desc, fdc_id in zip(df["description"].fillna("").tolist(), df["fdc_id"].tolist()):
                if desc:
                    descriptions.append(str(desc))
                    refs.append(str(fdc_id))
        return descriptions, refs
