"""KnowledgeBaseBuilder — offline USDA -> SQLite ETL (refactor of nutrition_extractor.py).

Fixes vs. the original script:
  * Writes through :class:`DatabaseManager` with named-column INSERTs (no 11-vs-9).
  * Encodes the USDA description corpus **once per source** instead of once per
    class (was ~10x redundant SentenceTransformer work).
  * Populates ``density`` / ``geometric_shape`` from the shared category maps, so
    those columns are no longer ``NULL`` and the DensityService has real data.
  * All paths come from :class:`AppConfig` (writes ``lokma_local.db``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from lokma.config import AppConfig
from lokma.core.exceptions import KnowledgeBaseBuildError
from lokma.density.categories import density_for, shape_for
from lokma.knowledge.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """Builds the ``nutrition`` table from USDA Survey + Foundation data."""

    TARGET_CLASSES: list[str] = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio",
        "beef_tartare", "beet_salad", "beignets", "bibimbap",
        "bread_pudding", "breakfast_burrito",
    ]

    # Curated overrides that defeat false-friend matches (e.g. carp vs carpaccio).
    EXCEPTIONS: dict[str, str] = {
        "beef_carpaccio": "Beef, raw, lean",
        "beet_salad": "Beets, salad",
        "beef_tartare": "Steak tartare, raw",
        "baby_back_ribs": "Pork ribs, cooked",
    }

    # USDA nutrient ids differ between the Survey and Foundation datasets.
    SOURCE_NUTRIENT_MAPS: dict[str, dict[int, str]] = {
        "survey": {208: "calories", 203: "protein", 204: "fat", 205: "carbs"},
        "foundation": {1008: "calories", 1003: "protein", 1004: "fat", 1005: "carbs"},
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._sentence_model: SentenceTransformer | None = None

    # --- public API ---------------------------------------------------------

    def build(self) -> int:
        """Build the knowledge base; returns the number of rows written."""
        logger.info("Starting semantic knowledge-base build -> %s", self.config.db_path)

        reference_areas = self._average_mask_areas()
        sources = self._prepare_sources()

        records: list[dict] = []
        for idx, cls in enumerate(self.TARGET_CLASSES):
            search_term = self.EXCEPTIONS.get(cls, cls.replace("_", " "))
            match = self._match_class(search_term, sources)
            if match is None:
                logger.warning("No match for '%s' in any source", cls)
                continue

            row, score, source_name, method = match
            ref_area = reference_areas.get(idx, self.config.default_ref_area)
            record = self._build_record(cls, row, source_name, sources, ref_area)
            records.append(record)
            logger.info(
                "%-18s -> %-40s | score=%.2f | %s/%s | rho=%.2f shape=%s",
                cls, str(row["description"])[:40], score, source_name, method,
                record["density"], record["geometric_shape"],
            )

        if not records:
            raise KnowledgeBaseBuildError("No records were built; check USDA data paths.")

        db = DatabaseManager(self.config.db_path, read_only=False)
        try:
            db.create_schema(drop_existing=True)
            db.upsert_records(records)
            db.commit()
        finally:
            db.close()

        logger.info("Knowledge base built: %d rows -> %s", len(records), self.config.db_path)
        return len(records)

    # --- matching -----------------------------------------------------------

    def _match_class(self, search_term: str, sources: list[dict]):
        """Survey-first: exact substring, then semantic, gated by threshold."""
        for src in sources:
            food_df = src["food"]
            strict = food_df[
                food_df["description"].str.contains(search_term, case=False, na=False, regex=False)
            ]
            if not strict.empty:
                return strict.iloc[0], 1.0, src["name"], "exact"

            row, score = self._semantic_match(src, search_term)
            if row is not None and score >= self.config.semantic_match_threshold:
                return row, score, src["name"], "semantic"
        return None

    def _semantic_match(self, src: dict, term: str):
        """Cosine similarity against the source's *precomputed* embeddings."""
        model = self._load_sentence_model()
        target_emb = model.encode(term, convert_to_tensor=True)
        scores = util.cos_sim(target_emb, src["desc_embs"])[0]
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())
        return src["food"].iloc[best_idx], best_score

    # --- record assembly ----------------------------------------------------

    def _build_record(self, cls: str, row, source_name: str, sources: list[dict], ref_area: float) -> dict:
        src = next(s for s in sources if s["name"] == source_name)
        fdc_id = row["fdc_id"]
        macros = self._extract_macros(fdc_id, src)
        portion_g = self._extract_portion(fdc_id, src)
        return {
            "class_name": cls,
            "source": source_name,
            "usda_desc": row["description"],
            "calories": macros["calories"],
            "protein": macros["protein"],
            "fat": macros["fat"],
            "carbs": macros["carbs"],
            "portion_g": portion_g,
            "ref_area": ref_area,
            "density": density_for(cls),
            "geometric_shape": shape_for(cls),
        }

    def _extract_macros(self, fdc_id, src: dict) -> dict[str, float]:
        nutrient_df = src["nutrient"]
        nutrient_map = src["nutrient_map"]
        macros = {name: 0.0 for name in nutrient_map.values()}
        for nutrient_id, name in nutrient_map.items():
            match = nutrient_df[
                (nutrient_df["fdc_id"] == fdc_id) & (nutrient_df["nutrient_id"] == nutrient_id)
            ]["amount"]
            if not match.empty:
                macros[name] = float(match.iloc[0])
        return macros

    def _extract_portion(self, fdc_id, src: dict) -> float:
        portion_df = src["portion"]
        portion = portion_df[portion_df["fdc_id"] == fdc_id]
        if not portion.empty:
            return float(portion.iloc[0]["gram_weight"])
        return 100.0

    # --- data loading -------------------------------------------------------

    def _prepare_sources(self) -> list[dict]:
        """Load USDA CSVs and precompute description embeddings once per source."""
        model = self._load_sentence_model()
        specs = [
            ("survey", self.config.usda_survey_dir),
            ("foundation", self.config.usda_foundation_dir),
        ]
        sources: list[dict] = []
        for name, folder in specs:
            data = self._load_usda(folder)
            descriptions = data["food"]["description"].fillna("").tolist()
            logger.info("Embedding %d '%s' descriptions (once)...", len(descriptions), name)
            desc_embs = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=False)
            sources.append({
                "name": name,
                "food": data["food"],
                "nutrient": data["nutrient"],
                "portion": data["portion"],
                "nutrient_map": self.SOURCE_NUTRIENT_MAPS[name],
                "desc_embs": desc_embs,
            })
        return sources

    def _load_usda(self, folder: Path) -> dict[str, pd.DataFrame]:
        try:
            return {
                "food": pd.read_csv(folder / "food.csv", low_memory=False),
                "nutrient": pd.read_csv(folder / "food_nutrient.csv", low_memory=False),
                "portion": pd.read_csv(folder / "food_portion.csv", low_memory=False),
            }
        except FileNotFoundError as exc:
            raise KnowledgeBaseBuildError(f"Missing USDA CSV under {folder}: {exc}") from exc

    def _average_mask_areas(self) -> dict[int, float]:
        """Per-class average mask pixel area from val labels (shoelace -> 640 grid)."""
        labels_dir = self.config.val_labels_dir
        default = self.config.default_ref_area
        n = len(self.TARGET_CLASSES)
        if not labels_dir.exists():
            logger.warning("Val labels dir %s missing; using default ref areas", labels_dir)
            return {i: default for i in range(n)}

        class_areas: dict[int, list[float]] = {i: [] for i in range(n)}
        grid = self.config.mask_resolution * self.config.mask_resolution
        for label_file in labels_dir.glob("*.txt"):
            for line in label_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 7:  # need a class id + >= 3 (x, y) pairs
                    continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                xs, ys = coords[0::2], coords[1::2]
                area = 0.5 * abs(
                    sum(xs[i] * ys[i + 1] - xs[i + 1] * ys[i] for i in range(-1, len(xs) - 1))
                )
                class_areas.setdefault(cls_id, []).append(area * grid)

        return {
            cid: (sum(areas) / len(areas) if areas else default)
            for cid, areas in class_areas.items()
        }

    def _load_sentence_model(self) -> SentenceTransformer:
        if self._sentence_model is None:
            logger.info("Loading sentence model '%s'...", self.config.sentence_model_name)
            self._sentence_model = SentenceTransformer(self.config.sentence_model_name)
        return self._sentence_model
