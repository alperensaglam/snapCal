"""EntityResolver — the layered, offline NLP resolver (build-time only).

Three tiers, in order of authority:

  * **Tier 0 — curated:** the taxonomy's hand-authored macros (ground truth).
  * **Tier 1 — cross-lingual embeddings:** embed every source description once
    with a multilingual model; match a food's en/tr names to the best source row.
    Used to *enrich* (attach a real ``source_ref`` + a ``source_desc`` alias) and,
    for uncurated foods, to derive facts. The heavy model runs **offline**, so the
    inference runtime pays nothing.
  * **Tier 2 — LLM fallback (optional):** a lightweight LLM decomposes hard dishes.
    Gated by ``config.enable_llm_resolver`` + ``ANTHROPIC_API_KEY``; never required.

The matching core (:meth:`match_in_corpus`) is pure numpy and unit-testable
without loading any model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from lokma.config import AppConfig
from lokma.knowledge.taxonomy import TaxonomyFood

logger = logging.getLogger(__name__)


@dataclass
class ResolvedFood:
    """Resolver output for one food: facts + any matched-source aliases."""

    facts: list[dict]
    source_aliases: list[tuple[str, str]] = field(default_factory=list)  # (lang, text)
    method: str = "curated"


class EntityResolver:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._corpus_descs: list[str] | None = None
        self._corpus_refs: list[str] | None = None
        self._corpus_mat: np.ndarray | None = None

    # --- Tier 1: cross-lingual embeddings -----------------------------------

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s'...", self.config.embedding_model)
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalized embeddings (so dot product == cosine)."""
        model = self._load_model()
        return np.asarray(
            model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def prepare_corpus(self, descriptions: list[str], refs: list[str]) -> None:
        """Embed the source corpus (USDA/TürKomp descriptions) once."""
        self._corpus_descs = descriptions
        self._corpus_refs = refs
        self._corpus_mat = self.embed(descriptions)
        logger.info("Embedded %d source descriptions (once, offline)", len(descriptions))

    @staticmethod
    def match_in_corpus(
        query_vec: np.ndarray, corpus_mat: np.ndarray, threshold: float
    ) -> tuple[int | None, float]:
        """Best cosine match index (or None) + score. Pure numpy — no model needed."""
        if corpus_mat.size == 0:
            return None, 0.0
        sims = corpus_mat @ query_vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        return (best if score >= threshold else None), score

    def match(self, queries: list[str]) -> tuple[str | None, str | None, float]:
        """Match the best query (e.g. [en_name, tr_name]) to the corpus."""
        if self._corpus_mat is None or not queries:
            return None, None, 0.0
        best_desc, best_ref, best_score = None, None, 0.0
        for vec in self.embed(queries):
            idx, score = self.match_in_corpus(vec, self._corpus_mat, self.config.semantic_match_threshold)
            if idx is not None and score > best_score:
                best_desc = self._corpus_descs[idx]
                best_ref = self._corpus_refs[idx]
                best_score = score
        return best_desc, best_ref, best_score

    # --- Tier 2: optional LLM ------------------------------------------------

    def llm_resolve(self, food: TaxonomyFood) -> dict | None:
        """Offline LLM decomposition for hard/uncurated foods. Optional + guarded."""
        if not self.config.enable_llm_resolver:
            return None
        try:
            import json
            import os

            if not os.getenv("ANTHROPIC_API_KEY"):
                logger.warning("enable_llm_resolver set but ANTHROPIC_API_KEY missing; skipping")
                return None
            import anthropic

            client = anthropic.Anthropic()
            prompt = (
                f"Give typical per-100g macros for the dish '{food.name_en}' "
                f"(Turkish: '{food.name_tr}'). Reply with strict JSON only: "
                '{"calories":N,"protein":N,"fat":N,"carbs":N}.'
            )
            msg = client.messages.create(
                model=self.config.llm_model, max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(msg.content[0].text)
            return {
                "source": "llm", "source_ref": None,
                "calories": float(data["calories"]), "protein": float(data["protein"]),
                "fat": float(data["fat"]), "carbs": float(data["carbs"]), "priority": 50,
            }
        except Exception as exc:  # pragma: no cover - network/optional
            logger.warning("LLM resolve failed for %s: %s", food.slug, exc)
            return None

    # --- orchestration -------------------------------------------------------

    def resolve(self, food: TaxonomyFood) -> ResolvedFood:
        """Produce nutrition_facts + source aliases for one taxonomy food."""
        # Tier 0: curated facts (always present in our taxonomy).
        facts = [{
            "source": food.source, "source_ref": None,
            **food.primary_facts, "priority": 0,
        }]
        source_aliases: list[tuple[str, str]] = []
        method = "curated"

        # Tier 1: enrich with a matched source description (if a corpus is loaded).
        if self._corpus_mat is not None and food.usda_query:
            queries = [food.usda_query, food.name_en, food.name_tr]
            desc, ref, score = self.match(queries)
            if desc is not None:
                facts[0]["source_ref"] = ref
                source_aliases.append(("en", desc))
                method = "curated+embedding"
                logger.debug("%s ~ '%s' (score=%.2f)", food.slug, desc, score)

        # Tier 2: only if Tier 0 produced nothing (uncurated foods) and enabled.
        if not facts and (llm_fact := self.llm_resolve(food)):
            facts.append(llm_fact)
            method = "llm"

        return ResolvedFood(facts=facts, source_aliases=source_aliases, method=method)
