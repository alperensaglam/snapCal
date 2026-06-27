"""Unified multi-modal data-pipeline orchestrator + Vocabulary Alignment Bridge.

LOKMA learns from three structurally different data modalities, each with its own
*target sink* and its own existing builder. This module does **not** reimplement
any ingestion; it registers input directories per modality, keeps each modality's
target strictly isolated, and reconciles external dataset labels to canonical
taxonomy slugs:

================  ==========================================  =====================================
Modality          Builds (target sink)                        Existing builder (delegated to)
================  ==========================================  =====================================
SEGMENTATION      YOLOv11-seg dataset (RGB + polygon masks)    training.dataset_builder.build_v2_dataset
NUTRITION         macro lookup (USDA / TürKomp text tables)    knowledge.builder.KnowledgeBaseBuilder
MULTIMODAL        RGB + depth + mass manifest (Nutrition5K)    training.nutrition5k.build_manifest
================  ==========================================  =====================================

**Target isolation:** a source declares exactly one :class:`Modality`, and the
orchestrator refuses to route a source whose ``fmt`` does not belong to that
modality (e.g. a segmentation export can never write the regression manifest).
This is the structural guard against training/data contamination.

**Vocabulary alignment:** :class:`VocabularyBridge` maps arbitrary raw dataset
labels (``turkish_lahmacun``) to canonical slugs (``lahmacun``) so the runtime
inference service resolves macros flawlessly.

Offline-only: nothing here runs on-device, so there is **no Swift/LokmaCore parity
contract** to mirror. The heavy builders (torch / ultralytics / pandas) are
imported lazily inside :meth:`Orchestrator.run`, so the schema + bridge + dry-run
:meth:`Orchestrator.plan` stay importable and unit-testable on the base interpreter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

from lokma.config import AppConfig
from lokma.data_pipeline.vocab_map import DEFAULT_LABEL_ALIASES
from lokma.knowledge import taxonomy

logger = logging.getLogger(__name__)


class VocabularyError(ValueError):
    """A vocabulary map points at a slug that does not exist in the taxonomy."""


class Modality(str, Enum):
    """The kind of data a source provides — fixes its target sink (isolation)."""

    SEGMENTATION = "segmentation"   # RGB + polygon masks -> YOLOv11-seg dataset
    NUTRITION = "nutrition"         # macro tables -> knowledge-base lookup
    MULTIMODAL = "multimodal"       # RGB + depth + mass -> regression manifest


#: Formats legal for each modality. A source whose ``fmt`` is not listed for its
#: declared modality is refused (cross-modality contamination guard).
_FORMATS_BY_MODALITY: dict[Modality, frozenset[str]] = {
    Modality.SEGMENTATION: frozenset({"yolo", "roboflow", "food101"}),
    Modality.NUTRITION: frozenset({"usda", "turkomp"}),
    Modality.MULTIMODAL: frozenset({"nutrition5k"}),
}


@dataclass(frozen=True)
class SourceSpec:
    """One registered input directory and the single modality it feeds.

    ``label_map`` augments the global :data:`DEFAULT_LABEL_ALIASES` for this source
    only (e.g. a Roboflow export whose classes use bespoke names); its targets are
    validated against the taxonomy exactly like the global map.
    """

    name: str
    modality: Modality
    input_dir: Path
    fmt: str = "yolo"
    label_map: Mapping[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Registered sources + the per-modality output sinks (sourced from AppConfig)."""

    sources: tuple[SourceSpec, ...] = ()
    seg_output_dir: Path | None = None          # YOLOv11-seg dataset dir
    nutrition_db_path: Path | None = None        # knowledge-base sqlite
    multimodal_manifest_dir: Path | None = None  # nutrition5k processed dir

    def sink_for(self, modality: Modality) -> Path | None:
        """The single output target for a modality (isolation: one sink each)."""
        return {
            Modality.SEGMENTATION: self.seg_output_dir,
            Modality.NUTRITION: self.nutrition_db_path,
            Modality.MULTIMODAL: self.multimodal_manifest_dir,
        }[modality]


def default_pipeline_config(cfg: AppConfig | None = None) -> PipelineConfig:
    """Register the project's known input dirs at their correct modalities.

    Reproduces today's wiring: the v2 retrain tree feeds segmentation, the USDA
    tables feed the nutrition lookup, and Nutrition5K feeds the regression manifest.
    """
    cfg = cfg or AppConfig()
    return PipelineConfig(
        sources=(
            SourceSpec("lokma_v2", Modality.SEGMENTATION, cfg.raw_v2_dir, fmt="yolo"),
            SourceSpec("usda_survey", Modality.NUTRITION, cfg.usda_survey_dir, fmt="usda"),
            SourceSpec("usda_foundation", Modality.NUTRITION, cfg.usda_foundation_dir, fmt="usda"),
            SourceSpec("nutrition5k", Modality.MULTIMODAL, cfg.nutrition5k_dir, fmt="nutrition5k"),
        ),
        seg_output_dir=cfg.yolo_v2_dataset_dir,
        nutrition_db_path=cfg.db_path,
        multimodal_manifest_dir=cfg.n5k_processed_dir,
    )


# --------------------------------------------------------------------------- #
# Vocabulary Alignment Bridge
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VocabularyReport:
    """Outcome of running labels through the bridge."""

    resolved: int = 0          # via an alias map
    identity: int = 0          # label was already a canonical slug
    skipped: int = 0           # unmapped (count of distinct unknown labels)
    unmapped: tuple[str, ...] = ()


class VocabularyBridge:
    """Maps arbitrary raw dataset labels → canonical taxonomy slugs.

    The alignment layer that lets external datasets (whose class names are not our
    slugs) feed the knowledge base and runtime macro lookup. Complements — does not
    replace — the DB-build ``food_alias`` / ``EntityResolver`` machinery, which
    resolves human-facing multilingual *names* at knowledge-base build time; the
    bridge operates earlier, on dataset *labels*.

    Two policies, by design:

    * **Targets — strict.** Every alias value must be a real taxonomy slug, else
      :class:`VocabularyError` at construction. A dangling target would silently
      mint a detection class the runtime can never resolve to macros.
    * **Inputs — lenient.** An unknown label is skipped (:meth:`resolve` returns
      ``None``), logged once, and collected into :meth:`report`. This keeps messy
      external datasets (with extra classes we don't model) from contaminating a
      run while making exactly what was dropped auditable.
    """

    def __init__(
        self,
        taxonomy_slugs: set[str] | None = None,
        *,
        golden_list: tuple[str, ...] = taxonomy.GOLDEN_LIST,
        aliases: Mapping[str, str] = DEFAULT_LABEL_ALIASES,
    ) -> None:
        self._slugs = set(taxonomy_slugs) if taxonomy_slugs is not None else set(taxonomy.by_slug())
        self._golden = tuple(golden_list)
        self._aliases = {self._norm(k): v for k, v in aliases.items()}
        self._unmapped: set[str] = set()
        self._resolved = 0
        self._identity = 0
        self.validate()

    @staticmethod
    def _norm(label: str) -> str:
        return label.lower().strip()

    def validate(self) -> None:
        """Fail fast if any alias target is not a real slug (would break runtime)."""
        bad = {a: t for a, t in self._aliases.items() if t not in self._slugs}
        if bad:
            raise VocabularyError(
                f"vocabulary maps {len(bad)} label(s) to unknown slug(s): "
                + ", ".join(f"{a!r}->{t!r}" for a, t in sorted(bad.items()))
            )
        # A valid slug that isn't in the Altın Liste still resolves for macros, but
        # has no segmentation class_id — note it without failing.
        for slug in sorted({t for t in self._aliases.values() if t not in self._golden}):
            logger.debug("alias target %r is a valid slug but absent from GOLDEN_LIST "
                         "(resolvable for macros, no segmentation class_id)", slug)

    def resolve(self, raw_label: str) -> str | None:
        """Canonical slug for a raw label, or ``None`` if unmapped (skip + report)."""
        key = self._norm(raw_label)
        if key in self._slugs:           # already canonical
            self._identity += 1
            return key
        slug = self._aliases.get(key)
        if slug is not None:
            self._resolved += 1
            return slug
        if key not in self._unmapped:
            self._unmapped.add(key)
            logger.warning("no canonical mapping for dataset label %r — skipping", raw_label)
        return None

    def class_id_for(self, slug: str) -> int | None:
        """The v2 segmentation ``class_id`` for a slug (GOLDEN_LIST index), or None."""
        try:
            return self._golden.index(slug)
        except ValueError:
            logger.warning("slug %r not in GOLDEN_LIST — no v2 segmentation class_id", slug)
            return None

    def report(self) -> VocabularyReport:
        """Snapshot of resolution outcomes since construction."""
        return VocabularyReport(
            resolved=self._resolved,
            identity=self._identity,
            skipped=len(self._unmapped),
            unmapped=tuple(sorted(self._unmapped)),
        )

    def with_overrides(self, label_map: Mapping[str, str] | None = None) -> "VocabularyBridge":
        """A *fresh*, independently-counted bridge layering per-source aliases.

        Used so a ``SourceSpec.label_map`` augments resolution without mutating the
        shared template, and so each source's :meth:`report` reflects only its own
        labels. The merged targets are validated the same (strict) way. Always
        returns a new instance, even for an empty/absent ``label_map``.
        """
        merged = dict(self._aliases)
        if label_map:
            merged.update({self._norm(k): v for k, v in label_map.items()})
        return VocabularyBridge(self._slugs, golden_list=self._golden, aliases=merged)


# --------------------------------------------------------------------------- #
# Lightweight YOLO data.yaml label reader (no PyYAML dependency)
# --------------------------------------------------------------------------- #
_NAME_MAP_RE = re.compile(r"^\s*\d+\s*:\s*(?P<name>.+?)\s*$")   # "  0: baklava"
_NAME_ITEM_RE = re.compile(r"^\s*-\s*(?P<name>.+?)\s*$")        # "  - baklava"


def _unquote(s: str) -> str:
    return s.strip().strip("'\"")


def read_yolo_names(data_yaml: Path) -> list[str]:
    """Class names from a YOLO ``data.yaml``, without a PyYAML dependency.

    Supports the block-map form this repo writes (``  0: name``), the block-list
    form (``  - name``), and the inline-flow form (``names: [a, b, c]``) common in
    Roboflow exports. Returns ``[]`` when the file is absent.
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        return []
    names: list[str] = []
    in_names = False
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            rest = line.split("names:", 1)[1].strip()
            if rest.startswith("[") and rest.endswith("]"):       # inline flow list
                return [_unquote(p) for p in rest[1:-1].split(",") if p.strip()]
            in_names = True
            continue
        if in_names:
            if line and not line[0].isspace():   # next top-level key ends the block
                break
            if (m := _NAME_MAP_RE.match(line) or _NAME_ITEM_RE.match(line)):
                names.append(_unquote(m.group("name")))
    return names


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourcePlan:
    """The dry-run resolution of one source: how its labels map, and where it lands."""

    name: str
    modality: Modality
    fmt: str
    input_dir: Path
    sink: Path | None
    #: raw dataset label -> (canonical slug | None, segmentation class_id | None)
    remap: dict[str, tuple[str | None, int | None]]
    report: VocabularyReport


@dataclass(frozen=True)
class PipelinePlan:
    """A side-effect-free description of what a run *would* do."""

    sources: tuple[SourcePlan, ...]
    unmapped: tuple[str, ...]   # union of unmapped labels across all sources


class Orchestrator:
    """Registers per-modality sources, enforces target isolation, and (lazily)
    delegates ingestion to the existing builders.

    :meth:`plan` is the side-effect-free core (resolves labels through the bridge,
    no heavy imports). :meth:`run` performs the actual ingestion by delegating to
    the three existing builders.
    """

    def __init__(
        self,
        config: PipelineConfig,
        bridge: VocabularyBridge | None = None,
        app: AppConfig | None = None,
    ) -> None:
        self.app = app or AppConfig()
        self.config = config
        self.bridge = bridge or VocabularyBridge()
        self._sources: list[SourceSpec] = []
        for spec in config.sources:
            self.register(spec)

    def register(self, spec: SourceSpec) -> None:
        """Add a source after checking its format belongs to its modality."""
        self._check_modality_format(spec)
        self._sources.append(spec)

    @property
    def sources(self) -> tuple[SourceSpec, ...]:
        return tuple(self._sources)

    @staticmethod
    def _check_modality_format(spec: SourceSpec) -> None:
        allowed = _FORMATS_BY_MODALITY[spec.modality]
        if spec.fmt not in allowed:
            raise ValueError(
                f"source {spec.name!r}: format {spec.fmt!r} is not valid for modality "
                f"{spec.modality.value!r} (allowed: {sorted(allowed)}) — refusing to "
                "route it to another modality's target (isolation)"
            )

    def _resolver_for(self, spec: SourceSpec) -> VocabularyBridge:
        return self.bridge.with_overrides(spec.label_map)

    def plan(self) -> PipelinePlan:
        """Resolve every enabled source's labels through the bridge — no side effects.

        Segmentation sources are read for their declared class names (``data.yaml``)
        and remapped to ``(slug, class_id)``. Nutrition/multimodal sources carry no
        class labels for the bridge (their targets come from the EntityResolver /
        per-dish manifest), so their remap is empty — they appear only to show the
        sink each lands in.
        """
        source_plans: list[SourcePlan] = []
        agg_unmapped: set[str] = set()
        for spec in self._sources:
            if not spec.enabled:
                continue
            sink = self.config.sink_for(spec.modality)
            remap: dict[str, tuple[str | None, int | None]] = {}
            report = VocabularyReport()
            if spec.modality is Modality.SEGMENTATION:
                resolver = self._resolver_for(spec)
                for label in read_yolo_names(Path(spec.input_dir) / "data.yaml"):
                    slug = resolver.resolve(label)
                    class_id = resolver.class_id_for(slug) if slug is not None else None
                    remap[label] = (slug, class_id)
                report = resolver.report()
                agg_unmapped |= set(report.unmapped)
            source_plans.append(SourcePlan(
                name=spec.name, modality=spec.modality, fmt=spec.fmt,
                input_dir=Path(spec.input_dir), sink=sink, remap=remap, report=report,
            ))
        return PipelinePlan(sources=tuple(source_plans), unmapped=tuple(sorted(agg_unmapped)))

    def run(self, modality: Modality | None = None) -> dict[str, object]:
        """Delegate ingestion to the existing builders (heavy deps imported lazily).

        ``NUTRITION`` and ``MULTIMODAL`` are corpus-level builds in the current
        architecture (the whole knowledge base / the whole manifest), so they run at
        most once per :meth:`run` regardless of how many sources feed them.
        Segmentation runs per enabled source. Pass ``modality`` to run a single lane.
        """
        results: dict[str, object] = {}
        built_corpus: set[Modality] = set()
        for spec in self._sources:
            if not spec.enabled or (modality is not None and spec.modality is not modality):
                continue
            self._check_modality_format(spec)
            if spec.modality is Modality.SEGMENTATION:
                results[spec.name] = self._run_segmentation(spec)
            elif spec.modality in built_corpus:
                logger.info("skip %r: %s corpus already built this run",
                            spec.name, spec.modality.value)
            elif spec.modality is Modality.NUTRITION:
                results[spec.name] = self._run_nutrition(spec)
                built_corpus.add(spec.modality)
            elif spec.modality is Modality.MULTIMODAL:
                results[spec.name] = self._run_multimodal(spec)
                built_corpus.add(spec.modality)
        return results

    def _run_segmentation(self, spec: SourceSpec) -> dict[str, int]:
        from lokma.training import dataset_builder

        resolver = self._resolver_for(spec)
        # Backward-compatible: passing a resolver remaps external label dirs -> slugs;
        # the builder's default (label_resolver=None) preserves today's ordinal flow.
        return dataset_builder.build_v2_dataset(self.app, label_resolver=resolver.resolve)

    def _run_nutrition(self, spec: SourceSpec) -> int:
        from lokma.knowledge.builder import KnowledgeBaseBuilder

        # Multi-source name/synonym merging stays the existing EntityResolver +
        # food_alias + priority machinery — not reimplemented here.
        return KnowledgeBaseBuilder(self.app).build()

    def _run_multimodal(self, spec: SourceSpec) -> dict:
        from lokma.training import nutrition5k

        return nutrition5k.build_manifest(self.app)


__all__ = [
    "Modality",
    "SourceSpec",
    "PipelineConfig",
    "default_pipeline_config",
    "VocabularyBridge",
    "VocabularyReport",
    "VocabularyError",
    "Orchestrator",
    "SourcePlan",
    "PipelinePlan",
    "read_yolo_names",
]


# Re-exported for type hints elsewhere; the resolver callable shape used by the
# segmentation builder hook.
LabelResolver = Callable[[str], "str | None"]
