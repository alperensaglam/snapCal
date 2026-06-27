"""Unified multi-modal data pipeline.

Registers diverse dataset sources per modality (segmentation / nutrition /
multimodal), keeps each modality's training target strictly isolated, and bridges
arbitrary external dataset labels to canonical taxonomy slugs so the runtime
inference service resolves macros flawlessly. See
:mod:`lokma.data_pipeline.orchestrator`.

Import-light: the schema, the :class:`~lokma.data_pipeline.orchestrator.VocabularyBridge`,
and the dry-run planner are usable on the base interpreter; the heavy builders are
imported lazily only when an actual ingestion is run.
"""

from lokma.data_pipeline.orchestrator import (
    Modality,
    Orchestrator,
    PipelineConfig,
    PipelinePlan,
    SourcePlan,
    SourceSpec,
    VocabularyBridge,
    VocabularyError,
    VocabularyReport,
    default_pipeline_config,
    read_yolo_names,
)
from lokma.data_pipeline.vocab_map import DEFAULT_LABEL_ALIASES

__all__ = [
    "DEFAULT_LABEL_ALIASES",
    "Modality",
    "Orchestrator",
    "PipelineConfig",
    "PipelinePlan",
    "SourcePlan",
    "SourceSpec",
    "VocabularyBridge",
    "VocabularyError",
    "VocabularyReport",
    "default_pipeline_config",
    "read_yolo_names",
]
