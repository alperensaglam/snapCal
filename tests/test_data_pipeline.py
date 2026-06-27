"""Unified data pipeline: Vocabulary Alignment Bridge + per-modality isolation.

Pure-Python — no torch / ultralytics / pandas (the orchestrator imports the heavy
builders lazily inside ``run``, which is exercised only in the runbook with real
datasets). These tests lock the offline contract: label→slug alignment, strict
target validation, lenient skip+report on unknown inputs, modality/sink isolation,
and the side-effect-free ``plan`` dry-run.
"""

from __future__ import annotations

import pytest

from lokma.data_pipeline import (
    DEFAULT_LABEL_ALIASES,
    Modality,
    Orchestrator,
    PipelineConfig,
    SourceSpec,
    VocabularyBridge,
    VocabularyError,
    default_pipeline_config,
    read_yolo_names,
)
from lokma.data_pipeline.orchestrator import _FORMATS_BY_MODALITY
from lokma.knowledge import taxonomy
from lokma.training.dataset_builder import resolve_seg_dirs, write_data_yaml

ALL_SLUGS = set(taxonomy.by_slug())
# A real taxonomy slug deliberately absent from the Altın Liste (no v2 class_id).
NON_GOLDEN_SLUG = "baby_back_ribs"


# --------------------------------------------------------------------------- #
# Vocabulary Alignment Bridge
# --------------------------------------------------------------------------- #
def test_default_aliases_are_self_consistent():
    """The shipped DEFAULT_LABEL_ALIASES must only target real taxonomy slugs."""
    bridge = VocabularyBridge()  # validates on construction; raises if any target is bogus
    assert NON_GOLDEN_SLUG in ALL_SLUGS  # guards the fixture below
    for target in DEFAULT_LABEL_ALIASES.values():
        assert target in ALL_SLUGS


def test_identity_passthrough():
    bridge = VocabularyBridge()
    assert bridge.resolve("lahmacun") == "lahmacun"
    assert bridge.resolve("Doner") == "doner"  # case-insensitive normalize
    rep = bridge.report()
    assert rep.identity == 2
    assert rep.resolved == 0
    assert rep.skipped == 0


def test_alias_resolution():
    bridge = VocabularyBridge()
    assert bridge.resolve("turkish_lahmacun") == "lahmacun"
    assert bridge.resolve("DONER_KEBAB") == "doner"  # case-insensitive
    assert bridge.resolve("  rice_pilaf  ") == "pilav"  # whitespace-insensitive
    assert bridge.report().resolved == 3


def test_unmapped_label_is_skipped_and_reported():
    bridge = VocabularyBridge()
    assert bridge.resolve("mystery_label") is None
    assert bridge.resolve("another_unknown") is None
    assert bridge.resolve("mystery_label") is None  # repeat is deduped
    rep = bridge.report()
    assert rep.skipped == 2
    assert rep.unmapped == ("another_unknown", "mystery_label")  # sorted, deduped


def test_bad_target_raises_at_construction():
    with pytest.raises(VocabularyError):
        VocabularyBridge(aliases={"foo": "not_a_real_slug"})


def test_class_id_matches_golden_index():
    bridge = VocabularyBridge()
    assert bridge.class_id_for("baklava") == taxonomy.GOLDEN_LIST.index("baklava")
    assert bridge.class_id_for("breakfast_burrito") == taxonomy.GOLDEN_LIST.index("breakfast_burrito")


def test_valid_but_non_golden_slug_has_no_class_id():
    bridge = VocabularyBridge()
    # Still resolvable for macros (identity), but it has no segmentation class_id.
    assert bridge.resolve(NON_GOLDEN_SLUG) == NON_GOLDEN_SLUG
    assert bridge.class_id_for(NON_GOLDEN_SLUG) is None


def test_per_source_override_layers_on_globals():
    bridge = VocabularyBridge()
    scoped = bridge.with_overrides({"weird_kofte_name": "kofte"})
    assert scoped.resolve("weird_kofte_name") == "kofte"  # override
    assert scoped.resolve("turkish_lahmacun") == "lahmacun"  # globals still apply
    # The shared bridge is untouched by the override.
    assert bridge.resolve("weird_kofte_name") is None


def test_per_source_override_target_is_validated():
    bridge = VocabularyBridge()
    with pytest.raises(VocabularyError):
        bridge.with_overrides({"x": "not_a_real_slug"})


# --------------------------------------------------------------------------- #
# read_yolo_names (no PyYAML dependency)
# --------------------------------------------------------------------------- #
def test_read_yolo_names_block_map(tmp_path):
    path = write_data_yaml(tmp_path / "yolo", taxonomy.GOLDEN_LIST)
    names = read_yolo_names(path)
    assert names == list(taxonomy.GOLDEN_LIST)


def test_read_yolo_names_flow_list(tmp_path):
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 3\n"
        "names: ['turkish_lahmacun', 'doner_kebab', 'mystery_label']\n",
        encoding="utf-8",
    )
    assert read_yolo_names(yaml) == ["turkish_lahmacun", "doner_kebab", "mystery_label"]


def test_read_yolo_names_missing_file(tmp_path):
    assert read_yolo_names(tmp_path / "absent.yaml") == []


# --------------------------------------------------------------------------- #
# resolve_seg_dirs — backward-compatible label_resolver hook in dataset_builder
# --------------------------------------------------------------------------- #
def test_resolve_seg_dirs_default_is_ordinal_golden():
    plan = resolve_seg_dirs("/tmp/whatever")  # no resolver -> today's behavior
    assert [(cid, slug) for cid, slug, _ in plan] == list(enumerate(taxonomy.GOLDEN_LIST))


def test_resolve_seg_dirs_remaps_external_labels(tmp_path):
    (tmp_path / "turkish_lahmacun").mkdir()
    (tmp_path / "doner_kebab").mkdir()
    (tmp_path / "mystery_label").mkdir()  # unmapped -> dropped (no contamination)
    bridge = VocabularyBridge()
    plan = resolve_seg_dirs(tmp_path, label_resolver=bridge.resolve)
    by_slug = {slug: (cid, d.name) for cid, slug, d in plan}
    assert by_slug["lahmacun"] == (taxonomy.GOLDEN_LIST.index("lahmacun"), "turkish_lahmacun")
    assert by_slug["doner"] == (taxonomy.GOLDEN_LIST.index("doner"), "doner_kebab")
    assert "mystery_label" not in {slug for _, slug, _ in plan}


# --------------------------------------------------------------------------- #
# Config schema + target isolation
# --------------------------------------------------------------------------- #
def test_default_pipeline_config_registers_modalities():
    cfg = default_pipeline_config()
    by_name = {s.name: s for s in cfg.sources}
    assert by_name["lokma_v2"].modality is Modality.SEGMENTATION
    assert by_name["usda_survey"].modality is Modality.NUTRITION
    assert by_name["usda_foundation"].modality is Modality.NUTRITION
    assert by_name["nutrition5k"].modality is Modality.MULTIMODAL


def test_sink_for_isolates_targets():
    cfg = default_pipeline_config()
    assert cfg.sink_for(Modality.SEGMENTATION) == cfg.seg_output_dir
    assert cfg.sink_for(Modality.NUTRITION) == cfg.nutrition_db_path
    assert cfg.sink_for(Modality.MULTIMODAL) == cfg.multimodal_manifest_dir
    # Three distinct sinks — no two modalities share a target.
    sinks = {cfg.sink_for(m) for m in Modality}
    assert len(sinks) == 3


def test_every_modality_has_allowed_formats():
    for modality in Modality:
        assert _FORMATS_BY_MODALITY[modality]  # non-empty


def test_cross_modality_format_is_refused(tmp_path):
    bad = SourceSpec("bad", Modality.MULTIMODAL, tmp_path, fmt="yolo")  # yolo is seg-only
    with pytest.raises(ValueError, match="isolation"):
        Orchestrator(PipelineConfig(sources=(bad,)))


def test_register_rejects_mismatched_format(tmp_path):
    orch = Orchestrator(PipelineConfig())
    with pytest.raises(ValueError, match="isolation"):
        orch.register(SourceSpec("oops", Modality.NUTRITION, tmp_path, fmt="nutrition5k"))


# --------------------------------------------------------------------------- #
# plan() — side-effect-free dry run
# --------------------------------------------------------------------------- #
def test_plan_resolves_segmentation_labels_without_side_effects(tmp_path):
    src_dir = tmp_path / "roboflow_export"
    src_dir.mkdir()
    (src_dir / "data.yaml").write_text(
        "names: ['turkish_lahmacun', 'doner_kebab', 'mystery_label']\n", encoding="utf-8"
    )
    sink = tmp_path / "never_created" / "yolo_v2"
    config = PipelineConfig(
        sources=(SourceSpec("ext", Modality.SEGMENTATION, src_dir, fmt="roboflow"),),
        seg_output_dir=sink,
    )
    plan = Orchestrator(config).plan()

    assert len(plan.sources) == 1
    sp = plan.sources[0]
    assert sp.sink == sink
    assert sp.remap["turkish_lahmacun"] == ("lahmacun", taxonomy.GOLDEN_LIST.index("lahmacun"))
    assert sp.remap["doner_kebab"] == ("doner", taxonomy.GOLDEN_LIST.index("doner"))
    assert sp.remap["mystery_label"] == (None, None)
    assert sp.report.resolved == 2
    assert plan.unmapped == ("mystery_label",)
    # Strictly side-effect-free: the sink directory must not have been created.
    assert not sink.exists()


def test_plan_uses_per_source_overrides(tmp_path):
    src_dir = tmp_path / "custom"
    src_dir.mkdir()
    (src_dir / "data.yaml").write_text("names: ['house_special_kebap']\n", encoding="utf-8")
    config = PipelineConfig(
        sources=(SourceSpec(
            "custom", Modality.SEGMENTATION, src_dir, fmt="yolo",
            label_map={"house_special_kebap": "doner"},
        ),),
        seg_output_dir=tmp_path / "out",
    )
    plan = Orchestrator(config).plan()
    assert plan.sources[0].remap["house_special_kebap"] == ("doner", taxonomy.GOLDEN_LIST.index("doner"))
    assert plan.unmapped == ()


def test_plan_disabled_source_is_skipped(tmp_path):
    src_dir = tmp_path / "x"
    src_dir.mkdir()
    (src_dir / "data.yaml").write_text("names: ['doner_kebab']\n", encoding="utf-8")
    config = PipelineConfig(
        sources=(SourceSpec("x", Modality.SEGMENTATION, src_dir, fmt="yolo", enabled=False),),
        seg_output_dir=tmp_path / "out",
    )
    assert Orchestrator(config).plan().sources == ()


def test_plan_nutrition_source_has_empty_remap(tmp_path):
    config = PipelineConfig(
        sources=(SourceSpec("usda", Modality.NUTRITION, tmp_path, fmt="usda"),),
        nutrition_db_path=tmp_path / "kb.db",
    )
    plan = Orchestrator(config).plan()
    assert plan.sources[0].remap == {}  # bridge does not apply to text tables
    assert plan.sources[0].sink == tmp_path / "kb.db"
