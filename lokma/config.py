"""Central configuration — the single source of truth for every path and parameter.

This module dissolves the ``snapCal``/``lokma`` path fragility: nothing in the
codebase should hardcode a path or a tunable constant. ``AppConfig`` resolves
everything relative to the repository root (computed from ``__file__``), so the
project is portable across machines and immune to the working-directory
mismatches that previously broke the app at runtime.

Every field is overridable through ``LOKMA_*`` environment variables via
:meth:`AppConfig.from_env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# ``lokma/config.py`` -> parents[1] is the repository root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration.

    Construct the default with ``AppConfig()`` or an env-aware instance with
    ``AppConfig.from_env()``. Use :func:`dataclasses.replace` for ad-hoc tweaks.
    """

    # --- Core paths ---------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    #: Live knowledge base. Was incorrectly ``snapcal_local.db`` (non-existent).
    db_path: Path = PROJECT_ROOT / "data" / "processed" / "lokma_local.db"
    #: Trained YOLOv11-seg weights. Was an absolute ``/Desktop/snapCal/...`` path.
    model_path: Path = (
        PROJECT_ROOT
        / "runs" / "segment" / "runs" / "train"
        / "snapCal_v1_seg10" / "weights" / "best.pt"
    )
    #: Pretrained seg checkpoint used for training / auto-labeling.
    pretrained_seg_model: Path = PROJECT_ROOT / "models" / "pretrained" / "yolo11n-seg.pt"

    # --- Data sources (knowledge-base build) --------------------------------
    usda_survey_dir: Path = PROJECT_ROOT / "data" / "raw" / "usda" / "survey"
    usda_foundation_dir: Path = PROJECT_ROOT / "data" / "raw" / "usda" / "foundation"
    nutrition5k_dir: Path = PROJECT_ROOT / "data" / "raw" / "Nutrition5k-main"
    yolo_dataset_dir: Path = PROJECT_ROOT / "data" / "processed" / "yolo_dataset"
    val_labels_dir: Path = (
        PROJECT_ROOT / "data" / "processed" / "yolo_dataset" / "labels" / "val"
    )

    # --- Inference parameters ----------------------------------------------
    conf_threshold: float = 0.7
    #: Working resolution at which mask pixel area is measured.
    mask_resolution: int = 640
    #: Soft-mask -> binary threshold (fixes the old ``mask.sum()`` over-count).
    mask_threshold: float = 0.5
    #: Fallback reference area when a class has no stored ``ref_area``.
    default_ref_area: float = 40_000.0
    camera_index: int = 0
    #: Active mass-estimation strategy: ``"auto"`` (default), ``"pixel_ratio"``, or ``"volumetric"``.
    strategy: str = "auto"

    # --- Calibration defaults (Phase 2 tunable) -----------------------------
    # These feed the *correct* pinhole formula via CameraIntrinsics.from_mm and
    # replace the old magic ``SENSOR_FACTOR = 0.0035``. They only affect the
    # (non-default) volumetric strategy, so they are safe placeholders for now.
    focal_length_mm: float = 4.2
    pixel_pitch_mm: float = 0.0012  # ~1.2 µm, typical smartphone sensor
    native_resolution: tuple[int, int] = (4032, 3024)
    default_distance_mm: float = 300.0
    #: Default extrusion height for volume priors (cm).
    default_height_cm: float = 2.5

    # --- Knowledge-base build parameters ------------------------------------
    semantic_match_threshold: float = 0.65
    sentence_model_name: str = "all-MiniLM-L6-v2"

    # --- Phase 2: frictionless calibration ----------------------------------
    #: Assumed dinner-plate diameter (natural anchor), cm. Turkish main course ≈ 26-28.
    default_plate_diameter_cm: float = 27.0
    #: Volumetric activates only when calibration confidence >= this (safe activation).
    calibration_confidence_threshold: float = 0.5
    #: PlateEllipseDetector tuning.
    plate_canny_low: int = 50
    plate_canny_high: int = 150
    plate_min_frame_fraction: float = 0.30     # plate major axis / max(frame side)
    plate_max_frame_fraction: float = 1.05
    plate_min_axis_ratio: float = 0.55         # minor/major; below ⇒ too tilted to trust
    #: If no plate is found, optionally assume one spanning the frame (very low confidence).
    assume_default_plate: bool = True
    assumed_plate_frame_fraction: float = 0.70
    #: Optional COCO utensil anchor (loads models/pretrained/yolo11n.pt on demand).
    enable_utensil_anchor: bool = False
    coco_model_path: Path = PROJECT_ROOT / "models" / "pretrained" / "yolo11n.pt"

    @property
    def working_resolution(self) -> tuple[int, int]:
        """Square working resolution at which masks are measured."""
        return (self.mask_resolution, self.mask_resolution)

    @classmethod
    def from_env(cls, **overrides) -> "AppConfig":
        """Build a config, applying ``LOKMA_*`` env overrides then ``**overrides``."""
        env: dict[str, object] = {}
        if (v := os.getenv("LOKMA_DB_PATH")):
            env["db_path"] = Path(v)
        if (v := os.getenv("LOKMA_MODEL_PATH")):
            env["model_path"] = Path(v)
        if (v := os.getenv("LOKMA_STRATEGY")):
            env["strategy"] = v
        if (v := os.getenv("LOKMA_CAMERA_INDEX")):
            env["camera_index"] = int(v)
        if (v := os.getenv("LOKMA_DISTANCE_MM")):
            env["default_distance_mm"] = float(v)
        if (v := os.getenv("LOKMA_CONF_THRESHOLD")):
            env["conf_threshold"] = float(v)
        env.update(overrides)
        return replace(cls(), **env)


# Convenience module-level default for simple call sites.
DEFAULT_CONFIG = AppConfig()
