"""CalibrationResolver hierarchy, device-input levels, and the auto-gating strategy.

Pure-math tests — no OpenCV / YOLO needed (runs on the base interpreter).
"""

from dataclasses import replace

import numpy as np

from lokma.config import AppConfig
from lokma.core.models import (
    CalibrationSource,
    CameraIntrinsics,
    Detection,
    FoodRecord,
    FrameContext,
    ScaleEstimate,
)
from lokma.density.density_service import DensityService
from lokma.geometry.anchor_detector import NaturalAnchorDetector
from lokma.geometry.calibration_service import (
    CalibrationResolver,
    IntrinsicDepthCalibration,
    IntrinsicPlaneCalibration,
    NaturalAnchorCalibration,
    StaticCalibration,
)
from lokma.geometry.strategies import AutoStrategy, MassContext
from lokma.geometry.volume_engine import VolumeEngineService


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# --- device-input calibration levels ---------------------------------------


def test_intrinsic_depth_scale():
    intr = CameraIntrinsics(focal_px=500.0, image_size=(640, 480))
    ctx = FrameContext(frame=_frame(), intrinsics=intr, depth_mm=1000.0)
    scale = IntrinsicDepthCalibration().estimate_scale(ctx, [])
    assert scale is not None
    assert scale.source == CalibrationSource.DEPTH_INTRINSICS
    assert abs(scale.mm_per_px - 2.0) < 1e-9  # 1000 / 500


def test_intrinsic_depth_requires_depth():
    intr = CameraIntrinsics(focal_px=500.0, image_size=(640, 480))
    ctx = FrameContext(frame=_frame(), intrinsics=intr)  # no depth
    assert IntrinsicDepthCalibration().estimate_scale(ctx, []) is None


def test_intrinsic_depth_from_depth_map_median():
    intr = CameraIntrinsics(focal_px=500.0, image_size=(640, 480))
    depth = np.full((480, 640), 1500.0, dtype=np.float32)
    ctx = FrameContext(frame=_frame(), intrinsics=intr, depth_map=depth)
    scale = IntrinsicDepthCalibration().estimate_scale(ctx, [])
    assert abs(scale.mm_per_px - 3.0) < 1e-6  # 1500 / 500


def test_intrinsic_plane_uses_tilt_without_depth():
    cfg = AppConfig()
    intr = CameraIntrinsics(focal_px=500.0, image_size=(640, 480))
    ctx = FrameContext(frame=_frame(), intrinsics=intr, tilt_deg=10.0)
    scale = IntrinsicPlaneCalibration(cfg).estimate_scale(ctx, [])
    assert scale is not None
    assert scale.source == CalibrationSource.INTRINSICS_PLANE


# --- resolver fallback ------------------------------------------------------


def test_resolver_prefers_highest_confidence():
    cfg = AppConfig()
    intr = CameraIntrinsics(focal_px=500.0, image_size=(640, 480))
    ctx = FrameContext(frame=_frame(), intrinsics=intr, depth_mm=900.0)
    resolver = CalibrationResolver([
        IntrinsicDepthCalibration(),
        StaticCalibration.from_config(cfg),
    ])
    assert resolver.estimate_scale(ctx, []).source == CalibrationSource.DEPTH_INTRINSICS


def test_resolver_falls_back_to_static_when_nothing_found():
    cfg = replace(AppConfig(), assume_default_plate=False)

    class _Nothing(NaturalAnchorDetector):
        def detect(self, frame, food_detections=None):
            return None

    resolver = CalibrationResolver([
        NaturalAnchorCalibration(_Nothing(), cfg),
        StaticCalibration.from_config(cfg),
    ])
    scale = resolver.estimate_scale(FrameContext(frame=_frame()), [])
    assert scale.source == CalibrationSource.STATIC_DEFAULT


def test_natural_anchor_assumed_plate_when_enabled():
    cfg = AppConfig()  # assume_default_plate=True by default

    class _Nothing(NaturalAnchorDetector):
        def detect(self, frame, food_detections=None):
            return None

    scale = NaturalAnchorCalibration(_Nothing(), cfg).estimate_scale(FrameContext(frame=_frame()), [])
    assert scale is not None
    assert scale.source == CalibrationSource.NATURAL_ANCHOR
    assert scale.confidence == 0.30  # assumed, not measured


# --- auto-gating strategy ---------------------------------------------------


def _food(**overrides) -> FoodRecord:
    base = dict(
        class_name="apple_pie", source="s", usda_desc="x",
        calories_per_100g=100.0, protein_per_100g=0.0, fat_per_100g=0.0,
        carbs_per_100g=0.0, portion_g=200.0, ref_area=40000.0,
        density=0.6, geometric_shape="prism",
    )
    base.update(overrides)
    return FoodRecord(**base)


def _det() -> Detection:
    return Detection(
        class_id=0, class_name="apple_pie", confidence=0.9,
        mask=np.zeros((4, 4), dtype=np.float32), bbox=(0, 0, 10, 10),
        mask_area_px=40000.0, mask_area_px_frame=40000.0, frame_size=(640, 480),
    )


def _ctx(scale):
    return MassContext(
        scale=scale, volume_engine=VolumeEngineService(),
        density_service=DensityService(), config=AppConfig(),
    )


def test_auto_uses_volumetric_when_confident():
    scale = ScaleEstimate(mm_per_px=2.0, source=CalibrationSource.DEPTH_INTRINSICS, confidence=0.9)
    est = AutoStrategy().estimate(_det(), _food(), _ctx(scale))
    assert est.method.startswith("volumetric")
    assert est.calibration_confidence == 0.9
    assert est.volume_cm3 is not None


def test_auto_falls_back_to_pixel_ratio_when_unconfident():
    scale = ScaleEstimate(mm_per_px=2.0, source=CalibrationSource.STATIC_DEFAULT, confidence=0.2)
    est = AutoStrategy().estimate(_det(), _food(), _ctx(scale))
    assert est.method.startswith("pixel_ratio")
    assert "uncalibrated" in est.method
    assert est.calibration_source == CalibrationSource.STATIC_DEFAULT.value
