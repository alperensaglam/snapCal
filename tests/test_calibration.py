"""Calibration math: CameraIntrinsics.from_mm + StaticCalibration scale (Phase 2 interface)."""

import numpy as np

from lokma.core.models import CameraIntrinsics, FrameContext
from lokma.geometry.calibration_service import StaticCalibration


def _ctx():
    return FrameContext(frame=np.zeros((10, 10, 3), dtype=np.uint8))


def test_static_scale_and_area():
    intr = CameraIntrinsics(focal_px=1000.0, image_size=(640, 640))
    scale = StaticCalibration(intr, distance_mm=1000.0).estimate_scale(_ctx(), [])
    assert scale is not None
    assert abs(scale.mm_per_px - 1.0) < 1e-9              # D/f = 1000/1000
    assert abs(scale.area_px_to_cm2(1000.0) - 10.0) < 1e-9  # 1000 mm² -> 10 cm²


def test_area_scales_with_distance_squared():
    intr = CameraIntrinsics(focal_px=800.0, image_size=(640, 640))
    near = StaticCalibration(intr, 500.0).estimate_scale(_ctx(), [])
    far = StaticCalibration(intr, 1000.0).estimate_scale(_ctx(), [])
    assert abs(far.area_px_to_cm2(2000.0) - 4.0 * near.area_px_to_cm2(2000.0)) < 1e-9


def test_from_mm_folds_resize_ratio():
    intr = CameraIntrinsics.from_mm(4.2, 0.0012, (4032, 3024), (640, 640))
    expected = (4.2 / 0.0012) * (640 / 4032)
    assert abs(intr.focal_px - expected) < 1e-6
    assert intr.principal_point == (320.0, 320.0)
