"""Tests for the pinhole calibration math (the old SENSOR_FACTOR replacement)."""

from lokma.core.models import CameraIntrinsics
from lokma.geometry.calibration_service import StaticCalibration


def test_px_area_to_cm2_basic():
    # focal_px == distance -> (D/f) == 1, so area_px maps 1:1 to mm² (=> /100 cm²).
    intrinsics = CameraIntrinsics(focal_px=1000.0, image_size=(640, 640))
    calib = StaticCalibration(intrinsics, distance_mm=1000.0)
    assert abs(calib.px_area_to_cm2(1000.0) - 10.0) < 1e-9


def test_area_scales_with_distance_squared():
    intrinsics = CameraIntrinsics(focal_px=800.0, image_size=(640, 640))
    calib = StaticCalibration(intrinsics, distance_mm=500.0)
    near = calib.px_area_to_cm2(2000.0, 500.0)
    far = calib.px_area_to_cm2(2000.0, 1000.0)
    assert abs(far - 4.0 * near) < 1e-9


def test_from_mm_folds_resize_ratio():
    intrinsics = CameraIntrinsics.from_mm(
        focal_length_mm=4.2,
        pixel_pitch_mm=0.0012,
        native_resolution=(4032, 3024),
        working_resolution=(640, 640),
    )
    expected = (4.2 / 0.0012) * (640 / 4032)
    assert abs(intrinsics.focal_px - expected) < 1e-6
    assert intrinsics.principal_point == (320.0, 320.0)
