"""PlateEllipseDetector on a synthetic plate + homography contour area.

Needs OpenCV; skipped automatically on interpreters without it.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lokma.config import AppConfig  # noqa: E402
from lokma.core.models import CalibrationSource, ScaleEstimate  # noqa: E402
from lokma.geometry.anchor_detector import PlateEllipseDetector  # noqa: E402


def test_plate_detector_recovers_scale():
    cfg = AppConfig()
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    diameter_px = 400
    cv2.circle(img, (400, 300), diameter_px // 2, (200, 200, 200), -1)  # top-down plate

    anchor = PlateEllipseDetector(cfg).detect(img)
    assert anchor is not None
    assert anchor.kind == "plate"
    # Major axis ~ the drawn diameter.
    assert abs(anchor.pixel_size - diameter_px) / diameter_px < 0.10
    mm_per_px = anchor.assumed_real_mm / anchor.pixel_size
    expected = (cfg.default_plate_diameter_cm * 10.0) / diameter_px
    assert abs(mm_per_px - expected) / expected < 0.10
    assert anchor.confidence > 0.5  # clearly top-down -> trustworthy


def test_contour_area_homography_roundtrip():
    # 100x100 px square, homography px->mm uniform 0.5 => 50mm x 50mm = 25 cm².
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    homography = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float32)
    scale = ScaleEstimate(
        mm_per_px=0.5, source=CalibrationSource.DEPTH_INTRINSICS,
        confidence=0.9, homography=homography,
    )
    assert abs(scale.contour_area_to_cm2(square) - 25.0) < 1e-3
