"""CalibrationService — converts pixel area to real-world area via pinhole geometry.

This replaces the old ``calculate_real_metrics`` with its dimensionally-unsound
magic ``SENSOR_FACTOR = 0.0035``. The correct projected-area relation is::

    A_real = A_px · (D / f_px)²            [f_px = focal length in working-grid pixels]

``f_px`` (from :meth:`CameraIntrinsics.from_mm` or, later, device intrinsics)
folds the sensor pixel pitch and the native→working resize ratio into one
physically meaningful number, so there is no free fudge factor to overfit.

This is the seam for Phase 2 (reference-object auto-calibration) and Phase 4
(ARKit intrinsics + LiDAR distance): subclass and override the two abstract
methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lokma.config import AppConfig
from lokma.core.exceptions import CalibrationError
from lokma.core.models import CameraIntrinsics, Detection


class CalibrationService(ABC):
    """Abstract source of camera intrinsics and per-detection distance."""

    @abstractmethod
    def get_intrinsics(self) -> CameraIntrinsics:
        """Return the working-grid camera intrinsics."""

    @abstractmethod
    def get_distance_mm(self, detection: Detection | None = None) -> float:
        """Return the camera→food distance in millimetres."""

    def px_area_to_cm2(self, area_px: float, distance_mm: float | None = None) -> float:
        """Project a pixel area (working grid) to real-world cm²."""
        intrinsics = self.get_intrinsics()
        if intrinsics.focal_px <= 0:
            raise CalibrationError("focal_px must be positive")
        if distance_mm is None:
            distance_mm = self.get_distance_mm()
        if distance_mm < 0:
            raise CalibrationError("distance_mm must be non-negative")
        real_area_mm2 = area_px * (distance_mm / intrinsics.focal_px) ** 2
        return real_area_mm2 / 100.0  # mm² -> cm²


class StaticCalibration(CalibrationService):
    """Fixed intrinsics and a single assumed distance (the Phase 1 default)."""

    def __init__(self, intrinsics: CameraIntrinsics, distance_mm: float) -> None:
        self._intrinsics = intrinsics
        self._distance_mm = distance_mm

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def get_distance_mm(self, detection: Detection | None = None) -> float:
        return self._distance_mm

    @classmethod
    def from_config(cls, config: AppConfig) -> "StaticCalibration":
        intrinsics = CameraIntrinsics.from_mm(
            focal_length_mm=config.focal_length_mm,
            pixel_pitch_mm=config.pixel_pitch_mm,
            native_resolution=config.native_resolution,
            working_resolution=config.working_resolution,
        )
        return cls(intrinsics=intrinsics, distance_mm=config.default_distance_mm)
