"""CalibrationService — the layered, frictionless scale resolver.

Per frame, a :class:`CalibrationResolver` asks each source for a :class:`ScaleEstimate`
and returns the most confident one, mirroring the DensityService fallback hierarchy:

  1. ``DEPTH_INTRINSICS`` — device intrinsics + metric depth (ARKit LiDAR)   [Phase 4 data]
  2. ``INTRINSICS_PLANE`` — device intrinsics + tilt + plane distance        [Phase 4 data]
  3. ``NATURAL_ANCHOR``   — a plate/utensil of known size detected in RGB    [LIVE]
  4. ``STATIC_DEFAULT``   — assumed top-down + config focal/distance         [LIVE fallback]

Levels 1–2 need device data carried on the :class:`FrameContext`; on desktop they
return ``None`` and the resolver falls through to the natural anchor. The Phase 1
pinhole math (``mm_per_px = D / f_px``) survives as Level 4.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from lokma.config import AppConfig
from lokma.core.models import (
    CalibrationSource,
    CameraIntrinsics,
    Detection,
    FrameContext,
    ScaleEstimate,
)
from lokma.geometry.anchor_detector import NaturalAnchorDetector

logger = logging.getLogger(__name__)


class CalibrationService(ABC):
    """A single source of per-frame scale. Returns ``None`` when not applicable."""

    @abstractmethod
    def estimate_scale(
        self, context: FrameContext, detections: list[Detection]
    ) -> ScaleEstimate | None:
        ...


class IntrinsicDepthCalibration(CalibrationService):
    """Level 1: exact scale from device intrinsics + metric depth (ARKit/LiDAR)."""

    confidence = 0.90

    def estimate_scale(self, context, detections):
        intr = context.intrinsics
        if intr is None or intr.focal_px <= 0:
            return None
        depth_mm = self._representative_depth(context)
        if depth_mm is None or depth_mm <= 0:
            return None
        return ScaleEstimate(
            mm_per_px=depth_mm / intr.focal_px,
            source=CalibrationSource.DEPTH_INTRINSICS,
            confidence=self.confidence,
            tilt_deg=context.tilt_deg,
            working_size=intr.image_size,
        )

    @staticmethod
    def _representative_depth(context):
        if context.depth_mm is not None:
            return float(context.depth_mm)
        if context.depth_map is not None:
            dm = context.depth_map
            valid = dm[np.isfinite(dm) & (dm > 0)]
            if valid.size:
                return float(np.median(valid))
        return None


class IntrinsicPlaneCalibration(CalibrationService):
    """Level 2: intrinsics + tilt + a plane distance (AR plane tracking, no LiDAR)."""

    confidence = 0.70

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def estimate_scale(self, context, detections):
        intr = context.intrinsics
        if intr is None or intr.focal_px <= 0 or context.tilt_deg is None:
            return None
        distance_mm = (
            context.depth_mm if context.depth_mm is not None else self.config.default_distance_mm
        )
        return ScaleEstimate(
            mm_per_px=distance_mm / intr.focal_px,
            source=CalibrationSource.INTRINSICS_PLANE,
            confidence=self.confidence,
            tilt_deg=context.tilt_deg,
            working_size=intr.image_size,
        )


class NaturalAnchorCalibration(CalibrationService):
    """Level 3: a plate/utensil of known real size detected in the RGB frame."""

    def __init__(self, detector: NaturalAnchorDetector, config: AppConfig) -> None:
        self.detector = detector
        self.config = config

    def estimate_scale(self, context, detections):
        anchor = self.detector.detect(context.frame, detections)
        if anchor is not None and anchor.pixel_size > 0:
            return ScaleEstimate(
                mm_per_px=anchor.assumed_real_mm / anchor.pixel_size,
                source=CalibrationSource.NATURAL_ANCHOR,
                confidence=anchor.confidence,
                tilt_deg=anchor.tilt_deg,
            )
        # "or assume a standard plate bound": last-ditch, very low confidence.
        if self.config.assume_default_plate and context.frame is not None and context.frame.size:
            h, w = context.frame.shape[:2]
            pixel_size = self.config.assumed_plate_frame_fraction * max(w, h)
            if pixel_size > 0:
                return ScaleEstimate(
                    mm_per_px=(self.config.default_plate_diameter_cm * 10.0) / pixel_size,
                    source=CalibrationSource.NATURAL_ANCHOR,
                    confidence=0.30,
                    tilt_deg=0.0,
                )
        return None


class StaticCalibration(CalibrationService):
    """Level 4: assumed top-down + config focal length & distance (always returns)."""

    confidence = 0.20

    def __init__(self, intrinsics: CameraIntrinsics, distance_mm: float) -> None:
        self._intrinsics = intrinsics
        self._distance_mm = distance_mm

    def estimate_scale(self, context, detections):
        focal_px = self._intrinsics.focal_px
        if focal_px <= 0:
            return None
        return ScaleEstimate(
            mm_per_px=self._distance_mm / focal_px,
            source=CalibrationSource.STATIC_DEFAULT,
            confidence=self.confidence,
            tilt_deg=0.0,
            working_size=self._intrinsics.image_size,
        )

    @classmethod
    def from_config(cls, config: AppConfig) -> "StaticCalibration":
        intrinsics = CameraIntrinsics.from_mm(
            focal_length_mm=config.focal_length_mm,
            pixel_pitch_mm=config.pixel_pitch_mm,
            native_resolution=config.native_resolution,
            working_resolution=config.working_resolution,
        )
        return cls(intrinsics=intrinsics, distance_mm=config.default_distance_mm)


class CalibrationResolver:
    """Tries calibration sources and returns the most confident valid ScaleEstimate."""

    def __init__(self, services: list[CalibrationService]) -> None:
        if not services:
            raise ValueError("CalibrationResolver needs at least one service")
        self.services = services

    def estimate_scale(
        self, context: FrameContext, detections: list[Detection]
    ) -> ScaleEstimate | None:
        best: ScaleEstimate | None = None
        for service in self.services:
            try:
                estimate = service.estimate_scale(context, detections)
            except Exception as exc:  # a flaky detector must not crash the frame
                logger.debug("%s failed: %s", type(service).__name__, exc)
                continue
            if estimate is None:
                continue
            if best is None or estimate.confidence > best.confidence:
                best = estimate
        return best

    @classmethod
    def default(cls, config: AppConfig, anchor_detector: NaturalAnchorDetector) -> "CalibrationResolver":
        return cls([
            IntrinsicDepthCalibration(),
            IntrinsicPlaneCalibration(config),
            NaturalAnchorCalibration(anchor_detector, config),
            StaticCalibration.from_config(config),
        ])
