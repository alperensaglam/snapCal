"""Typed data models — the boundaries between LOKMA services.

These dataclasses replace the ad-hoc tuples that previously flowed between the
script's functions (e.g. ``desc, kcal_100, prot_100, ... = nut_data``), which
were the source of brittle unpacking bugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

# --- Geometry / camera ------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics expressed in the pixel grid used for measurement.

    ``focal_px`` is the focal length in *pixels of the working grid*. Expressing
    it this way folds the sensor pixel pitch and the native->working resize ratio
    into a single, physically meaningful number — eliminating the old magic
    ``SENSOR_FACTOR`` constant.
    """

    focal_px: float
    image_size: tuple[int, int]
    principal_point: tuple[float, float] | None = None

    @classmethod
    def from_mm(
        cls,
        focal_length_mm: float,
        pixel_pitch_mm: float,
        native_resolution: tuple[int, int],
        working_resolution: tuple[int, int],
    ) -> "CameraIntrinsics":
        """Derive working-grid intrinsics from physical sensor parameters.

        focal_px(native) = focal_mm / pixel_pitch_mm, then scaled by the
        native->working width ratio so the focal length matches the grid on
        which mask areas are actually counted.
        """
        if pixel_pitch_mm <= 0:
            raise ValueError("pixel_pitch_mm must be positive")
        if native_resolution[0] <= 0:
            raise ValueError("native_resolution width must be positive")

        focal_px_native = focal_length_mm / pixel_pitch_mm
        scale = working_resolution[0] / native_resolution[0]
        focal_px = focal_px_native * scale
        cx = working_resolution[0] / 2.0
        cy = working_resolution[1] / 2.0
        return cls(focal_px=focal_px, image_size=working_resolution, principal_point=(cx, cy))


@dataclass(frozen=True)
class VolumeEstimate:
    """Output of the volume engine for a single detection."""

    volume_cm3: float
    shape: str
    height_cm: float
    real_area_cm2: float
    method: str


# --- Knowledge base ---------------------------------------------------------


@dataclass(frozen=True)
class FoodRecord:
    """A single row of the ``nutrition`` table, with macros stored per 100 g."""

    class_name: str
    source: str | None
    usda_desc: str | None
    calories_per_100g: float
    protein_per_100g: float
    fat_per_100g: float
    carbs_per_100g: float
    portion_g: float
    ref_area: float | None
    density: float | None
    geometric_shape: str | None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FoodRecord":
        """Build from a ``sqlite3.Row`` (or any name-keyed mapping)."""
        return cls(
            class_name=row["class_name"],
            source=row["source"],
            usda_desc=row["usda_desc"],
            calories_per_100g=_as_float(row["calories"]),
            protein_per_100g=_as_float(row["protein"]),
            fat_per_100g=_as_float(row["fat"]),
            carbs_per_100g=_as_float(row["carbs"]),
            portion_g=_as_float(row["portion_g"]),
            ref_area=_as_optional_float(row["ref_area"]),
            density=_as_optional_float(row["density"]),
            geometric_shape=row["geometric_shape"],
        )


# --- Inference results ------------------------------------------------------


@dataclass(frozen=True)
class DepthSample:
    """Co-registered depth + mask arrays in depth-map pixel space + that grid's
    intrinsics — the input to :class:`DepthVolumeEngineService`. Row-major lists of
    length ``width*height``; depth <= 0 marks an invalid pixel. Mirrors Swift
    ``DepthSample``."""

    depth_mm: list[float]
    mask: list[float]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Detection:
    """One segmented food instance from the YOLO model.

    ``mask`` is the raw model mask (native mask resolution, float [0, 1]).
    ``mask_area_px`` is the *thresholded* pixel count at the working resolution,
    fixing the old ``mask_resized.sum()`` soft-edge over-count.
    """

    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray
    bbox: tuple[float, float, float, float]
    mask_area_px: float                       # at working (640²) grid — pixel_ratio parity
    mask_area_px_frame: float = 0.0           # at native frame grid — volumetric/scale
    frame_size: tuple[int, int] = (0, 0)      # (W, H) of the source frame
    depth_sample: "DepthSample | None" = None  # Tier 2 LiDAR depth+mask (optional)
    predicted_porosity: float | None = None    # Phase 5: ML porosity P, else category fallback
    predicted_fill_density: float | None = None  # Phase 8: ML fill-density D=ρ·(1−P); mass=V·D
    predicted_volume_cm3: float | None = None  # Phase 9: RGB volume head (LiDAR-less); mass=V·D


@dataclass(frozen=True)
class MassEstimate:
    """Estimated mass for a detection and how it was derived."""

    grams: float
    method: str
    density_used: float | None = None
    volume_cm3: float | None = None
    calibration_source: str | None = None
    calibration_confidence: float | None = None


@dataclass(frozen=True)
class NutritionResult:
    """Macros scaled to an estimated portion mass (absolute grams / kcal)."""

    calories: float
    protein: float
    fat: float
    carbs: float

    @classmethod
    def from_food(cls, food: FoodRecord, grams: float) -> "NutritionResult":
        factor = grams / 100.0
        return cls(
            calories=food.calories_per_100g * factor,
            protein=food.protein_per_100g * factor,
            fat=food.fat_per_100g * factor,
            carbs=food.carbs_per_100g * factor,
        )


@dataclass(frozen=True)
class AnnotatedResult:
    """The structured output of the pipeline for one detection."""

    detection: Detection
    food: FoodRecord | None = None
    mass: MassEstimate | None = None
    nutrition: NutritionResult | None = None

    @property
    def is_identified(self) -> bool:
        """True when the detection matched a knowledge-base record."""
        return self.food is not None and self.mass is not None and self.nutrition is not None

    @property
    def label(self) -> str:
        """Human-readable overlay label."""
        name = self.detection.class_name.upper()
        if self.is_identified:
            assert self.mass is not None and self.nutrition is not None
            return f"{name}: ~{self.mass.grams:.0f}g | {self.nutrition.calories:.0f} kcal"
        return f"{name}: (no DB match)"


# --- helpers ----------------------------------------------------------------


def _as_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _as_optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


# --- Phase 2: frictionless calibration --------------------------------------


class CalibrationSource(str, Enum):
    """Which level of the calibration hierarchy produced a scale."""

    DEPTH_INTRINSICS = "depth_intrinsics"
    INTRINSICS_PLANE = "intrinsics_plane"
    NATURAL_ANCHOR = "natural_anchor"
    STATIC_DEFAULT = "static_default"


@dataclass(frozen=True)
class AnchorDetection:
    """A natural anchor (plate/utensil) of known real size found in the frame."""

    kind: str
    pixel_size: float                         # major axis (plate) / length (utensil), frame px
    assumed_real_mm: float
    confidence: float
    center: tuple[float, float] | None = None
    axes: tuple[float, float] | None = None   # (major, minor) for a plate ellipse
    angle_deg: float | None = None
    tilt_deg: float | None = None


#: Beyond this incidence angle the 1/cos(θ) area correction is clamped and the
#: volumetric path is gated off (AutoStrategy) — grazing views aren't trustworthy.
MAX_FORESHORTENING_TILT_DEG: float = 65.0


def _foreshortening(tilt_deg: float | None) -> float:
    """Area foreshortening factor cos(θ) for a plane viewed at incidence θ.

    A flat food patch viewed at tilt θ from top-down projects to cos(θ)× fewer
    pixels, so recovering its true footprint area divides by this factor. Unknown
    tilt → 1.0 (assume top-down); clamped at ``MAX_FORESHORTENING_TILT_DEG`` so the
    reciprocal can't blow up at grazing angles.
    """
    if tilt_deg is None:
        return 1.0
    clamped = min(max(tilt_deg, 0.0), MAX_FORESHORTENING_TILT_DEG)
    return max(math.cos(math.radians(clamped)), 1e-3)


@dataclass(frozen=True)
class ScaleEstimate:
    """The per-frame calibration result: how many millimetres a pixel spans."""

    mm_per_px: float
    source: CalibrationSource
    confidence: float
    tilt_deg: float | None = None
    homography: np.ndarray | None = None
    working_size: tuple[int, int] | None = None

    @property
    def mm2_per_px2(self) -> float:
        return self.mm_per_px ** 2

    def area_px_to_cm2(self, area_px: float) -> float:
        """Metric footprint area from pixel area, tilt-corrected.

        Top-down scaling ``area_px · mm²/px² / 100`` is exact only when the optical
        axis is perpendicular to the food plane; at incidence ``tilt_deg`` the
        footprint is foreshortened, so divide by ``cos(θ)`` (see ``_foreshortening``).
        """
        topdown = area_px * self.mm2_per_px2 / 100.0
        return topdown / _foreshortening(self.tilt_deg)

    def contour_area_to_cm2(self, contour: np.ndarray) -> float:
        """Geometry-aware path (Phase 4): warp a contour to metric space, then area.

        Falls back to scalar scaling when no homography is present.
        """
        import cv2  # lazy import keeps this module usable without OpenCV

        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
        if self.homography is not None:
            warped = cv2.perspectiveTransform(pts, np.asarray(self.homography, dtype=np.float32))
            return abs(cv2.contourArea(warped)) / 100.0
        return self.area_px_to_cm2(abs(cv2.contourArea(pts)))


@dataclass(frozen=True)
class FrameContext:
    """Per-frame capture bundle.

    Optional device fields are populated by ARKit/AVFoundation on iOS (Phase 4);
    desktop uses :meth:`simulated_topdown`. The calibration engine consumes
    whatever is available, so the same pipeline runs with or without a device.
    """

    frame: np.ndarray
    intrinsics: CameraIntrinsics | None = None
    tilt_deg: float | None = None
    gravity: tuple[float, float, float] | None = None
    depth_map: np.ndarray | None = None
    depth_mm: float | None = None

    @classmethod
    def simulated_topdown(cls, frame: np.ndarray, config: Any = None) -> "FrameContext":
        """Desktop/webcam: plain RGB, assume near top-down, no device intrinsics/depth."""
        return cls(frame=frame, intrinsics=None, tilt_deg=0.0)

    @classmethod
    def from_device(
        cls,
        frame: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None = None,
        tilt_deg: float | None = None,
        depth_map: np.ndarray | None = None,
        depth_mm: float | None = None,
        gravity: tuple[float, float, float] | None = None,
    ) -> "FrameContext":
        """Phase 4 hook: ARKit ARFrame.camera.intrinsics / sceneDepth / attitude."""
        return cls(
            frame=frame, intrinsics=intrinsics, tilt_deg=tilt_deg,
            depth_map=depth_map, depth_mm=depth_mm, gravity=gravity,
        )
