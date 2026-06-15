"""Typed data models — the boundaries between LOKMA services.

These dataclasses replace the ad-hoc tuples that previously flowed between the
script's functions (e.g. ``desc, kcal_100, prot_100, ... = nut_data``), which
were the source of brittle unpacking bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    mask_area_px: float


@dataclass(frozen=True)
class MassEstimate:
    """Estimated mass for a detection and how it was derived."""

    grams: float
    method: str
    density_used: float | None = None
    volume_cm3: float | None = None


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
