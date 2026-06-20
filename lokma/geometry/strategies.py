"""Mass-estimation strategies (Strategy pattern).

  * :class:`PixelRatioStrategy` — the Phase 1 2D heuristic (``portion_g · area/ref_area``).
  * :class:`VolumetricStrategy` — ``m = V·ρ`` from the per-frame :class:`ScaleEstimate`
    + a per-class height prior.
  * :class:`AutoStrategy` — **the default**: volumetric when calibration confidence
    clears ``config.calibration_confidence_threshold``, else pixel_ratio tagged
    ``"uncalibrated"``. This is the safe-activation gate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from lokma.config import AppConfig
from lokma.core.exceptions import ConfigurationError
from lokma.core.models import (
    MAX_FORESHORTENING_TILT_DEG,
    Detection,
    FoodRecord,
    MassEstimate,
    ScaleEstimate,
)
from lokma.density.categories import height_for, porosity_for
from lokma.density.density_service import DensityService
from lokma.geometry.depth_volume_engine import DepthVolumeEngineService
from lokma.geometry.volume_engine import VolumeEngineService


@dataclass
class MassContext:
    """Per-frame dependencies for a strategy. ``scale`` is recomputed each frame."""

    scale: ScaleEstimate | None
    volume_engine: VolumeEngineService
    density_service: DensityService
    config: AppConfig
    depth_engine: DepthVolumeEngineService = field(default_factory=DepthVolumeEngineService)


class MassEstimationStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def estimate(self, detection: Detection, food: FoodRecord, ctx: MassContext) -> MassEstimate:
        ...


class PixelRatioStrategy(MassEstimationStrategy):
    """2D heuristic: scale a known portion by the ratio of observed to reference area."""

    name = "pixel_ratio"

    def estimate(self, detection, food, ctx):
        ref_area = food.ref_area if (food.ref_area and food.ref_area > 0) else ctx.config.default_ref_area
        scale = detection.mask_area_px / ref_area if ref_area > 0 else 0.0
        grams = (food.portion_g or 0.0) * scale
        return MassEstimate(grams=grams, method=self.name, density_used=None)


def _effective_porosity(detection, food) -> float:
    """Effective porosity P for ``× (1 − P)``: the per-instance ML-predicted value when
    present, else the per-class category fallback (mirrors the DensityService hierarchy)."""
    if detection.predicted_porosity is not None:
        return detection.predicted_porosity
    return porosity_for(food.class_name)


class VolumetricStrategy(MassEstimationStrategy):
    """Physical path: metric footprint area → volume → mass via ``m = V·ρ·(1−P)``."""

    name = "volumetric"

    def estimate(self, detection, food, ctx):
        if ctx.scale is None:
            raise ConfigurationError("VolumetricStrategy requires a ScaleEstimate")
        # Native-frame area (square pixels) keeps the metric scaling correct.
        area_px = detection.mask_area_px_frame or detection.mask_area_px
        real_area_cm2 = ctx.scale.area_px_to_cm2(area_px)
        density, density_source = ctx.density_service.resolve(food)
        shape = food.geometric_shape or "prism"
        height_cm = height_for(food.class_name)
        volume = ctx.volume_engine.estimate_volume(real_area_cm2, shape, height_cm)
        grams = volume.volume_cm3 * density * (1.0 - _effective_porosity(detection, food))
        return MassEstimate(
            grams=grams,
            method=f"{self.name}:{density_source.value}",
            density_used=density,
            volume_cm3=volume.volume_cm3,
            calibration_source=ctx.scale.source.value,
            calibration_confidence=ctx.scale.confidence,
        )


def _tilt_trustworthy(tilt_deg: float | None) -> bool:
    """Volumetric area scaling degrades past a grazing incidence; gate it off there."""
    return tilt_deg is None or tilt_deg <= MAX_FORESHORTENING_TILT_DEG


class AutoStrategy(MassEstimationStrategy):
    """Safe activation: volumetric only when calibration is confident enough."""

    name = "auto"

    def __init__(self) -> None:
        self._volumetric = VolumetricStrategy()
        self._pixel_ratio = PixelRatioStrategy()

    def estimate(self, detection, food, ctx):
        # 1. Measured LiDAR depth volume — most accurate, and inherently tilt-robust
        #    (the fitted plane absorbs orientation), so it needs no tilt gate.
        if detection.depth_sample is not None:
            dv = ctx.depth_engine.integrate_sample(detection.depth_sample)
            if dv is not None:
                density, density_source = ctx.density_service.resolve(food)
                return MassEstimate(
                    grams=dv.volume_cm3 * density * (1.0 - _effective_porosity(detection, food)),
                    method=f"volumetric_depth:{density_source.value}",
                    density_used=density,
                    volume_cm3=dv.volume_cm3,
                    calibration_source="depth_plane",
                    calibration_confidence=dv.coverage,
                )
        # 2. Scalar volumetric when calibration is confident and not too oblique.
        scale = ctx.scale
        threshold = ctx.config.calibration_confidence_threshold
        if scale is not None and scale.confidence >= threshold and _tilt_trustworthy(scale.tilt_deg):
            return self._volumetric.estimate(detection, food, ctx)
        # 3. Not trustworthy enough (low confidence or too oblique) — fall back and mark it.
        base = self._pixel_ratio.estimate(detection, food, ctx)
        return MassEstimate(
            grams=base.grams,
            method=f"{base.method}:uncalibrated",
            density_used=base.density_used,
            volume_cm3=base.volume_cm3,
            calibration_source=(scale.source.value if scale else None),
            calibration_confidence=(scale.confidence if scale else None),
        )


_STRATEGIES: dict[str, type[MassEstimationStrategy]] = {
    PixelRatioStrategy.name: PixelRatioStrategy,
    VolumetricStrategy.name: VolumetricStrategy,
    AutoStrategy.name: AutoStrategy,
}


def get_strategy(name: str) -> MassEstimationStrategy:
    """Instantiate a strategy by name (``config.strategy``)."""
    try:
        return _STRATEGIES[name]()
    except KeyError as exc:
        raise ConfigurationError(
            f"Unknown strategy '{name}'. Available: {sorted(_STRATEGIES)}"
        ) from exc
