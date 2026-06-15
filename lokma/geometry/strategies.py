"""Mass-estimation strategies (Strategy pattern).

Two interchangeable ways to turn a detection into grams:

  * :class:`PixelRatioStrategy` — the **current MVP behavior**, preserved exactly
    (``grams = portion_g · mask_area_px / ref_area``). This is the Phase 1
    default so displayed numbers do not change.
  * :class:`VolumetricStrategy` — the *real* ``m = V · ρ`` path (calibration →
    volume engine → density). Fully implemented and unit-tested; it becomes the
    default in Phase 2 once calibration/distance are trustworthy.

Swapping strategies requires no change to the pipeline — only ``config.strategy``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from lokma.config import AppConfig
from lokma.core.exceptions import ConfigurationError
from lokma.core.models import Detection, FoodRecord, MassEstimate
from lokma.density.density_service import DensityService
from lokma.geometry.calibration_service import CalibrationService
from lokma.geometry.volume_engine import VolumeEngineService


@dataclass
class MassContext:
    """Dependencies a strategy may need, injected at pipeline build time."""

    calibration: CalibrationService
    volume_engine: VolumeEngineService
    density_service: DensityService
    config: AppConfig


class MassEstimationStrategy(ABC):
    """Interface: turn a detection + food record into a :class:`MassEstimate`."""

    name: str = "base"

    @abstractmethod
    def estimate(self, detection: Detection, food: FoodRecord, ctx: MassContext) -> MassEstimate:
        ...


class PixelRatioStrategy(MassEstimationStrategy):
    """2D heuristic: scale a known portion by the ratio of observed to reference area."""

    name = "pixel_ratio"

    def estimate(self, detection: Detection, food: FoodRecord, ctx: MassContext) -> MassEstimate:
        ref_area = food.ref_area if (food.ref_area and food.ref_area > 0) else ctx.config.default_ref_area
        scale = detection.mask_area_px / ref_area if ref_area > 0 else 0.0
        grams = (food.portion_g or 0.0) * scale
        return MassEstimate(grams=grams, method=self.name, density_used=None)


class VolumetricStrategy(MassEstimationStrategy):
    """Physical path: area → real area → volume → mass via ``m = V · ρ``."""

    name = "volumetric"

    def estimate(self, detection: Detection, food: FoodRecord, ctx: MassContext) -> MassEstimate:
        distance_mm = ctx.calibration.get_distance_mm(detection)
        real_area_cm2 = ctx.calibration.px_area_to_cm2(detection.mask_area_px, distance_mm)
        density, source = ctx.density_service.resolve(food)
        shape = food.geometric_shape or "prism"
        volume = ctx.volume_engine.estimate_volume(real_area_cm2, shape, ctx.config.default_height_cm)
        grams = volume.volume_cm3 * density
        return MassEstimate(
            grams=grams,
            method=f"{self.name}:{source.value}",
            density_used=density,
            volume_cm3=volume.volume_cm3,
        )


_STRATEGIES: dict[str, type[MassEstimationStrategy]] = {
    PixelRatioStrategy.name: PixelRatioStrategy,
    VolumetricStrategy.name: VolumetricStrategy,
}


def get_strategy(name: str) -> MassEstimationStrategy:
    """Instantiate a strategy by name (``config.strategy``)."""
    try:
        return _STRATEGIES[name]()
    except KeyError as exc:
        raise ConfigurationError(
            f"Unknown strategy '{name}'. Available: {sorted(_STRATEGIES)}"
        ) from exc
