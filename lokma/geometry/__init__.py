"""Geometry layer: calibration resolver, natural-anchor detectors, volume engine, strategies."""

from lokma.geometry.anchor_detector import (
    NaturalAnchorDetector,
    PlateEllipseDetector,
    UtensilAnchorDetector,
)
from lokma.geometry.calibration_service import (
    CalibrationResolver,
    CalibrationService,
    IntrinsicDepthCalibration,
    IntrinsicPlaneCalibration,
    NaturalAnchorCalibration,
    StaticCalibration,
)
from lokma.geometry.strategies import (
    AutoStrategy,
    MassContext,
    MassEstimationStrategy,
    PixelRatioStrategy,
    VolumetricStrategy,
    get_strategy,
)
from lokma.geometry.volume_engine import VolumeEngineService

__all__ = [
    "AutoStrategy",
    "CalibrationResolver",
    "CalibrationService",
    "IntrinsicDepthCalibration",
    "IntrinsicPlaneCalibration",
    "MassContext",
    "MassEstimationStrategy",
    "NaturalAnchorCalibration",
    "NaturalAnchorDetector",
    "PixelRatioStrategy",
    "PlateEllipseDetector",
    "StaticCalibration",
    "UtensilAnchorDetector",
    "VolumeEngineService",
    "VolumetricStrategy",
    "get_strategy",
]
