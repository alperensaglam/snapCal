"""InferencePipeline — headless orchestrator + DI composition root (Phase 2).

Each frame: segment → build detections (working-grid + native-grid areas) →
resolve a per-frame :class:`ScaleEstimate` via the :class:`CalibrationResolver`
(plate natural-anchor → static fallback; device depth/intrinsics light up on iOS)
→ run the (auto-gated) strategy → structured :class:`AnnotatedResult`s.

No camera or display side effects live here — that is the ``run_live`` adapter,
and on iOS the same logic runs behind an ARKit adapter that fills ``FrameContext``.
"""

from __future__ import annotations

import logging

import cv2
from ultralytics import YOLO

from lokma.config import AppConfig
from lokma.core.exceptions import ModelLoadError
from lokma.core.models import AnnotatedResult, Detection, FrameContext, NutritionResult
from lokma.density.density_service import DensityService
from lokma.geometry.anchor_detector import PlateEllipseDetector, UtensilAnchorDetector
from lokma.geometry.calibration_service import CalibrationResolver, NaturalAnchorCalibration
from lokma.geometry.strategies import MassContext, MassEstimationStrategy, get_strategy
from lokma.geometry.volume_engine import VolumeEngineService
from lokma.knowledge.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Segments a frame and produces per-detection mass + nutrition estimates."""

    def __init__(
        self,
        model: YOLO,
        database: DatabaseManager,
        resolver: CalibrationResolver,
        volume_engine: VolumeEngineService,
        density_service: DensityService,
        strategy: MassEstimationStrategy,
        config: AppConfig,
    ) -> None:
        self.model = model
        self.db = database
        self.resolver = resolver
        self.volume_engine = volume_engine
        self.density_service = density_service
        self.strategy = strategy
        self.config = config

    @classmethod
    def build(cls, config: AppConfig | None = None) -> "InferencePipeline":
        config = config or AppConfig()

        if not config.model_path.exists():
            raise ModelLoadError(
                f"Model weights not found at {config.model_path}. "
                "Check AppConfig.model_path or set LOKMA_MODEL_PATH."
            )

        logger.info("Loading YOLO segmentation model: %s", config.model_path)
        model = YOLO(str(config.model_path))

        database = DatabaseManager(config.db_path, read_only=True)
        database.load()

        resolver = CalibrationResolver.default(config, PlateEllipseDetector(config))
        if config.enable_utensil_anchor:
            # Insert the utensil anchor just above the static fallback.
            resolver.services.insert(
                len(resolver.services) - 1,
                NaturalAnchorCalibration(UtensilAnchorDetector(config), config),
            )

        strategy = get_strategy(config.strategy)
        logger.info("InferencePipeline ready (strategy=%s, conf=%.2f)", strategy.name, config.conf_threshold)
        return cls(model, database, resolver, VolumeEngineService(), DensityService(), strategy, config)

    def process_frame(
        self, frame, context: FrameContext | None = None
    ) -> list[AnnotatedResult]:
        """Run segmentation + estimation on one BGR frame. No display side effects."""
        context = context or FrameContext.simulated_topdown(frame, self.config)
        results = self.model(frame, conf=self.config.conf_threshold, verbose=False)

        detections: list[Detection] = []
        for result in results:
            if result.masks is None:
                continue
            masks = result.masks.data
            boxes = result.boxes
            for i in range(len(masks)):
                detections.append(self._build_detection(masks[i], boxes, i, frame.shape))

        # One calibration per frame, shared by every detection.
        scale = self.resolver.estimate_scale(context, detections)
        ctx = MassContext(
            scale=scale,
            volume_engine=self.volume_engine,
            density_service=self.density_service,
            config=self.config,
        )

        annotated: list[AnnotatedResult] = []
        for detection in detections:
            food = self.db.get_food_record(detection.class_name)
            if food is None:
                annotated.append(AnnotatedResult(detection=detection))
                continue
            mass = self.strategy.estimate(detection, food, ctx)
            # Safety cap: an implausible mass (false positive / runaway V) shows no
            # calories rather than an impossible number.
            if mass.grams > self.config.max_plausible_grams:
                annotated.append(AnnotatedResult(detection=detection, food=food))
                continue
            nutrition = NutritionResult.from_food(food, mass.grams)
            annotated.append(AnnotatedResult(detection, food, mass, nutrition))
        return annotated

    def _build_detection(self, mask_tensor, boxes, i: int, frame_shape) -> Detection:
        mask_np = mask_tensor.cpu().numpy()
        res = self.config.mask_resolution
        thr = self.config.mask_threshold

        # Working 640² grid — matches the stored ref_area (pixel_ratio parity).
        mask_sq = cv2.resize(mask_np, (res, res))
        area_sq = float((mask_sq > thr).sum())

        # Native frame grid — square camera pixels, correct for metric scaling.
        h, w = frame_shape[:2]
        mask_full = cv2.resize(mask_np, (w, h))
        area_frame = float((mask_full > thr).sum())

        class_id = int(boxes.cls[i])
        class_name = self.model.names[class_id]
        confidence = float(boxes.conf[i])
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
        return Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            mask=mask_np,
            bbox=(x1, y1, x2, y2),
            mask_area_px=area_sq,
            mask_area_px_frame=area_frame,
            frame_size=(w, h),
        )

    def close(self) -> None:
        self.db.close()
