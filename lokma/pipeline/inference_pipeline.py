"""InferencePipeline — the headless orchestrator and dependency composition root.

This is the decoupled heart of the old ``predict_nutrition.py`` loop, with the
camera, the OpenCV window, and the import-time side effects removed. It accepts a
frame and returns structured :class:`AnnotatedResult` objects; rendering and
frame capture live in the thin ``scripts/run_live.py`` adapter. That separation
is exactly what lets Phase 4 reuse this logic behind an iOS/CoreML adapter.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from ultralytics import YOLO

from lokma.config import AppConfig
from lokma.core.exceptions import ModelLoadError
from lokma.core.models import AnnotatedResult, Detection, NutritionResult
from lokma.density.density_service import DensityService
from lokma.geometry.calibration_service import StaticCalibration
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
        strategy: MassEstimationStrategy,
        context: MassContext,
        config: AppConfig,
    ) -> None:
        self.model = model
        self.db = database
        self.strategy = strategy
        self.ctx = context
        self.config = config

    @classmethod
    def build(cls, config: AppConfig | None = None) -> "InferencePipeline":
        """Wire the model and all services from configuration (composition root)."""
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

        calibration = StaticCalibration.from_config(config)
        context = MassContext(
            calibration=calibration,
            volume_engine=VolumeEngineService(),
            density_service=DensityService(),
            config=config,
        )
        strategy = get_strategy(config.strategy)

        logger.info(
            "InferencePipeline ready (strategy=%s, conf=%.2f)", strategy.name, config.conf_threshold
        )
        return cls(model, database, strategy, context, config)

    def process_frame(self, frame: np.ndarray) -> list[AnnotatedResult]:
        """Run segmentation + estimation on one BGR frame. No display side effects."""
        results = self.model(frame, conf=self.config.conf_threshold, verbose=False)

        annotated: list[AnnotatedResult] = []
        for result in results:
            if result.masks is None:
                continue
            masks = result.masks.data
            boxes = result.boxes
            for i in range(len(masks)):
                detection = self._build_detection(masks[i], boxes, i)
                food = self.db.get_food_record(detection.class_name)
                if food is None:
                    annotated.append(AnnotatedResult(detection=detection))
                    continue
                mass = self.strategy.estimate(detection, food, self.ctx)
                nutrition = NutritionResult.from_food(food, mass.grams)
                annotated.append(AnnotatedResult(detection, food, mass, nutrition))
        return annotated

    def _build_detection(self, mask_tensor, boxes, i: int) -> Detection:
        mask_np = mask_tensor.cpu().numpy()
        res = self.config.mask_resolution
        mask_resized = cv2.resize(mask_np, (res, res))
        # Threshold the soft mask before counting (fixes the old .sum() over-count).
        area_px = float((mask_resized > self.config.mask_threshold).sum())

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
            mask_area_px=area_px,
        )

    def close(self) -> None:
        self.db.close()
