"""Natural-anchor detectors — find a known-size object already in the frame.

The frictionless alternative to a coin/card: the **plate** (or a utensil) already
on the table is the ruler. ``cv2`` and ``ultralytics`` are imported lazily so this
module stays importable where they are absent (the pure unit-test interpreter).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from lokma.config import AppConfig
from lokma.core.models import AnchorDetection, Detection

#: Approximate real lengths (mm) of common utensils (bbox-diagonal proxy).
KNOWN_UTENSIL_MM: dict[str, float] = {
    "fork": 195.0,
    "knife": 230.0,
    "spoon": 190.0,
    "bowl": 150.0,
}


class NaturalAnchorDetector(ABC):
    """Finds the single best natural anchor in a frame, or ``None``."""

    @abstractmethod
    def detect(
        self, frame: np.ndarray, food_detections: list[Detection] | None = None
    ) -> AnchorDetection | None:
        ...


class PlateEllipseDetector(NaturalAnchorDetector):
    """Classical-CV plate detector: fit ellipses to large contours, pick the plate.

    A round plate viewed near top-down projects to an ellipse whose **major axis**
    is the least-foreshortened estimate of the true diameter; the minor/major ratio
    encodes tilt (``cos θ``) and feeds the confidence.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def detect(self, frame, food_detections=None):
        import cv2

        if frame is None or frame.size == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.config.plate_canny_low, self.config.plate_canny_high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        larger = float(max(w, h))
        cfg = self.config
        best: AnchorDetection | None = None
        best_score = 0.0

        for contour in contours:
            if len(contour) < 5:
                continue
            (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
            major = max(axis_a, axis_b)
            minor = min(axis_a, axis_b)
            if major <= 0:
                continue
            fraction = major / larger
            if not (cfg.plate_min_frame_fraction <= fraction <= cfg.plate_max_frame_fraction):
                continue
            axis_ratio = minor / major
            if axis_ratio < cfg.plate_min_axis_ratio:
                continue
            ellipse_area = math.pi * (major / 2.0) * (minor / 2.0)
            contour_area = cv2.contourArea(contour)
            if ellipse_area <= 0 or contour_area <= 0:
                continue
            fit = min(contour_area, ellipse_area) / max(contour_area, ellipse_area)
            size_score = min(fraction / 0.6, 1.0)
            score = fit * axis_ratio * size_score
            if score > best_score:
                best_score = score
                tilt = math.degrees(math.acos(max(0.0, min(1.0, axis_ratio))))
                best = AnchorDetection(
                    kind="plate",
                    pixel_size=major,
                    assumed_real_mm=cfg.default_plate_diameter_cm * 10.0,
                    confidence=float(min(0.85, score)),
                    center=(float(cx), float(cy)),
                    axes=(float(major), float(minor)),
                    angle_deg=float(angle),
                    tilt_deg=float(tilt),
                )
        return best


class UtensilAnchorDetector(NaturalAnchorDetector):
    """Optional COCO-based fallback anchor: detect a fork/knife/spoon/bowl.

    Reuses the existing ``models/pretrained/yolo11n.pt`` (COCO). Off by default;
    enable via ``config.enable_utensil_anchor``.
    """

    def __init__(self, config: AppConfig, model=None) -> None:
        self.config = config
        self._model = model

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.config.coco_model_path))
        return self._model

    def detect(self, frame, food_detections=None):
        model = self._load()
        results = model(frame, verbose=False)
        best: AnchorDetection | None = None
        best_conf = 0.0
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                name = model.names[int(result.boxes.cls[i])]
                if name not in KNOWN_UTENSIL_MM:
                    continue
                conf = float(result.boxes.conf[i])
                x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[i])
                length = math.hypot(x2 - x1, y2 - y1)
                if length > 0 and conf > best_conf:
                    best_conf = conf
                    best = AnchorDetection(
                        kind=name,
                        pixel_size=length,
                        assumed_real_mm=KNOWN_UTENSIL_MM[name],
                        confidence=float(min(0.70, conf)),
                        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    )
        return best
