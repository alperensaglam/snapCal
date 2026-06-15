#!/usr/bin/env python3
"""LOKMA live webcam adapter.

The thin UI shell around :class:`InferencePipeline`: it owns frame capture, the
OpenCV window, and rendering — nothing else. All estimation logic lives in the
pipeline, so this file is the desktop analogue of the future iOS adapter.

Run from the repository root::

    python scripts/run_live.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/run_live.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from lokma.config import AppConfig  # noqa: E402
from lokma.pipeline.inference_pipeline import InferencePipeline  # noqa: E402

MASK_COLOR = (0, 255, 0)


def _render(frame, results, mask_threshold: float):
    """Overlay masks + labels onto the frame (in place) and return it."""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    for annotated in results:
        detection = annotated.detection
        full_mask = cv2.resize(detection.mask, (width, height))
        overlay[full_mask > mask_threshold] = MASK_COLOR
        x1, y1, _, _ = (int(v) for v in detection.bbox)
        cv2.putText(
            frame, annotated.label, (x1, max(y1 - 10, 12)),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, MASK_COLOR, 2,
        )
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AppConfig.from_env()
    pipeline = InferencePipeline.build(config)

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {config.camera_index}")

    print("LOKMA live detection — press 'q' to quit")
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            results = pipeline.process_frame(frame)
            frame = _render(frame, results, config.mask_threshold)
            cv2.imshow("LOKMA - Real Time Nutrition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    main()
