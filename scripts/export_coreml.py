#!/usr/bin/env python3
"""Export the YOLOv11-seg model to an Apple CoreML ``.mlpackage`` for the iOS app.

Wraps Ultralytics' CoreML exporter so the export is reproducible and lands in the
iOS app bundle with a metadata sidecar (model_version + class names + imgsz) that
the Swift side reads. ``imgsz`` is pinned to ``config.mask_resolution`` (640) so
on-device mask areas land on the same grid the stored ``ref_area`` and
``area_px_to_cm2`` assume.

Run in the ML env (ultralytics):

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_coreml.py            # v1 weights
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_coreml.py --pretrained
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_coreml.py --weights runs/train/foodyolo_v2/weights/best.pt \
        --model-version foodyolo_v2

Swap to v2 later: re-run with the v2 weights + ``--model-version foodyolo_v2``,
then flip ``active_model_version`` in the bundled DB (rebuild the KB).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUT = AppConfig().project_root / "ios" / "Lokma" / "Resources" / "FoodSeg.mlpackage"


def parse_args() -> argparse.Namespace:
    cfg = AppConfig()
    p = argparse.ArgumentParser(description="Export YOLOv11-seg -> CoreML .mlpackage")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--weights", type=Path, help=f"weights .pt (default: {cfg.model_path})")
    src.add_argument("--pretrained", action="store_true",
                     help="use the pretrained yolo11n-seg.pt as a placeholder model")
    p.add_argument("--imgsz", type=int, default=cfg.mask_resolution, help="export resolution (must match mask_resolution)")
    p.add_argument("--model-version", default=cfg.model_version, help="model_version tag written to the sidecar")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="destination .mlpackage path")
    p.add_argument("--no-half", action="store_true", help="export fp32 instead of fp16")
    p.add_argument("--no-nms", action="store_true", help="do not fold NMS into the package")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = AppConfig()

    if args.pretrained:
        weights = cfg.pretrained_seg_model
    else:
        weights = args.weights or cfg.model_path
    weights = Path(weights)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Pass --weights PATH, use --pretrained for a placeholder, or train first."
        )
    if args.imgsz != cfg.mask_resolution:
        logger.warning(
            "imgsz=%d != config.mask_resolution=%d — on-device areas will not match "
            "the stored ref_area grid.", args.imgsz, cfg.mask_resolution,
        )

    from ultralytics import YOLO  # lazy — heavy, ML-env only

    logger.info("Loading weights: %s", weights)
    model = YOLO(str(weights))

    logger.info("Exporting CoreML (imgsz=%d, half=%s, nms=%s) ...",
                args.imgsz, not args.no_half, not args.no_nms)
    exported = model.export(
        format="coreml",
        imgsz=args.imgsz,
        nms=not args.no_nms,
        half=not args.no_half,
    )
    exported = Path(exported)
    logger.info("Ultralytics wrote: %s", exported)

    # Move into the app bundle location.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out) if out.is_dir() else out.unlink()
    shutil.move(str(exported), str(out))

    # Sidecar metadata: the Swift side maps class_id -> name -> FoodRecord, and
    # picks the matching class_map rows by model_version.
    names = getattr(model, "names", {}) or {}
    names = {int(k): v for k, v in names.items()}
    sidecar = out.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps({
        "model_version": args.model_version,
        "imgsz": args.imgsz,
        "task": "segment",
        "source_weights": str(weights),
        "names": names,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nCoreML package -> {out}")
    print(f"Sidecar        -> {sidecar}")
    print(f"model_version  =  {args.model_version}  ({len(names)} classes)")
    print("\nNext: `cd ios && xcodegen generate` (if needed), then build & run in Xcode.")


if __name__ == "__main__":
    main()
