#!/usr/bin/env python3
"""Train the non-LiDAR volume regression head (Phase 9).

Reads data/processed/nutrition5k/dataset_manifest.json (from ingest_nutrition5k.py)
and trains a small CNN+MLP to predict log1p(volume_cm3) from the masked food crop +
mask_area_fraction (RGB only — no depth). Weights -> runs/train/height_head/height_head.pt.

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_height_head.py [--epochs N --limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402
from lokma.training.height_regressor import run_training  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap manifest rows (smoke test)")
    args = ap.parse_args()
    run_training(AppConfig(), epochs=args.epochs, batch=args.batch, lr=args.lr, limit=args.limit)


if __name__ == "__main__":
    main()
