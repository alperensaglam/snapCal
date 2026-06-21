#!/usr/bin/env python3
"""Train the decoupled fill-density regression head (Phase 7).

Reads data/processed/nutrition5k/dataset_manifest.json (from ingest_nutrition5k.py)
and trains a small CNN+MLP to predict fill_density = M/V from food appearance +
depth-derived scalars. Weights -> runs/train/fill_head/fill_head.pt.

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_fill_head.py [--epochs N --limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402
from lokma.training.fill_regressor import run_training  # noqa: E402


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
