#!/usr/bin/env python3
"""Auto-label Nutrition5K into a fill-density training manifest (Phase 7).

Runs the parity-tested depth_volume_engine over the overhead RealSense depth to get
outer volume V, pairs it with the true scale mass M, and emits D = M/V per dish to
data/processed/nutrition5k/dataset_manifest.json (+ cached foreground masks).

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/ingest_nutrition5k.py [--limit N]

Inspect the printed fill-density distribution to VALIDATE the RealSense intrinsics
(config n5k_fx/fy) before a full run: a typical dish should land ~0.3–1.3 g/cm³.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402
from lokma.training.nutrition5k import build_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap dishes (smoke test)")
    args = ap.parse_args()

    stats = build_manifest(AppConfig(), limit=args.limit)
    print(f"Wrote {stats['kept']} manifest rows -> {stats['manifest_path']}")
    print(f"Dropped: {stats['dropped']}")
    fd = stats["fill_density"]
    if fd["p50"] is not None:
        print(f"Fill-density D=M/V (g/cm³): min={fd['min']:.3f} p50={fd['p50']:.3f} "
              f"mean={fd['mean']:.3f} max={fd['max']:.3f}")
        print("  -> expect a typical dish ~0.3–1.3 g/cm³; a constant offset means "
              "the n5k_fx/fy intrinsics need scaling.")
    else:
        print("No dishes passed QC — check intrinsics / data paths.")


if __name__ == "__main__":
    main()
