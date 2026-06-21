#!/usr/bin/env python3
"""Plate-anchored RealSense intrinsic calibration for Nutrition5K (Phase 7b).

Solves fx, fy from the plate as a physical ruler (fx = plate_diameter_px ·
z_plate / plate_real_mm), independent of any food-density prior, so the auto-
labeled volume V is physically correct cm³ and transfers to the ARKit-intrinsic
iPhone. Prints the resolved intrinsics + scaling vector to bake into config.

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/calibrate_nutrition5k.py [--sample N --plate-cm 27]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402
from lokma.training.nutrition5k import calibrate_intrinsics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--plate-cm", type=float, default=None, help="real plate diameter (default: config 27)")
    args = ap.parse_args()

    r = calibrate_intrinsics(AppConfig(), sample_n=args.sample, target_plate_cm=args.plate_cm)
    print(f"dishes used:       {r['n_dishes']}   plate_z_med={r['plate_z_med_mm']:.0f} mm   "
          f"target_plate={r['plate_real_cm']:.1f} cm")
    print(f"plate diam px x/y: {r['plate_diam_px_x_med']:.0f} / {r['plate_diam_px_y_med']:.0f}")
    print(f"OLD  fx, fy:       {r['old_fx']:.1f}, {r['old_fy']:.1f}")
    print(f"RESOLVED fx, fy:   {r['fx']:.1f}, {r['fy']:.1f}")
    print(f"SCALING VECTOR:    ({r['scale_x']:.3f}, {r['scale_y']:.3f})   (new/old)")
    print(f"implied D rescale: x{r['scale_x'] * r['scale_y']:.3f}   (V ∝ 1/(fx·fy))")
    print("\nNOTE: this dataset can't cleanly isolate the plate, so the estimate BRACKETS fx")
    print("(food-biased low ~400 / tray-biased high ~940); the D415 nominal ~595 sits between.")
    print("Do NOT bake blindly — fix absolute scale via an on-device calibration constant (Phase 8).")


if __name__ == "__main__":
    main()
