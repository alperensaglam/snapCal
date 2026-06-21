"""Phase 7 auto-labeler core — pure numpy (no torch/dataset needed).

Synthetic overhead scene: a flat table plane with a raised box. Asserts the
depth-foreground mask finds the box, the volume matches a direct engine call, and
fill_density = M / V.
"""

from dataclasses import replace

import numpy as np

from lokma.config import AppConfig
from lokma.geometry.depth_volume_engine import DepthVolumeEngineService
from lokma.training.nutrition5k import foreground_mask, label_dish

W, H = 64, 48
# Intrinsics matched to the small synthetic grid (override the RealSense defaults).
CFG = replace(AppConfig(), n5k_fx=60.0, n5k_fy=60.0, n5k_cx=32.0, n5k_cy=24.0,
              n5k_height_threshold_mm=8.0)
TABLE_Z, BOX_Z = 400.0, 370.0   # box rises 30 mm toward the overhead camera


def _scene():
    depth = np.full((H, W), TABLE_Z, dtype=np.float64)
    depth[18:30, 24:40] = BOX_Z            # central raised box → food
    return depth


def test_foreground_mask_finds_the_box():
    depth = _scene()
    fg = foreground_mask(depth, CFG.n5k_height_threshold_mm)
    assert fg is not None
    box = np.zeros((H, W), dtype=np.float64)
    box[18:30, 24:40] = 1.0
    assert np.array_equal(fg, box)         # exactly the raised region, nothing else


def test_label_volume_matches_engine_and_fill_density():
    depth = _scene()
    mass_g = 250.0
    mask, label = label_dish(depth, mass_g, CFG)
    assert label is not None

    # Volume must equal a direct engine call on the same foreground mask.
    res = DepthVolumeEngineService().integrate(
        depth.flatten().tolist(), mask.flatten().tolist(), W, H,
        CFG.n5k_fx, CFG.n5k_fy, CFG.n5k_cx, CFG.n5k_cy)
    assert abs(label.volume_cm3 - res.volume_cm3) < 1e-9
    assert label.volume_cm3 > 0
    # D = M / V, and coverage is full (every box pixel has valid depth).
    assert abs(label.fill_density - mass_g / label.volume_cm3) < 1e-9
    assert label.coverage == 1.0


def test_plate_diameter_and_intrinsic_solve():
    from lokma.training.nutrition5k import plate_diameter_px

    H2, W2, r, cyc, cxc = 200, 200, 60, 100, 100
    yy, xx = np.ogrid[:H2, :W2]
    disc = ((yy - cyc) ** 2 + (xx - cxc) ** 2 <= r * r).astype(np.float64)
    dx, dy = plate_diameter_px(disc)
    assert abs(dx - 2 * r) <= 3 and abs(dy - 2 * r) <= 3          # outer diameter ~120 px

    # Annulus (food-occluded centre) — outer diameter still recovered.
    annulus = disc.copy()
    annulus[(yy - cyc) ** 2 + (xx - cxc) ** 2 <= (r // 2) ** 2] = 0.0
    ax, ay = plate_diameter_px(annulus)
    assert abs(ax - 2 * r) <= 3 and abs(ay - 2 * r) <= 3

    # fx solve recovers the planted focal length within ~3%.
    z, d_real_mm = 400.0, 270.0
    fx_true = (2 * r) * z / d_real_mm
    assert abs(dx * z / d_real_mm - fx_true) / fx_true < 0.03


def test_qc_rejects_zero_mass_and_implausible_density():
    depth = _scene()
    assert label_dish(depth, 0.0, CFG) == (None, None)         # no mass
    # A wildly heavy mass pushes D above the plausible band → rejected.
    assert label_dish(depth, 5_000_000.0, CFG) == (None, None)
