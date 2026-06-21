"""Phase 7: Nutrition5K auto-labeler.

Uses the shipped, parity-tested ``depth_volume_engine`` as a feature extractor on
Nutrition5K's overhead RealSense depth: derive a whole-dish foreground mask (food =
what rises above the fitted table plane), integrate the 2.5D outer **volume V**, pair
with the dataset's true scale **mass M**, and emit the regression target

    fill_density  D = M / V   (g/cm³ of envelope volume = the combined ρ·(1−P))

We learn D directly because ``M = V·ρ·(1−P)`` has two unknowns and Nutrition5K's
multi-ingredient dishes give no clean per-dish ρ. Pure numpy + PIL (no torch); the
core is unit-tested on synthetic depth.

Run via ``scripts/ingest_nutrition5k.py``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lokma.config import AppConfig
from lokma.geometry.depth_volume_engine import DepthVolumeEngineService, DepthVolumeParams

_ENGINE = DepthVolumeEngineService()


@dataclass(frozen=True)
class DishLabel:
    """Auto-derived training label for one dish."""

    volume_cm3: float
    fill_density: float       # M / V
    coverage: float
    plane_residual_mm: float


# --- depth I/O --------------------------------------------------------------


def load_depth_mm(path: Path) -> np.ndarray:
    """Load a 16-bit RealSense ``depth_raw.png`` as float millimetres (0 = invalid).

    Authoritative scale from the Nutrition5K docs: "depth units of 10,000
    (1 meter = 10,000 units)" → 1 unit = 0.1 mm, so depth_mm = raw / 10. A few
    frames decode to implausibly small distances (e.g. < 100 mm); those are dropped
    downstream by the depth window in `foreground_mask`.
    """
    return np.asarray(Image.open(path), dtype=np.float64) / 10.0


def load_masses(csv_path: Path) -> dict[str, float]:
    """dish_id -> true mass (g) from ``dish_nutrition_values.csv``."""
    masses: dict[str, float] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                masses[row["dish_id"]] = float(row["mass"])
            except (KeyError, ValueError):
                continue
    return masses


# --- core (pure numpy, unit-tested) -----------------------------------------


def foreground_mask(
    depth_mm: np.ndarray, height_threshold_mm: float, *,
    center_frac: float = 0.6, plate_percentile: float = 75.0,
    depth_lo_mm: float = 100.0, depth_hi_mm: float = 1500.0,
) -> np.ndarray | None:
    """Food mask = pixels rising above the **plate** surface (not the table).

    The food's support surface is the plate, so the reference is the plate level —
    estimated as a high depth percentile of the central region (the plate shows,
    deeper, around the closer food). Food = pixels at least ``height_threshold_mm``
    nearer the overhead camera than the plate. A depth window drops sensor dropouts
    and far background. Returns a float {0,1} mask, or None if too few valid pixels.

    (Referencing the table instead grabs the whole plate — its volume swamps the
    food's and yields implausible fill-density; this plate-relative form fixes that.)
    """
    h, w = depth_mm.shape
    valid = (depth_mm > depth_lo_mm) & (depth_mm < depth_hi_mm)
    central = np.zeros((h, w), dtype=bool)
    my, mx = int(h * (1 - center_frac) / 2), int(w * (1 - center_frac) / 2)
    central[my:h - my, mx:w - mx] = True
    cv = depth_mm[central & valid]
    if cv.size < 100:
        return None
    plate_z = float(np.percentile(cv, plate_percentile))
    fg = central & valid & ((plate_z - depth_mm) > height_threshold_mm)
    return fg.astype(np.float64)


def label_dish(
    depth_mm: np.ndarray, mass_g: float, cfg: AppConfig,
    params: DepthVolumeParams | None = None,
) -> tuple[np.ndarray, DishLabel] | tuple[None, None]:
    """Auto-label one dish → (foreground mask, DishLabel), or (None, None) on QC fail."""
    if mass_g <= 0:
        return None, None
    h, w = depth_mm.shape
    mask = foreground_mask(depth_mm, cfg.n5k_height_threshold_mm,
                           center_frac=cfg.n5k_center_frac, plate_percentile=cfg.n5k_plate_percentile,
                           depth_lo_mm=cfg.n5k_depth_lo_mm, depth_hi_mm=cfg.n5k_depth_hi_mm)
    if mask is None or mask.sum() == 0:
        return None, None
    res = _ENGINE.integrate(depth_mm.flatten().tolist(), mask.flatten().tolist(), w, h,
                            cfg.n5k_fx, cfg.n5k_fy, cfg.n5k_cx, cfg.n5k_cy, params)
    if res is None or res.volume_cm3 <= 0:
        return None, None
    fill = mass_g / res.volume_cm3
    if not (cfg.n5k_fill_density_min <= fill <= cfg.n5k_fill_density_max):
        return None, None
    return mask, DishLabel(volume_cm3=res.volume_cm3, fill_density=fill,
                           coverage=res.coverage, plane_residual_mm=res.plane_residual_mm)


# --- manifest build (file I/O wrapper) --------------------------------------


def build_manifest(cfg: AppConfig | None = None, limit: int | None = None) -> dict:
    """Walk realsense_overhead dishes, auto-label, cache masks, write the manifest.

    Returns a stats dict (kept/dropped counts + fill-density distribution) — inspect
    it to validate the intrinsics before a full run.
    """
    cfg = cfg or AppConfig()
    overhead = cfg.nutrition5k_dir / "imagery" / "realsense_overhead"
    masses = load_masses(cfg.nutrition5k_dir / "dish_nutrition_values.csv")
    masks_dir = cfg.n5k_processed_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    dropped = {"no_mass": 0, "no_depth": 0, "bad_depth": 0, "qc_fail": 0}
    dishes = sorted(d for d in overhead.iterdir() if d.is_dir())
    if limit is not None:
        dishes = dishes[:limit]

    for dish in dishes:
        dish_id = dish.name
        depth_path, rgb_path = dish / "depth_raw.png", dish / "rgb.png"
        if dish_id not in masses:
            dropped["no_mass"] += 1; continue
        if not depth_path.exists() or not rgb_path.exists():
            dropped["no_depth"] += 1; continue
        try:
            depth_mm = load_depth_mm(depth_path)
        except Exception:               # corrupt / partial (.gstmp) PNGs
            dropped["bad_depth"] += 1; continue
        mask, label = label_dish(depth_mm, masses[dish_id], cfg)
        if label is None:
            dropped["qc_fail"] += 1; continue
        mask_path = masks_dir / f"{dish_id}.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        manifest.append({
            "dish_id": dish_id,
            "image_path": str(rgb_path),
            "depth_path": str(depth_path),
            "mask_path": str(mask_path),
            "calculated_volume_cm3": label.volume_cm3,
            "true_mass_g": masses[dish_id],
            "fill_density": label.fill_density,
            "coverage": label.coverage,
            "plane_residual_mm": label.plane_residual_mm,
        })

    out = cfg.n5k_processed_dir / "dataset_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return _manifest_stats(manifest, dropped, out)


def _manifest_stats(manifest: list[dict], dropped: dict, out: Path) -> dict:
    fills = np.array([m["fill_density"] for m in manifest], dtype=np.float64)
    stats = {
        "kept": len(manifest),
        "dropped": dropped,
        "manifest_path": str(out),
        "fill_density": {
            "min": float(fills.min()) if fills.size else None,
            "p50": float(np.median(fills)) if fills.size else None,
            "mean": float(fills.mean()) if fills.size else None,
            "max": float(fills.max()) if fills.size else None,
        },
    }
    return stats


# --- intrinsic calibration (plate-as-ruler, Phase 7b) -----------------------


def plate_diameter_px(mask: np.ndarray) -> tuple[float, float]:
    """Robust outer (x, y) pixel diameter of a plate/blob region.

    The 90th percentile of per-row x-spans (and per-col y-spans). Unbiased for both
    a filled disc and an annulus (food-occluded centre): the rim reaches the full
    width on the centre rows, and the percentile shrugs off a few noisy rows. (A
    plain coordinate-percentile span under-reads a disc's true diameter ~14%.)
    """
    ys, xs = np.nonzero(mask > 0.5)
    if ys.size == 0:
        return 0.0, 0.0

    def span(group: np.ndarray, val: np.ndarray) -> float:
        n = int(group.max()) + 1
        mn = np.full(n, np.inf); mx = np.full(n, -np.inf)
        np.minimum.at(mn, group, val); np.maximum.at(mx, group, val)
        w = (mx - mn)[np.isfinite(mx - mn)]
        return float(np.percentile(w, 90)) if w.size else 0.0

    return span(ys, xs), span(xs, ys)   # x-diameter (per row), y-diameter (per col)


def calibrate_intrinsics(
    cfg: AppConfig | None = None, *, sample_n: int = 400, target_plate_cm: float | None = None,
    plate_tol_mm: float = 12.0, min_plate_px: int = 1500,
) -> dict:
    """Estimate the RealSense fx, fy from the plate as a physical ruler.

    Overhead, plate ≈ fronto-parallel at depth ``z``: a disc of known diameter
    ``D_real`` projects to ``d_px = f·D_real/z``, so ``f = d_px·z/D_real``. We take
    the plate level as the *central-region* median depth (the wide tray/table sits at
    the deepest level and would over-read), then measure the flat region's diameter.

    CAVEAT (documented honestly): this dataset does not permit a clean plate solve —
    simple depth thresholding can't isolate the round plate from the food blob it
    bears (under-reads → fx low) nor from the tray (over-reads → fx high). Empirically
    the estimate **brackets** fx ~400–940 depending on the level chosen; the D415
    nominal (~595–600, our seed) sits in between. So treat the output as a sanity
    bracket, NOT a value to bake blindly — absolute scale is better fixed by an
    on-device calibration constant against weighed references (Phase 8).
    """
    cfg = cfg or AppConfig()
    target_mm = (target_plate_cm or cfg.default_plate_diameter_cm) * 10.0
    overhead = cfg.nutrition5k_dir / "imagery" / "realsense_overhead"
    fxs, fys, zs, dxs, dys = [], [], [], [], []
    n = 0
    for dish in sorted(d for d in overhead.iterdir() if d.is_dir()):
        dp = dish / "depth_raw.png"
        if not dp.exists():
            continue
        try:
            depth = load_depth_mm(dp)
        except Exception:
            continue
        h, w = depth.shape
        central = np.zeros((h, w), dtype=bool)
        central[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)] = True
        valid = central & (depth > cfg.n5k_depth_lo_mm) & (depth < cfg.n5k_depth_hi_mm)
        if int(valid.sum()) < 3000:
            continue
        z_plate = float(np.median(depth[valid]))   # central level avoids the wide tray
        plate = valid & (np.abs(depth - z_plate) < plate_tol_mm)
        if int(plate.sum()) < min_plate_px:
            continue
        dx, dy = plate_diameter_px(plate.astype(np.float64))
        if dx <= 0 or dy <= 0:
            continue
        fxs.append(dx * z_plate / target_mm)
        fys.append(dy * z_plate / target_mm)
        zs.append(z_plate); dxs.append(dx); dys.append(dy)
        n += 1
        if n >= sample_n:
            break
    if not fxs:
        raise RuntimeError("calibration found no usable plates — check data paths / depth window")
    fx, fy = float(np.median(fxs)), float(np.median(fys))
    return {
        "n_dishes": n,
        "fx": fx, "fy": fy,
        "old_fx": cfg.n5k_fx, "old_fy": cfg.n5k_fy,
        "scale_x": fx / cfg.n5k_fx, "scale_y": fy / cfg.n5k_fy,
        "plate_real_cm": target_mm / 10.0,
        "plate_diam_px_x_med": float(np.median(dxs)),
        "plate_diam_px_y_med": float(np.median(dys)),
        "plate_z_med_mm": float(np.median(zs)),
    }
