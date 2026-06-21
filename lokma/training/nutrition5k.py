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

    Nutrition5K mixes two depth encodings — some frames are 1 mm/unit, others
    0.1 mm/unit (raw values ~10× larger). Normalize by the nonzero median: a
    food-overhead camera sits ~0.3–0.7 m away, so a median > 1500 means the frame
    is 0.1 mm-encoded → scale to mm.
    """
    d = np.asarray(Image.open(path), dtype=np.float64)
    nz = d[d > 0]
    if nz.size and np.median(nz) > 1500.0:
        d = d * 0.1
    return d


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
