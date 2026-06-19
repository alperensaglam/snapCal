#!/usr/bin/env python3
"""Emit golden parity vectors for the Swift port of the LOKMA math.

This drives the **real** Python classes (``lokma.core.models``,
``lokma.geometry.*``, ``lokma.density.*``) on a fixed set of synthetic inputs and
serialises the inputs + expected outputs to JSON. The Swift ``LokmaCore`` test
target loads the same JSON and asserts its port matches within a tight epsilon —
so the two implementations can never silently drift.

Pure math only: no DB, no model, no OpenCV. Runs on the base interpreter.

    python scripts/dump_golden_vectors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402
from lokma.core.models import (  # noqa: E402
    CameraIntrinsics,
    Detection,
    FoodRecord,
    FrameContext,
    NutritionResult,
    ScaleEstimate,
)
from lokma.density.categories import density_for, height_for  # noqa: E402
from lokma.density.density_service import DensityService  # noqa: E402
from lokma.geometry.calibration_service import (  # noqa: E402
    CalibrationResolver,
    IntrinsicDepthCalibration,
    IntrinsicPlaneCalibration,
    NaturalAnchorCalibration,
    StaticCalibration,
)
from lokma.geometry.strategies import (  # noqa: E402
    AutoStrategy,
    MassContext,
    PixelRatioStrategy,
    VolumetricStrategy,
)
from lokma.geometry.volume_engine import VolumeEngineService

# Default config carries the on-device constants the Swift AppConfig must mirror.
CFG = AppConfig()
VOL = VolumeEngineService()
DENS = DensityService()
DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _food(**kw) -> FoodRecord:
    """A FoodRecord with sensible defaults; override only what a case needs."""
    base = dict(
        class_name="generic", source="curated", usda_desc=None,
        calories_per_100g=200.0, protein_per_100g=5.0, fat_per_100g=8.0,
        carbs_per_100g=25.0, portion_g=150.0, ref_area=50_000.0,
        density=None, geometric_shape="prism",
    )
    base.update(kw)
    return FoodRecord(**base)


def _detection(area_sq: float, area_frame: float) -> Detection:
    return Detection(
        class_id=0, class_name="generic", confidence=0.9,
        mask=np.zeros((4, 4), dtype=np.float32), bbox=(0.0, 0.0, 1.0, 1.0),
        mask_area_px=area_sq, mask_area_px_frame=area_frame, frame_size=(640, 480),
    )


def _ctx(scale: ScaleEstimate | None) -> MassContext:
    return MassContext(scale=scale, volume_engine=VOL, density_service=DENS, config=CFG)


# --- case builders ----------------------------------------------------------


def intrinsics_from_mm() -> list[dict]:
    cases = []
    for focal_mm, pitch_mm, native, working in [
        (4.2, 0.0012, (4032, 3024), (640, 640)),
        (6.0, 0.0008, (4032, 3024), (640, 640)),
        (4.2, 0.0012, (1920, 1440), (640, 640)),
    ]:
        intr = CameraIntrinsics.from_mm(focal_mm, pitch_mm, native, working)
        cases.append({
            "focal_mm": focal_mm, "pitch_mm": pitch_mm,
            "native_w": native[0], "native_h": native[1],
            "working_w": working[0], "working_h": working[1],
            "expect_focal_px": intr.focal_px,
            "expect_cx": intr.principal_point[0], "expect_cy": intr.principal_point[1],
        })
    return cases


def area_px_to_cm2() -> list[dict]:
    cases = []
    # (mm_per_px, area_px, tilt_deg). tilt None/0 is top-down (factor 1.0); oblique
    # angles exercise the 1/cos(θ) foreshortening; 80° trips the 65° clamp.
    specs = [
        (0.5, 50_000.0, None),
        (1.2, 12_345.0, 0.0),
        (0.83, 0.0, 30.0),
        (2.0, 90_000.0, 45.0),
        (0.5, 50_000.0, 60.0),
        (0.5, 50_000.0, 80.0),  # clamped at MAX_FORESHORTENING_TILT_DEG
    ]
    for mm_per_px, area_px, tilt_deg in specs:
        se = ScaleEstimate(mm_per_px=mm_per_px, source=_first_source(), confidence=0.9, tilt_deg=tilt_deg)
        cases.append({
            "mm_per_px": mm_per_px, "area_px": area_px, "tilt_deg": tilt_deg,
            "expect_mm2_per_px2": se.mm2_per_px2,
            "expect_cm2": se.area_px_to_cm2(area_px),
        })
    return cases


def volume() -> list[dict]:
    cases = []
    for area_cm2, shape, height in [
        (100.0, "prism", 3.0), (100.0, "cylinder", 5.0),
        (100.0, "paraboloid", 4.0), (100.0, "flat", 99.0),  # height ignored for flat
        (250.0, "unknown_shape", 2.5),
    ]:
        v = VOL.estimate_volume(area_cm2, shape, height)
        cases.append({
            "area_cm2": area_cm2, "shape": shape, "height_cm": height,
            "expect_volume_cm3": v.volume_cm3, "expect_shape": v.shape,
            "expect_effective_height_cm": v.height_cm,
        })
    return cases


def density_lookup() -> list[dict]:
    names = ["baklava", "apple_pie", "breakfast_burrito", "doner", "egg", "banana", "unknown_food"]
    cases = []
    for n in names:
        rho = density_for(n)
        cases.append({"class_name": n, "expect_density": rho})  # None -> JSON null
    return cases


def height_lookup() -> list[dict]:
    names = ["baklava", "apple_pie", "beef_carpaccio", "doner", "egg", "banana", "unknown_food"]
    return [{"class_name": n, "expect_height_cm": height_for(n)} for n in names]


def density_resolve() -> list[dict]:
    cases = []
    specs = [
        ("baklava", 1.2),       # DB density present -> DATABASE
        ("apple_pie", None),    # null DB -> category (pastry 0.60)
        ("unknown_food", None),  # null DB, no category -> water 1.0
    ]
    for name, db_density in specs:
        rho, src = DENS.resolve(_food(class_name=name, density=db_density))
        cases.append({
            "class_name": name, "db_density": db_density,
            "expect_density": rho, "expect_source": src.value,
        })
    return cases


def nutrition() -> list[dict]:
    cases = []
    for kcal, p, f, c, grams in [(520.0, 8.0, 30.0, 55.0, 180.0), (52.0, 0.3, 0.2, 14.0, 120.0)]:
        food = _food(calories_per_100g=kcal, protein_per_100g=p, fat_per_100g=f, carbs_per_100g=c)
        n = NutritionResult.from_food(food, grams)
        cases.append({
            "kcal_100": kcal, "protein_100": p, "fat_100": f, "carbs_100": c, "grams": grams,
            "expect_calories": n.calories, "expect_protein": n.protein,
            "expect_fat": n.fat, "expect_carbs": n.carbs,
        })
    return cases


def pixel_ratio() -> list[dict]:
    pr = PixelRatioStrategy()
    cases = []
    specs = [
        (60_000.0, 50_000.0, 150.0),  # normal
        (60_000.0, None, 150.0),       # null ref_area -> config.default_ref_area
        (60_000.0, 0.0, 150.0),        # zero ref_area -> default
    ]
    for area_sq, ref_area, portion in specs:
        food = _food(ref_area=ref_area, portion_g=portion)
        det = _detection(area_sq=area_sq, area_frame=area_sq)
        m = pr.estimate(det, food, _ctx(None))
        cases.append({
            "mask_area_px": area_sq, "ref_area": ref_area, "portion_g": portion,
            "default_ref_area": CFG.default_ref_area,
            "expect_grams": m.grams, "expect_method": m.method,
        })
    return cases


def volumetric() -> list[dict]:
    vs = VolumetricStrategy()
    cases = []
    specs = [
        # (mm_per_px, area_frame, area_sq, class_name, db_density, shape)
        (0.5, 90_000.0, 60_000.0, "baklava", 1.2, "prism"),
        (1.0, 40_000.0, 30_000.0, "apple_pie", None, "prism"),
        (0.83, 0.0, 50_000.0, "egg", 0.9, "paraboloid"),  # area_frame=0 -> falls back to area_sq
    ]
    for mm_per_px, area_frame, area_sq, name, db_density, shape in specs:
        se = ScaleEstimate(mm_per_px=mm_per_px, source=_first_source(), confidence=0.9)
        food = _food(class_name=name, density=db_density, geometric_shape=shape)
        det = _detection(area_sq=area_sq, area_frame=area_frame)
        m = vs.estimate(det, food, _ctx(se))
        cases.append({
            "mm_per_px": mm_per_px, "mask_area_px_frame": area_frame, "mask_area_px": area_sq,
            "class_name": name, "db_density": db_density, "geometric_shape": shape,
            "expect_grams": m.grams, "expect_volume_cm3": m.volume_cm3,
            "expect_density_used": m.density_used,
        })
    return cases


def auto_strategy() -> list[dict]:
    auto = AutoStrategy()
    cases = []
    threshold = CFG.calibration_confidence_threshold
    # (confidence, tilt_deg). The tilt gate sends a confident-but-oblique frame to
    # the uncalibrated fallback; a moderate tilt stays volumetric (foreshortening-corrected).
    specs = [
        (0.90, None),   # confident, top-down -> volumetric
        (0.30, None),   # not confident -> pixel_ratio:uncalibrated
        (None, None),   # no scale -> pixel_ratio:uncalibrated
        (0.90, 70.0),   # confident but too oblique -> gated to pixel_ratio:uncalibrated
        (0.90, 40.0),   # confident, moderate tilt -> volumetric
    ]
    for conf, tilt_deg in specs:
        se = None if conf is None else ScaleEstimate(
            mm_per_px=0.5, source=_first_source(), confidence=conf, tilt_deg=tilt_deg
        )
        food = _food(class_name="baklava", density=1.2, ref_area=50_000.0, portion_g=150.0)
        det = _detection(area_sq=60_000.0, area_frame=90_000.0)
        m = auto.estimate(det, food, _ctx(se))
        cases.append({
            "scale_confidence": conf, "tilt_deg": tilt_deg, "threshold": threshold,
            "expect_grams": m.grams, "expect_method": m.method,
        })
    return cases


def calibration_levels() -> list[dict]:
    """Each pure (device-data) calibration level, exercised in isolation."""
    cases = []
    working = (640, 640)
    intr = CameraIntrinsics.from_mm(CFG.focal_length_mm, CFG.pixel_pitch_mm, CFG.native_resolution, working)

    # L1 DEPTH_INTRINSICS: mm_per_px = depth_mm / focal_px, conf 0.90
    l1 = IntrinsicDepthCalibration()
    ctx = FrameContext.from_device(DUMMY_FRAME, intrinsics=intr, depth_mm=300.0, tilt_deg=5.0)
    est = l1.estimate_scale(ctx, [])
    cases.append({
        "level": "depth_intrinsics", "focal_px": intr.focal_px, "depth_mm": 300.0,
        "expect_mm_per_px": est.mm_per_px, "expect_source": est.source.value,
        "expect_confidence": est.confidence,
    })

    # L2 INTRINSICS_PLANE: mm_per_px = distance / focal_px (distance = depth_mm or default), conf 0.70
    l2 = IntrinsicPlaneCalibration(CFG)
    ctx_d = FrameContext.from_device(DUMMY_FRAME, intrinsics=intr, depth_mm=420.0, tilt_deg=10.0)
    est_d = l2.estimate_scale(ctx_d, [])
    cases.append({
        "level": "intrinsics_plane", "focal_px": intr.focal_px, "depth_mm": 420.0,
        "default_distance_mm": CFG.default_distance_mm,
        "expect_mm_per_px": est_d.mm_per_px, "expect_source": est_d.source.value,
        "expect_confidence": est_d.confidence,
    })
    ctx_nd = FrameContext.from_device(DUMMY_FRAME, intrinsics=intr, tilt_deg=10.0)  # no depth -> default
    est_nd = l2.estimate_scale(ctx_nd, [])
    cases.append({
        "level": "intrinsics_plane", "focal_px": intr.focal_px, "depth_mm": None,
        "default_distance_mm": CFG.default_distance_mm,
        "expect_mm_per_px": est_nd.mm_per_px, "expect_source": est_nd.source.value,
        "expect_confidence": est_nd.confidence,
    })

    # L4 STATIC_DEFAULT: mm_per_px = default_distance / focal_px, conf 0.20
    l4 = StaticCalibration.from_config(CFG)
    est4 = l4.estimate_scale(FrameContext.simulated_topdown(DUMMY_FRAME), [])
    cases.append({
        "level": "static_default", "focal_px": intr.focal_px,
        "default_distance_mm": CFG.default_distance_mm,
        "expect_mm_per_px": est4.mm_per_px, "expect_source": est4.source.value,
        "expect_confidence": est4.confidence,
    })
    return cases


def assumed_plate() -> list[dict]:
    """The pure-arithmetic 'assume a standard plate' fallback of Level 3.

    mm_per_px = (plate_diameter_cm * 10) / (assumed_fraction * max(w, h)), conf 0.30.
    """
    det_svc = NaturalAnchorCalibration.__new__(NaturalAnchorCalibration)
    # Compute directly to avoid the CV detector; mirror the code path exactly.
    w, h = 640, 480
    pixel_size = CFG.assumed_plate_frame_fraction * max(w, h)
    mm_per_px = (CFG.default_plate_diameter_cm * 10.0) / pixel_size
    return [{
        "frame_w": w, "frame_h": h,
        "plate_diameter_cm": CFG.default_plate_diameter_cm,
        "assumed_fraction": CFG.assumed_plate_frame_fraction,
        "expect_mm_per_px": mm_per_px, "expect_confidence": 0.30,
        "expect_source": "natural_anchor",
    }]


def resolver_selection() -> list[dict]:
    """Resolver returns the most confident estimate among available levels."""
    working = (640, 640)
    intr = CameraIntrinsics.from_mm(CFG.focal_length_mm, CFG.pixel_pitch_mm, CFG.native_resolution, working)
    resolver = CalibrationResolver([
        IntrinsicDepthCalibration(),
        IntrinsicPlaneCalibration(CFG),
        StaticCalibration.from_config(CFG),
    ])
    cases = []
    # With depth + intrinsics -> L1 (0.90) wins.
    ctx_full = FrameContext.from_device(DUMMY_FRAME, intrinsics=intr, depth_mm=300.0, tilt_deg=5.0)
    best = resolver.estimate_scale(ctx_full, [])
    cases.append({"scenario": "depth+intrinsics", "expect_source": best.source.value,
                  "expect_confidence": best.confidence, "expect_mm_per_px": best.mm_per_px})
    # Intrinsics + tilt, no depth -> L2 (0.70) wins over L4.
    ctx_tilt = FrameContext.from_device(DUMMY_FRAME, intrinsics=intr, tilt_deg=8.0)
    best2 = resolver.estimate_scale(ctx_tilt, [])
    cases.append({"scenario": "intrinsics+tilt", "expect_source": best2.source.value,
                  "expect_confidence": best2.confidence, "expect_mm_per_px": best2.mm_per_px})
    # Bare frame -> only L4 (0.20).
    ctx_bare = FrameContext.simulated_topdown(DUMMY_FRAME)
    best3 = resolver.estimate_scale(ctx_bare, [])
    cases.append({"scenario": "bare", "expect_source": best3.source.value,
                  "expect_confidence": best3.confidence, "expect_mm_per_px": best3.mm_per_px})
    return cases


def _first_source():
    from lokma.core.models import CalibrationSource
    return CalibrationSource.DEPTH_INTRINSICS


# --- main -------------------------------------------------------------------


def main() -> None:
    payload = {
        "_meta": {
            "generated_by": "scripts/dump_golden_vectors.py",
            "purpose": "Swift LokmaCore parity vectors — assert Swift port == Python reference",
            "config": {
                "focal_length_mm": CFG.focal_length_mm,
                "pixel_pitch_mm": CFG.pixel_pitch_mm,
                "native_resolution": list(CFG.native_resolution),
                "default_distance_mm": CFG.default_distance_mm,
                "default_ref_area": CFG.default_ref_area,
                "calibration_confidence_threshold": CFG.calibration_confidence_threshold,
                "default_plate_diameter_cm": CFG.default_plate_diameter_cm,
                "assumed_plate_frame_fraction": CFG.assumed_plate_frame_fraction,
                "flat_layer_cm": VOL.FLAT_LAYER_CM,
                "water_density": DENS.water_density,
                "default_height_cm": height_for("___definitely_unknown___"),
            },
        },
        "intrinsics_from_mm": intrinsics_from_mm(),
        "area_px_to_cm2": area_px_to_cm2(),
        "volume": volume(),
        "density_lookup": density_lookup(),
        "height_lookup": height_lookup(),
        "density_resolve": density_resolve(),
        "nutrition": nutrition(),
        "pixel_ratio": pixel_ratio(),
        "volumetric": volumetric(),
        "auto_strategy": auto_strategy(),
        "calibration_levels": calibration_levels(),
        "assumed_plate": assumed_plate(),
        "resolver_selection": resolver_selection(),
    }

    out = (
        Path(__file__).resolve().parents[1]
        / "ios" / "LokmaCore" / "Tests" / "LokmaCoreTests" / "Fixtures" / "golden_vectors.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    n = sum(len(v) for k, v in payload.items() if k != "_meta")
    print(f"Wrote {n} golden vectors across {len(payload) - 1} groups -> {out}")


if __name__ == "__main__":
    main()
