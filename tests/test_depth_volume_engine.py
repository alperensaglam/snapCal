"""DepthVolumeEngineService — Tier 2 depth-integrated volume above a fitted plane.

Pure-numpy tests (no OpenCV/YOLO). These lock the Python reference independently
of the Swift parity contract; `ParityTests.testDepthVolume` asserts the Swift port
reproduces the same numbers byte-for-byte.
"""

import numpy as np

from lokma.geometry.depth_volume_engine import DepthVolumeEngineService

W, H = 24, 18
FX = FY = 30.0
CX, CY = 12.0, 9.0
Z0, H_BOX = 300.0, 20.0
BOX_U = range(8, 16)
BOX_V = range(6, 12)


def _base_flat():
    depth = np.full((H, W), Z0, dtype=np.float64)
    mask = np.zeros((H, W), dtype=np.float64)
    for v in BOX_V:
        for u in BOX_U:
            depth[v, u] = Z0 - H_BOX
            mask[v, u] = 1.0
    return depth, mask


def _integrate(depth, mask):
    return DepthVolumeEngineService().integrate(
        depth.flatten().tolist(), mask.flatten().tolist(), W, H, FX, FY, CX, CY
    )


def test_flat_topdown_equals_height_times_area():
    depth, mask = _base_flat()
    res = _integrate(depth, mask)
    assert res is not None
    n_box = sum(1 for _ in BOX_V for _ in BOX_U)
    footprint_mm2 = n_box * (Z0 - H_BOX) ** 2 / (FX * FY)
    expected_cm3 = H_BOX * footprint_mm2 / 1000.0
    assert abs(res.volume_cm3 - expected_cm3) < 1e-9
    assert res.coverage == 1.0
    assert res.method == "depth_plane"


def test_tilted_plane_within_one_percent_of_flat():
    flat, mask = _base_flat()
    flat_v = _integrate(flat, mask).volume_cm3
    tilted, mask = _base_flat()
    a = 0.2
    for v in range(H):
        for u in range(W):
            zp = Z0 / (1.0 - a * (u - CX) / FX)
            tilted[v, u] = zp - H_BOX if mask[v, u] >= 0.5 else zp
    tilted_v = _integrate(tilted, mask).volume_cm3
    assert abs(tilted_v - flat_v) / flat_v < 0.01  # foreshortening cancels


def test_ring_outliers_are_trimmed():
    flat, mask = _base_flat()
    flat_v = _integrate(flat, mask).volume_cm3
    noisy, mask = _base_flat()
    for (vv, uu) in [(2, 2), (3, 20), (15, 4), (16, 19)]:
        noisy[vv, uu] = Z0 + 120.0
    noisy_v = _integrate(noisy, mask).volume_cm3
    assert abs(noisy_v - flat_v) < 1e-9  # MAD-trim recovers the flat plane


def test_low_coverage_returns_none():
    depth, mask = _base_flat()
    cnt = 0
    for v in BOX_V:
        for u in BOX_U:
            if cnt % 5 != 0:
                depth[v, u] = 0.0  # invalid sentinel
            cnt += 1
    assert _integrate(depth, mask) is None


def test_auto_strategy_prefers_depth_volume():
    from lokma.config import AppConfig
    from lokma.core.models import DepthSample, Detection, FoodRecord
    from lokma.density.density_service import DensityService
    from lokma.geometry.strategies import AutoStrategy, MassContext
    from lokma.geometry.volume_engine import VolumeEngineService

    depth, mask = _base_flat()
    sample = DepthSample(
        depth_mm=depth.flatten().tolist(), mask=mask.flatten().tolist(),
        width=W, height=H, fx=FX, fy=FY, cx=CX, cy=CY,
    )
    food = FoodRecord(
        class_name="baklava", source="curated", usda_desc=None,
        calories_per_100g=520.0, protein_per_100g=8.0, fat_per_100g=30.0,
        carbs_per_100g=55.0, portion_g=150.0, ref_area=50_000.0,
        density=1.2, geometric_shape="prism",
    )
    det = Detection(
        class_id=0, class_name="baklava", confidence=0.9,
        mask=np.zeros((4, 4), dtype=np.float32), bbox=(0.0, 0.0, 1.0, 1.0),
        mask_area_px=60_000.0, mask_area_px_frame=90_000.0, frame_size=(640, 480),
        depth_sample=sample,
    )
    # No ScaleEstimate at all — the depth path is self-calibrated and still wins.
    ctx = MassContext(scale=None, volume_engine=VolumeEngineService(),
                      density_service=DensityService(), config=AppConfig())
    m = AutoStrategy().estimate(det, food, ctx)
    flat_v = _integrate(*_base_flat()).volume_cm3
    assert m.method == "volumetric_depth:database"
    assert abs(m.volume_cm3 - flat_v) < 1e-9
    assert abs(m.grams - flat_v * 1.2) < 1e-6
