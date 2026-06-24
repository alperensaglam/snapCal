"""Tier 2: LiDAR depth-integrated food volume above a fitted support plane.

Pure-numpy and deterministic — the Swift `DepthVolumeEngine` mirrors this exact
arithmetic for golden-vector parity (no cv2/scipy). Operates in depth-map pixel
space; never upsamples depth.

Math (derived in the plan): with the support plane fit as ``z = a·x + b·y + c``
from the depth ring around the food, the perpendicular-height × plane-footprint
integral collapses (the √(a²+b²+1) foreshortening terms cancel) to

    V_mm³ = (1 / (fx·fy)) · Σ_food  max(0, z_plane(xᵢ,yᵢ) − zᵢ) · zᵢ²

which reduces to exactly height × footprint-area for a flat top-down plate and
stays correct under tilt. Back-projection uses ``x = (u−cx)/fx·z`` (camera mm,
perpendicular-Z convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DET_EPS = 1e-9
_MAD_EPS = 1e-9


@dataclass(frozen=True)
class DepthVolumeParams:
    """Gating + robustness knobs (mirrored verbatim in Swift)."""

    min_coverage: float = 0.5       # fraction of food pixels needing valid depth
    bbox_margin_frac: float = 0.25  # ring band width as a fraction of the food bbox
    mad_k: float = 3.0              # MAD multiplier for ring outlier rejection
    min_ring_points: int = 30       # below this the plane fit isn't trustworthy
    height_clamp_mm: float = 0.0    # ignore surface points at/below this height
    max_food_height_mm: float = 250.0  # reject if the median food height exceeds this
    #                                    (no near support plane → V would explode)


@dataclass(frozen=True)
class DepthVolumeResult:
    volume_cm3: float
    coverage: float
    plane_residual_mm: float
    method: str = "depth_plane"


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _det3(m) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _solve_plane(xs: list[float], ys: list[float], zs: list[float]):
    """Least-squares ``z = a·x + b·y + c`` via 3×3 normal equations + Cramer's rule.

    Sequential accumulation (not ``np.linalg.lstsq``) so Swift can reproduce the
    coefficients bit-for-bit. Returns ``(a, b, c)`` or ``None`` when singular.
    """
    sxx = syy = sxy = sx = sy = sxz = syz = sz = 0.0
    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        z = zs[i]
        sxx += x * x
        syy += y * y
        sxy += x * y
        sx += x
        sy += y
        sxz += x * z
        syz += y * z
        sz += z
    A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(n)]]
    b = [sxz, syz, sz]
    det = _det3(A)
    if abs(det) < _DET_EPS:
        return None
    aa = [[b[0], A[0][1], A[0][2]], [b[1], A[1][1], A[1][2]], [b[2], A[2][1], A[2][2]]]
    ab = [[A[0][0], b[0], A[0][2]], [A[1][0], b[1], A[1][2]], [A[2][0], b[2], A[2][2]]]
    ac = [[A[0][0], A[0][1], b[0]], [A[1][0], A[1][1], b[1]], [A[2][0], A[2][1], b[2]]]
    return (_det3(aa) / det, _det3(ab) / det, _det3(ac) / det)


class DepthVolumeEngineService:
    """Integrate food volume above a deterministically fitted support plane."""

    def integrate(
        self,
        depth_mm,
        mask,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        params: DepthVolumeParams | None = None,
    ) -> DepthVolumeResult | None:
        p = params or DepthVolumeParams()
        if width <= 0 or height <= 0 or fx == 0.0 or fy == 0.0:
            return None
        depth = np.asarray(depth_mm, dtype=np.float64).reshape(height, width)
        m = np.asarray(mask, dtype=np.float64).reshape(height, width)

        valid = np.isfinite(depth) & (depth > 0.0)
        food = m >= 0.5
        food_total = int(food.sum())
        if food_total == 0:
            return None

        # Food bounding box, expanded by a margin to form the ring band.
        ys, xs = np.nonzero(food)
        min_u, max_u = int(xs.min()), int(xs.max())
        min_v, max_v = int(ys.min()), int(ys.max())
        mx = int(p.bbox_margin_frac * (max_u - min_u + 1) + 0.5)
        my = int(p.bbox_margin_frac * (max_v - min_v + 1) + 0.5)
        e_min_u, e_max_u = max(0, min_u - mx), min(width - 1, max_u + mx)
        e_min_v, e_max_v = max(0, min_v - my), min(height - 1, max_v + my)

        # Ring = valid, non-food pixels inside the expanded bbox.
        ring = np.zeros((height, width), dtype=bool)
        ring[e_min_v:e_max_v + 1, e_min_u:e_max_u + 1] = True
        ring &= valid & (~food)
        rv, ru = np.nonzero(ring)
        if rv.size < p.min_ring_points:
            return None

        rz = depth[rv, ru]
        rx = (ru.astype(np.float64) - cx) / fx * rz
        ry = (rv.astype(np.float64) - cy) / fy * rz

        plane = _solve_plane(rx.tolist(), ry.tolist(), rz.tolist())
        if plane is None:
            return None
        a, b, c = plane

        # One MAD-trim of ring residuals, then refit.
        resid = a * rx + b * ry + c - rz
        med = _median(resid.tolist())
        absdev = np.abs(resid - med)
        mad = _median(absdev.tolist())
        if mad > _MAD_EPS:
            keep = absdev <= p.mad_k * mad
            if int(keep.sum()) >= 3:
                kx, ky, kz = rx[keep], ry[keep], rz[keep]
                refit = _solve_plane(kx.tolist(), ky.tolist(), kz.tolist())
                if refit is not None:
                    a, b, c = refit
                    rx, ry, rz = kx, ky, kz

        resid_final = a * rx + b * ry + c - rz
        plane_residual = float(np.sqrt(np.mean(resid_final * resid_final))) if rz.size else 0.0

        # Coverage gate.
        food_valid = food & valid
        coverage = int(food_valid.sum()) / food_total
        if coverage < p.min_coverage:
            return None

        # Integrate V = (1/(fx·fy)) Σ max(0, z_plane − z) · z².
        fv, fu = np.nonzero(food_valid)
        fz = depth[fv, fu]
        fx_world = (fu.astype(np.float64) - cx) / fx * fz
        fy_world = (fv.astype(np.float64) - cy) / fy * fz
        z_plane = a * fx_world + b * fy_world + c
        h_raw = z_plane - fz
        # Plausibility: real food rises modestly above its plate. With no near support
        # plane (object over a far wall) the median height is huge → reject so V can't
        # explode into impossible mass/calories.
        heights = np.maximum(0.0, h_raw)
        if heights.size == 0 or float(np.median(heights)) > p.max_food_height_mm:
            return None
        h = np.where(h_raw > p.height_clamp_mm, h_raw, 0.0)
        sum_term = float(np.sum(h * fz * fz))
        volume_cm3 = (sum_term / (fx * fy)) / 1000.0
        return DepthVolumeResult(
            volume_cm3=volume_cm3, coverage=coverage, plane_residual_mm=plane_residual
        )

    def integrate_sample(self, sample, params: DepthVolumeParams | None = None) -> DepthVolumeResult | None:
        """Convenience over a :class:`DepthSample` (mirrors the Swift overload)."""
        return self.integrate(
            sample.depth_mm, sample.mask, sample.width, sample.height,
            sample.fx, sample.fy, sample.cx, sample.cy, params,
        )
