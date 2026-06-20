// DepthVolumeEngine — Tier 2 LiDAR depth-integrated volume above a fitted plane.
//
// The Swift mirror of `lokma/geometry/depth_volume_engine.py`. Pure, deterministic,
// and arithmetic-identical (same ring band, same explicit 3×3 Cramer solve, same
// median/MAD trim, same integration order) so the golden-vector parity holds.
//
// Math: fit the support plane z = a·x + b·y + c from the depth ring around the
// food, then
//     V_mm³ = (1/(fx·fy)) · Σ_food max(0, z_plane(xᵢ,yᵢ) − zᵢ) · zᵢ²
// which reduces to height × footprint-area for a flat top-down plate and stays
// correct under tilt. Operates in depth-map pixel space (never upsamples depth).
import Foundation

public struct DepthVolumeParams: Sendable {
    public var minCoverage: Double = 0.5
    public var bboxMarginFrac: Double = 0.25
    public var madK: Double = 3.0
    public var minRingPoints: Int = 30
    public var heightClampMm: Double = 0.0
    public init() {}
}

public struct DepthVolumeResult: Sendable, Equatable {
    public let volumeCm3: Double
    public let coverage: Double
    public let planeResidualMm: Double
    public let method: String

    public init(volumeCm3: Double, coverage: Double, planeResidualMm: Double, method: String = "depth_plane") {
        self.volumeCm3 = volumeCm3
        self.coverage = coverage
        self.planeResidualMm = planeResidualMm
        self.method = method
    }
}

public struct DepthVolumeEngine: Sendable {
    public init() {}

    private static let detEps = 1e-9
    private static let madEps = 1e-9

    /// Integrate food volume (cm³) above the fitted support plane, or nil when the
    /// depth is too sparse / the ring too small / the plane singular (caller falls
    /// back to the scalar volumetric path). `depthMm`/`mask` are row-major, length
    /// `width*height`; depth ≤ 0 (or non-finite) marks an invalid pixel.
    public func integrate(
        depthMm: [Double], mask: [Double], width: Int, height: Int,
        fx: Double, fy: Double, cx: Double, cy: Double,
        params: DepthVolumeParams = DepthVolumeParams()
    ) -> DepthVolumeResult? {
        let n = width * height
        guard n > 0, depthMm.count == n, mask.count == n, fx != 0, fy != 0 else { return nil }

        // Food bounding box.
        var foodTotal = 0
        var minU = width, maxU = -1, minV = height, maxV = -1
        for v in 0..<height {
            for u in 0..<width where mask[v * width + u] >= 0.5 {
                foodTotal += 1
                if u < minU { minU = u }
                if u > maxU { maxU = u }
                if v < minV { minV = v }
                if v > maxV { maxV = v }
            }
        }
        guard foodTotal > 0 else { return nil }

        let mx = Int(params.bboxMarginFrac * Double(maxU - minU + 1) + 0.5)
        let my = Int(params.bboxMarginFrac * Double(maxV - minV + 1) + 0.5)
        let eMinU = max(0, minU - mx), eMaxU = min(width - 1, maxU + mx)
        let eMinV = max(0, minV - my), eMaxV = min(height - 1, maxV + my)

        // Ring = valid, non-food pixels inside the expanded bbox (row-major order).
        var rx: [Double] = [], ry: [Double] = [], rz: [Double] = []
        for v in eMinV...eMaxV {
            for u in eMinU...eMaxU {
                let i = v * width + u
                let z = depthMm[i]
                guard z.isFinite, z > 0, mask[i] < 0.5 else { continue }
                rx.append((Double(u) - cx) / fx * z)
                ry.append((Double(v) - cy) / fy * z)
                rz.append(z)
            }
        }
        guard rx.count >= params.minRingPoints else { return nil }

        guard var plane = Self.solvePlane(rx, ry, rz) else { return nil }

        // One MAD-trim of ring residuals, then refit.
        var resid = [Double](repeating: 0, count: rx.count)
        for j in 0..<rx.count { resid[j] = plane.a * rx[j] + plane.b * ry[j] + plane.c - rz[j] }
        let med = Self.median(resid)
        var absdev = [Double](repeating: 0, count: resid.count)
        for j in 0..<resid.count { absdev[j] = abs(resid[j] - med) }
        let mad = Self.median(absdev)
        if mad > Self.madEps {
            let cutoff = params.madK * mad
            var kx: [Double] = [], ky: [Double] = [], kz: [Double] = []
            for j in 0..<rx.count where absdev[j] <= cutoff {
                kx.append(rx[j]); ky.append(ry[j]); kz.append(rz[j])
            }
            if kx.count >= 3, let refit = Self.solvePlane(kx, ky, kz) {
                plane = refit; rx = kx; ry = ky; rz = kz
            }
        }

        var sse = 0.0
        for j in 0..<rx.count {
            let r = plane.a * rx[j] + plane.b * ry[j] + plane.c - rz[j]
            sse += r * r
        }
        let planeResidual = rx.isEmpty ? 0.0 : (sse / Double(rx.count)).squareRoot()

        // Coverage gate.
        var foodValid = 0
        for i in 0..<n where mask[i] >= 0.5 && depthMm[i].isFinite && depthMm[i] > 0 { foodValid += 1 }
        let coverage = Double(foodValid) / Double(foodTotal)
        guard coverage >= params.minCoverage else { return nil }

        // Integrate V = (1/(fx·fy)) Σ max(0, z_plane − z) · z².
        var sumTerm = 0.0
        for v in 0..<height {
            for u in 0..<width {
                let i = v * width + u
                guard mask[i] >= 0.5 else { continue }
                let z = depthMm[i]
                guard z.isFinite, z > 0 else { continue }
                let x = (Double(u) - cx) / fx * z
                let y = (Double(v) - cy) / fy * z
                let h = (plane.a * x + plane.b * y + plane.c) - z
                if h > params.heightClampMm { sumTerm += h * z * z }
            }
        }
        let volumeCm3 = (sumTerm / (fx * fy)) / 1000.0
        return DepthVolumeResult(volumeCm3: volumeCm3, coverage: coverage, planeResidualMm: planeResidual)
    }

    /// Convenience over a `DepthSample` (mirrors `integrate_sample` in Python).
    public func integrate(_ sample: DepthSample, params: DepthVolumeParams = DepthVolumeParams()) -> DepthVolumeResult? {
        integrate(depthMm: sample.depthMm, mask: sample.mask, width: sample.width, height: sample.height,
                  fx: sample.fx, fy: sample.fy, cx: sample.cx, cy: sample.cy, params: params)
    }

    // MARK: - deterministic helpers (mirror models in depth_volume_engine.py)

    static func median(_ xs: [Double]) -> Double {
        if xs.isEmpty { return 0.0 }
        let s = xs.sorted()
        let n = s.count, mid = n / 2
        return n % 2 == 1 ? s[mid] : 0.5 * (s[mid - 1] + s[mid])
    }

    private static func det3(_ m: [[Double]]) -> Double {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Least-squares z = a·x + b·y + c via 3×3 normal equations + Cramer's rule.
    static func solvePlane(_ xs: [Double], _ ys: [Double], _ zs: [Double]) -> (a: Double, b: Double, c: Double)? {
        var sxx = 0.0, syy = 0.0, sxy = 0.0, sx = 0.0, sy = 0.0, sxz = 0.0, syz = 0.0, sz = 0.0
        let n = xs.count
        for i in 0..<n {
            let x = xs[i], y = ys[i], z = zs[i]
            sxx += x * x; syy += y * y; sxy += x * y
            sx += x; sy += y; sxz += x * z; syz += y * z; sz += z
        }
        let A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, Double(n)]]
        let b = [sxz, syz, sz]
        let det = det3(A)
        if abs(det) < detEps { return nil }
        let aa = [[b[0], A[0][1], A[0][2]], [b[1], A[1][1], A[1][2]], [b[2], A[2][1], A[2][2]]]
        let ab = [[A[0][0], b[0], A[0][2]], [A[1][0], b[1], A[1][2]], [A[2][0], b[2], A[2][2]]]
        let ac = [[A[0][0], A[0][1], b[0]], [A[1][0], A[1][1], b[1]], [A[2][0], A[2][1], b[2]]]
        return (det3(aa) / det, det3(ab) / det, det3(ac) / det)
    }
}
