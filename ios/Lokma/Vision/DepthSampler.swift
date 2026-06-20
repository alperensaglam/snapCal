// DepthSampler — extracts a per-detection `DepthSample` from ARKit LiDAR buffers.
//
// Builds one shared `DepthField` (the frame's depth map → metric mm, low-confidence
// zeroed) per frame, then resamples each instance mask onto the depth grid via the
// pure `FrameGeometry`. Everything here is defensive: any inconsistency yields nil,
// so the estimator transparently falls back to the Tier-1 scalar path.
import ARKit
import LokmaCore

/// A frame's depth map flattened to metric mm (invalid/low-confidence → 0).
public struct DepthField: Sendable {
    public let mm: [Double]
    public let width: Int
    public let height: Int
}

/// Everything the Tier 2 path needs from one ARFrame: the depth field + the geometry
/// tying the depth grid to the model's crop. Delivered alongside the FrameContext.
public struct DepthCapture: Sendable {
    public let field: DepthField
    public let geometry: FrameGeometry
    /// Camera→world pose (ARKit), for unprojecting a detection's depth centroid to a
    /// stable world position the InstanceTracker keys on.
    public let cameraTransform: simd_float4x4
}

public enum DepthSampler {
    /// Flatten `sceneDepth` to metric mm, zeroing non-finite, ≤0, and low-confidence px.
    public static func depthField(from depth: ARDepthData?) -> DepthField? {
        guard let depth else { return nil }
        let map = depth.depthMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map)
        let h = CVPixelBufferGetHeight(map)
        let stride = CVPixelBufferGetBytesPerRow(map) / MemoryLayout<Float32>.size
        let ptr = base.assumingMemoryBound(to: Float32.self)

        let confMap = depth.confidenceMap
        if let confMap { CVPixelBufferLockBaseAddress(confMap, .readOnly) }
        defer { if let confMap { CVPixelBufferUnlockBaseAddress(confMap, .readOnly) } }
        let confPtr = confMap.flatMap { CVPixelBufferGetBaseAddress($0)?.assumingMemoryBound(to: UInt8.self) }
        let confStride = confMap.map { CVPixelBufferGetBytesPerRow($0) / MemoryLayout<UInt8>.size } ?? 0
        let lowConf = UInt8(ARConfidenceLevel.low.rawValue)

        var mm = [Double](repeating: 0, count: w * h)
        for v in 0..<h {
            for u in 0..<w {
                let z = ptr[v * stride + u]
                guard z.isFinite, z > 0 else { continue }
                if let confPtr, confPtr[v * confStride + u] == lowConf { continue }
                mm[v * w + u] = Double(z) * 1000.0   // metres → mm
            }
        }
        return DepthField(mm: mm, width: w, height: h)
    }

    /// Resample one instance mask (nearest) onto the depth grid and pair it with the
    /// shared depth field + depth-space intrinsics. nil on any inconsistency.
    public static func sample(detection raw: RawDetection, capture: DepthCapture) -> DepthSample? {
        let field = capture.field
        let geo = capture.geometry
        guard field.width == geo.depthWidth, field.height == geo.depthHeight,
              raw.maskWidth > 0, raw.maskHeight > 0, geo.cropSide > 0 else { return nil }
        let dw = field.width, dh = field.height
        let s = Double(geo.cropSide)
        let mwScale = Double(raw.maskWidth) / s
        let mhScale = Double(raw.maskHeight) / s

        var mask = [Double](repeating: 0, count: dw * dh)
        for dv in 0..<dh {
            for du in 0..<dw {
                guard let f = geo.depthPixelToCropFrame(du, dv) else { continue }
                let mu = Int(f.x * mwScale), mv = Int(f.y * mhScale)
                guard mu >= 0, mu < raw.maskWidth, mv >= 0, mv < raw.maskHeight else { continue }
                mask[dv * dw + du] = Double(raw.mask[mv * raw.maskWidth + mu])
            }
        }
        let i = geo.depthIntrinsics
        return DepthSample(depthMm: field.mm, mask: mask, width: dw, height: dh,
                           fx: i.fx, fy: i.fy, cx: i.cx, cy: i.cy)
    }
}
