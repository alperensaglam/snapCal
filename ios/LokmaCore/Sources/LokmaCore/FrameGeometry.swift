// FrameGeometry — the canonical coordinate map for the iOS LiDAR capture.
//
// Pure value type (no ARKit/CoreVideo) so it lives in LokmaCore and is unit-tested
// here. It ties together the three grids the Tier 2 depth path must cross:
//
//   capturedImage (imageW×imageH, landscape sensor space)
//     └─ center-cropped to S = min(imageW, imageH)  →  the S×S frame the model sees
//          └─ scaled to the 640 / 160 model+mask grids
//   sceneDepth.depthMap (depthW×depthH) — SAME field of view as capturedImage,
//          just lower resolution (so its intrinsics are the camera intrinsics scaled
//          by depthW/imageW, depthH/imageH).
//
// The mask lives in the S×S *center crop*; the depth map covers the *full* frame,
// so aligning them needs the crop offset `origin = ((imageW−S)/2, (imageH−S)/2)`.
// This replaces the previous hardcoded `(320,320)` principal point with the real,
// depth-space principal point.
import Foundation

public struct FrameGeometry: Sendable, Equatable {
    public let imageWidth: Int
    public let imageHeight: Int
    public let depthWidth: Int
    public let depthHeight: Int
    /// Camera intrinsics in CAPTURED-image pixels (fx, fy, cx, cy).
    public let fx: Double
    public let fy: Double
    public let cx: Double
    public let cy: Double

    public init(imageWidth: Int, imageHeight: Int, depthWidth: Int, depthHeight: Int,
                fx: Double, fy: Double, cx: Double, cy: Double) {
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.depthWidth = depthWidth
        self.depthHeight = depthHeight
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    }

    /// Side of the centered square crop the model sees.
    public var cropSide: Int { min(imageWidth, imageHeight) }
    public var cropOriginX: Double { Double(imageWidth - cropSide) / 2.0 }
    public var cropOriginY: Double { Double(imageHeight - cropSide) / 2.0 }

    /// Captured→depth resolution ratios (equal for a shared 4:3 FOV).
    public var scaleX: Double { Double(depthWidth) / Double(max(1, imageWidth)) }
    public var scaleY: Double { Double(depthHeight) / Double(max(1, imageHeight)) }

    /// Intrinsics expressed in depth-map pixels — what `DepthVolumeEngine` consumes.
    public var depthIntrinsics: (fx: Double, fy: Double, cx: Double, cy: Double) {
        (fx * scaleX, fy * scaleY, cx * scaleX, cy * scaleY)
    }

    /// Map a depth pixel (center) to S×S crop-frame coordinates, or nil when it falls
    /// outside the center crop (a region the model never saw). Used to resample the
    /// per-instance mask onto the depth grid.
    public func depthPixelToCropFrame(_ du: Int, _ dv: Int) -> (x: Double, y: Double)? {
        guard scaleX > 0, scaleY > 0 else { return nil }
        let capturedX = (Double(du) + 0.5) / scaleX
        let capturedY = (Double(dv) + 0.5) / scaleY
        let x = capturedX - cropOriginX
        let y = capturedY - cropOriginY
        let s = Double(cropSide)
        guard x >= 0, x < s, y >= 0, y < s else { return nil }
        return (x, y)
    }
}
