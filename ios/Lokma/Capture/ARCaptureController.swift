// ARCaptureController — the Swift analogue of `FrameContext.from_device`.
//
// Runs an ARKit world-tracking session (with LiDAR sceneDepth + horizontal plane
// detection when available) and, per processed frame, reduces the `ARFrame` to a
// `LokmaCore.FrameContext`. The calibration resolver then inherits the
// L1(depth) -> L2(plane) -> L3(anchor) -> L4(static) degradation for free.
//
// Field mapping (see Part 3 of the Phase 4 blueprint):
//   intrinsics.focalPx <- camera.intrinsics.fx scaled to the 640 working grid
//   depthMm            <- sceneDepth.depthMap sampled at the frame centre (m->mm)
//   tiltDeg            <- angle between the camera's view axis and world-down
//   frameSize          <- camera.imageResolution (drives the assumed-plate fallback)
import ARKit
import Foundation
import LokmaCore
import simd

public protocol ARCaptureDelegate: AnyObject {
    /// Called on a background-friendly cadence with one capture + its context.
    func capture(_ controller: ARCaptureController, didProduce pixelBuffer: CVPixelBuffer, context: FrameContext)
}

public final class ARCaptureController: NSObject, ARSessionDelegate {
    public let session = ARSession()
    public weak var delegate: ARCaptureDelegate?

    private let config: AppConfig
    private let workingWidth: Int
    private var lastProcessed: TimeInterval = 0
    /// Cap inference cadence; ARKit delivers ~60 fps but we only need a few.
    public var minInterval: TimeInterval = 1.0 / 4.0

    public init(config: AppConfig = AppConfig()) {
        self.config = config
        self.workingWidth = config.workingResolution.0
        super.init()
        session.delegate = self
    }

    public func start() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal]            // table plane -> Level 2
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)     // LiDAR -> Level 1
        }
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    public func pause() { session.pause() }

    // MARK: - ARSessionDelegate

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard frame.timestamp - lastProcessed >= minInterval else { return }
        lastProcessed = frame.timestamp
        let context = makeContext(from: frame)
        delegate?.capture(self, didProduce: frame.capturedImage, context: context)
    }

    // MARK: - ARFrame -> FrameContext

    func makeContext(from frame: ARFrame) -> FrameContext {
        let camera = frame.camera
        let imageRes = camera.imageResolution
        let imageW = Int(imageRes.width.rounded())
        let imageH = Int(imageRes.height.rounded())

        // Intrinsics: fx is in pixels of the captured image; rescale to the
        // 640 working grid the masks (and ref_area) are measured on.
        let fxCaptured = Double(camera.intrinsics.columns.0.x)
        let scaleToWorking = Double(workingWidth) / max(1.0, Double(imageW))
        let focalPxWorking = fxCaptured * scaleToWorking
        let intrinsics = CameraIntrinsics(
            focalPx: focalPxWorking,
            imageSize: config.workingResolution,
            principalPoint: (Double(workingWidth) / 2.0, Double(workingWidth) / 2.0)
        )

        let depthMm = representativeDepthMm(frame.sceneDepth ?? frame.smoothedSceneDepth)
        let tiltDeg = tiltDegFromTopDown(camera.transform)

        return FrameContext(
            intrinsics: intrinsics,
            tiltDeg: tiltDeg,
            depthMm: depthMm,
            frameSize: (imageW, imageH)
        )
    }

    /// Angle (deg) between the camera's viewing axis and world-down. 0° = top-down.
    /// ARKit's world is gravity-aligned (+Y up); the camera looks along its -Z.
    private func tiltDegFromTopDown(_ transform: simd_float4x4) -> Double {
        let viewDir = simd_normalize(-simd_float3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z))
        let down = simd_float3(0, -1, 0)
        let cosA = max(-1.0, min(1.0, Double(simd_dot(viewDir, down))))
        return acos(cosA) * 180.0 / .pi
    }

    /// Median of valid metric depths near the frame centre, in mm; nil if no depth.
    private func representativeDepthMm(_ depth: ARDepthData?) -> Double? {
        guard let depth else { return nil }
        let map = depth.depthMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }

        let w = CVPixelBufferGetWidth(map)
        let h = CVPixelBufferGetHeight(map)
        let rowBytes = CVPixelBufferGetBytesPerRow(map)
        let ptr = base.assumingMemoryBound(to: Float32.self)
        let stride = rowBytes / MemoryLayout<Float32>.size

        // Sample a centred window (~40% of each side).
        var samples: [Float] = []
        let x0 = Int(Double(w) * 0.3), x1 = Int(Double(w) * 0.7)
        let y0 = Int(Double(h) * 0.3), y1 = Int(Double(h) * 0.7)
        for y in y0..<y1 {
            for x in x0..<x1 {
                let v = ptr[y * stride + x]
                if v.isFinite && v > 0 { samples.append(v) }
            }
        }
        guard !samples.isEmpty else { return nil }
        samples.sort()
        let meters = Double(samples[samples.count / 2])
        return meters * 1000.0
    }
}
