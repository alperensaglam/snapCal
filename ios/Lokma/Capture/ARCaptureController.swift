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

/// App-level camera tracking quality, decoupled from ARKit's enum so view state
/// need not import ARKit. `.limited` folds every reason ARKit tracks poorly
/// (initialising, relocalising, insufficient features, excessive motion).
public enum TrackingQuality: Equatable {
    case normal
    case limited
    case unavailable

    init(_ state: ARCamera.TrackingState) {
        switch state {
        case .normal: self = .normal
        case .limited: self = .limited
        case .notAvailable: self = .unavailable
        @unknown default: self = .unavailable
        }
    }
}

public protocol ARCaptureDelegate: AnyObject {
    /// Called on a background-friendly cadence with one capture + its context, plus
    /// the optional LiDAR depth bundle the Tier 2 depth path consumes (nil without LiDAR).
    func capture(_ controller: ARCaptureController, didProduce pixelBuffer: CVPixelBuffer, context: FrameContext, depth: DepthCapture?)

    /// Live capture-quality signal (~10 Hz), emitted even when inference is gated
    /// or skipped, so the UI can guide the user. `tiltDeg` is the current top-down
    /// angle (nil only if no camera pose is available).
    func capture(_ controller: ARCaptureController, didUpdateTracking quality: TrackingQuality, tiltDeg: Double?)
}

public extension ARCaptureDelegate {
    func capture(_ controller: ARCaptureController, didUpdateTracking quality: TrackingQuality, tiltDeg: Double?) {}
}

public final class ARCaptureController: NSObject, ARSessionDelegate {
    public let session = ARSession()
    public weak var delegate: ARCaptureDelegate?

    private let config: AppConfig
    private let workingWidth: Int
    private var lastProcessed: TimeInterval = 0
    /// Cap inference cadence; ARKit delivers ~60 fps but we only need a few.
    public var minInterval: TimeInterval = 1.0 / 4.0
    /// Capture-quality signal cadence — higher than inference for responsive guidance.
    public var minTrackingInterval: TimeInterval = 1.0 / 10.0
    private var lastTrackingNotify: TimeInterval = 0

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
        let camera = frame.camera

        // Live guidance signal — emitted even while inference is gated/skipped so the
        // UI can react to limited tracking or a steep angle in real time.
        if frame.timestamp - lastTrackingNotify >= minTrackingInterval {
            lastTrackingNotify = frame.timestamp
            delegate?.capture(self, didUpdateTracking: TrackingQuality(camera.trackingState),
                              tiltDeg: tiltDegFromTopDown(camera.transform))
        }

        guard frame.timestamp - lastProcessed >= minInterval else { return }
        // Pose-derived scale (tilt + intrinsics) is only trustworthy under normal
        // tracking; skip initialising/relocalising frames so a bad camera transform
        // can't poison the depth sample or the tilt correction.
        guard case .normal = camera.trackingState else { return }
        lastProcessed = frame.timestamp
        let context = makeContext(from: frame)
        let depth = makeDepthCapture(from: frame)
        delegate?.capture(self, didProduce: frame.capturedImage, context: context, depth: depth)
    }

    /// Bundle the full LiDAR depth field + capture geometry for the Tier 2 depth path.
    private func makeDepthCapture(from frame: ARFrame) -> DepthCapture? {
        guard let field = DepthSampler.depthField(from: frame.sceneDepth ?? frame.smoothedSceneDepth) else { return nil }
        let camera = frame.camera
        let imageW = Int(camera.imageResolution.width.rounded())
        let imageH = Int(camera.imageResolution.height.rounded())
        let intr = camera.intrinsics
        let geometry = FrameGeometry(
            imageWidth: imageW, imageHeight: imageH,
            depthWidth: field.width, depthHeight: field.height,
            fx: Double(intr.columns.0.x), fy: Double(intr.columns.1.y),
            cx: Double(intr.columns.2.x), cy: Double(intr.columns.2.y)
        )
        return DepthCapture(field: field, geometry: geometry)
    }

    // MARK: - ARFrame -> FrameContext

    func makeContext(from frame: ARFrame) -> FrameContext {
        let camera = frame.camera
        let imageRes = camera.imageResolution
        let imageW = Int(imageRes.width.rounded())
        let imageH = Int(imageRes.height.rounded())

        // Intrinsics: fx is in pixels of the captured image; rescale to the 640
        // working grid the masks (and ref_area) are measured on. FoodSegModel
        // center-crops the buffer to a square (side = min(w,h)) before the 640
        // resize, so the focal must scale by the *crop* side — not the full width —
        // for the volumetric metric path to stay consistent with what the model saw.
        let cropSide = min(imageW, imageH)
        let fxCaptured = Double(camera.intrinsics.columns.0.x)
        let scaleToWorking = Double(workingWidth) / max(1.0, Double(cropSide))
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
            frameSize: (cropSide, cropSide)   // the centered square the model + estimators see
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

    /// Robust representative depth near the frame centre, in mm; nil if none.
    /// Drops low-confidence LiDAR returns (`ARConfidenceLevel.low`) and MAD-rejects
    /// outliers before the median, so edge/shiny-surface noise can't drag the scale.
    private func representativeDepthMm(_ depth: ARDepthData?) -> Double? {
        guard let depth else { return nil }
        let map = depth.depthMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }

        let w = CVPixelBufferGetWidth(map)
        let h = CVPixelBufferGetHeight(map)
        let stride = CVPixelBufferGetBytesPerRow(map) / MemoryLayout<Float32>.size
        let ptr = base.assumingMemoryBound(to: Float32.self)

        // Confidence map shares the depth map's dimensions (1 byte/px: 0 low, 1 med, 2 high).
        let confMap = depth.confidenceMap
        if let confMap { CVPixelBufferLockBaseAddress(confMap, .readOnly) }
        defer { if let confMap { CVPixelBufferUnlockBaseAddress(confMap, .readOnly) } }
        let confPtr = confMap.flatMap { CVPixelBufferGetBaseAddress($0)?.assumingMemoryBound(to: UInt8.self) }
        let confStride = confMap.map { CVPixelBufferGetBytesPerRow($0) / MemoryLayout<UInt8>.size } ?? 0
        let lowConf = UInt8(ARConfidenceLevel.low.rawValue)

        // Sample a centred window (~40% of each side); keep valid, non-low-confidence depths.
        var samples: [Float] = []
        let x0 = Int(Double(w) * 0.3), x1 = Int(Double(w) * 0.7)
        let y0 = Int(Double(h) * 0.3), y1 = Int(Double(h) * 0.7)
        for y in y0..<y1 {
            for x in x0..<x1 {
                let v = ptr[y * stride + x]
                guard v.isFinite, v > 0 else { continue }
                if let confPtr, confPtr[y * confStride + x] == lowConf { continue }
                samples.append(v)
            }
        }
        guard !samples.isEmpty else { return nil }

        let med = Self.median(&samples)                       // sorts samples in place
        guard samples.count >= 8 else { return Double(med) * 1000.0 }
        // Median absolute deviation outlier rejection, then the median of survivors.
        var deviations = samples.map { abs($0 - med) }
        let mad = Self.median(&deviations)
        let cutoff = 3.0 * mad + 1e-6
        var kept = samples.filter { abs($0 - med) <= cutoff }
        if kept.isEmpty { kept = samples }
        return Double(Self.median(&kept)) * 1000.0
    }

    /// In-place median (sorts the input); caller guarantees a non-empty array.
    private static func median(_ xs: inout [Float]) -> Float {
        xs.sort()
        return xs[xs.count / 2]
    }
}
