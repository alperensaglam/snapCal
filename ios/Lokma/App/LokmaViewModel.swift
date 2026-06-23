// LokmaViewModel — wires capture → segmentation → estimation and publishes the
// overlay state. It owns the only mutable state in the app; everything it calls
// (LokmaCore math, KnowledgeStore, EstimationEngine) is pure or read-only.
import Combine
import CoreVideo
import Foundation
import LokmaCore
import simd

@MainActor
public final class LokmaViewModel: ObservableObject {
    @Published public private(set) var labels: [String] = []
    @Published public private(set) var calibrationSource: String = "—"
    @Published public private(set) var calibrationConfidence: Double = 0
    @Published public private(set) var status: String = "Starting…"
    /// Live, actionable instruction derived from the same gates the math uses; nil = good to shoot.
    @Published public private(set) var guidance: CaptureGuidance?

    public let capture: ARCaptureController
    private let model: FoodSegModel?
    private let engine: EstimationEngine?
    private let config = AppConfig()
    private let inferenceQueue = DispatchQueue(label: "com.lokma.inference", qos: .userInitiated)
    /// Per-instance temporal smoothing of displayed grams (presentation stability).
    private let tracker = InstanceTracker()
    /// Phase 8: depth engine (for the fill-head's V/coverage features) + the fill-density
    /// head (nil when not bundled → the analytic ρ·(1−P) path is kept).
    private let depthEngine = DepthVolumeEngine()
    private let fillModel = FillDensityModel()
    private var busy = false

    // Live capture-quality signals feeding `guidance` (updated from two cadences:
    // tracking/tilt at ~10 Hz, depth-source at the inference rate).
    private var trackingQuality: TrackingQuality = .normal
    private var latestTiltDeg: Double?
    private var depthDegraded = false

    public init() {
        self.capture = ARCaptureController(config: config)
        var loadedModel: FoodSegModel?
        var loadedEngine: EstimationEngine?
        var bootStatus = ""
        do {
            let store = try KnowledgeStore()
            loadedEngine = EstimationEngine(store: store, config: config)
            bootStatus += "KB \(store.count) foods (\(store.activeModelVersion ?? "?")). "
        } catch {
            bootStatus += "KB load failed: \(error). "
        }
        do {
            loadedModel = try FoodSegModel()
            bootStatus += "Model \(loadedModel?.metadata.modelVersion ?? "?")."
        } catch {
            bootStatus += "Model load failed (export FoodSeg.mlpackage): \(error)."
        }
        self.model = loadedModel
        self.engine = loadedEngine
        self.status = bootStatus
        self.capture.delegate = self
    }

    public func start() { capture.start() }
    public func stop() { capture.pause() }

    /// Re-label results with per-instance temporally smoothed grams (and matching kcal),
    /// reusing `AnnotatedResult.label` so the overlay format stays in one place. Each
    /// detection is smoothed within its own spatial track (world centroid), so two
    /// plates of the same dish don't cross-contaminate.
    private func stabilizedLabels(_ results: [AnnotatedResult], cameraTransform: simd_float4x4?) -> [String] {
        let now = Date()   // one timestamp per frame so same-class plates can't share a track
        return results.map { r in
            guard let food = r.food, let mass = r.mass else { return r.label }
            let position = worldCentroid(r.detection, cameraTransform: cameraTransform)
            let grams = tracker.smooth(className: r.detection.className, grams: mass.grams,
                                       position: position, now: now)
            let smoothedMass = MassEstimate(
                grams: grams, method: mass.method, densityUsed: mass.densityUsed,
                volumeCm3: mass.volumeCm3, calibrationSource: mass.calibrationSource,
                calibrationConfidence: mass.calibrationConfidence
            )
            let nutrition = NutritionResult.fromFood(food, grams: grams)
            return AnnotatedResult(detection: r.detection, food: food,
                                   mass: smoothedMass, nutrition: nutrition).label
        }
    }

    /// World centroid (metres) of a detection from its depth sample + camera pose; nil
    /// → InstanceTracker falls back to class-bucket smoothing. The depth centroid is in
    /// the CV intrinsics frame (+Z forward, +Y down); flip Y,Z to ARKit camera-local
    /// (−Z forward, +Y up) before applying the camera→world transform.
    private func worldCentroid(_ detection: Detection, cameraTransform: simd_float4x4?) -> (Double, Double, Double)? {
        guard let transform = cameraTransform, let c = detection.depthSample?.cameraCentroidMm() else { return nil }
        let cam = SIMD4<Float>(Float(c.x / 1000.0), Float(-c.y / 1000.0), Float(-c.z / 1000.0), 1)
        let world = transform * cam
        return (Double(world.x), Double(world.y), Double(world.z))
    }

    /// Recompute the live on-screen guidance from the latest quality signals.
    /// Priority: fundamental tracking loss > steep angle > degraded depth — show the
    /// single most-actionable fix, and only republish when it actually changes.
    private func refreshGuidance() {
        let next = computeGuidance()
        if next != guidance { guidance = next }
    }

    private func computeGuidance() -> CaptureGuidance? {
        if trackingQuality != .normal {
            return CaptureGuidance(
                level: .warning,
                message: "🔍 Poor lighting or fast movement. Please hold steady or move to a brighter spot."
            )
        }
        if let tilt = latestTiltDeg, tilt > 60.0 {
            return CaptureGuidance(
                level: .warning,
                message: "⚠️ Please straighten your phone (Angle too steep)"
            )
        }
        if depthDegraded {
            return CaptureGuidance(
                level: .error,
                message: "🛑 Depth connection lost. Showing generic estimation. Try moving closer to the plate."
            )
        }
        return nil
    }
}

/// One piece of live, actionable guidance derived from the estimation math gates.
/// `.warning` (yellow) is a recoverable nudge; `.error` (red) means the volumetric
/// path is degraded to a generic estimate.
public struct CaptureGuidance: Equatable {
    public enum Level: Equatable { case warning, error }
    public let level: Level
    public let message: String

    public init(level: Level, message: String) {
        self.level = level
        self.message = message
    }
}

extension LokmaViewModel: ARCaptureDelegate {
    public nonisolated func capture(_ controller: ARCaptureController, didUpdateTracking quality: TrackingQuality, tiltDeg: Double?) {
        Task { @MainActor in
            self.trackingQuality = quality
            self.latestTiltDeg = tiltDeg
            self.refreshGuidance()
        }
    }

    public nonisolated func capture(_ controller: ARCaptureController, didProduce pixelBuffer: CVPixelBuffer, context: FrameContext, depth: DepthCapture?) {
        Task { @MainActor in
            guard !busy, let model, let engine else { return }
            let depthEngine = self.depthEngine
            let fillModel = self.fillModel
            busy = true
            inferenceQueue.async { [weak self] in
                // predict center-crops to a square and reports the (S, S) frame the
                // detections live on; DetectionBuilder reuses that for native-grid area.
                let prediction = (try? model.predict(pixelBuffer: pixelBuffer))
                    ?? (detections: [RawDetection](), frameSize: (0, 0))
                // Per detection: resample its mask onto the LiDAR depth grid (Tier 2), and
                // — when the fill-density head is bundled — predict D from the masked crop +
                // depth scalars (mass = V·D). nil at any step → the analytic ρ·(1−P) path.
                let detections = prediction.detections.map { raw -> Detection in
                    let sample = depth.flatMap { DepthSampler.sample(detection: raw, capture: $0) }
                    var fill: Double? = nil
                    if let sample, let dv = depthEngine.integrate(sample) {
                        fill = fillModel?.predict(capturedImage: pixelBuffer, raw: raw,
                                                  volumeCm3: dv.volumeCm3, coverage: dv.coverage)
                    }
                    return DetectionBuilder.makeDetection(from: raw, frameSize: prediction.frameSize,
                                                          depthSample: sample, predictedFillDensity: fill)
                }
                let (results, scale) = engine.process(detections: detections, context: context)
                Task { @MainActor in
                    self?.labels = self?.stabilizedLabels(results, cameraTransform: depth?.cameraTransform) ?? []
                    self?.calibrationSource = scale?.source.rawValue ?? "none"
                    self?.calibrationConfidence = scale?.confidence ?? 0
                    self?.status = results.isEmpty ? "No food detected" : "\(results.count) item(s)"
                    // Depth is "degraded" whenever the high-confidence LiDAR depth path
                    // isn't the active calibration source (sparse/lost depth → fallback).
                    self?.depthDegraded = (scale?.source != .depthIntrinsics)
                    self?.refreshGuidance()
                    self?.busy = false
                }
            }
        }
    }
}
