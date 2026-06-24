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
    private let config = AppConfig()
    /// Inference (segmentation + estimation) confined to an actor; nil if the model or
    /// knowledge base failed to load. Replaces the old DispatchQueue + non-Sendable captures.
    private let inference: InferenceService?
    /// Per-instance temporal smoothing of displayed grams (presentation stability).
    private let tracker = InstanceTracker()
    /// Drop frames while one is in flight (the actor processes one frame at a time).
    private var inflight = false

    // Live capture-quality signals feeding `guidance` (updated from two cadences:
    // tracking/tilt at ~10 Hz, depth-source at the inference rate).
    private var trackingQuality: TrackingQuality = .normal
    private var latestTiltDeg: Double?
    private var depthDegraded = false

    public init() {
        self.capture = ARCaptureController(config: config)
        var bootStatus = ""
        var engine: EstimationEngine?
        do {
            let store = try KnowledgeStore()
            engine = EstimationEngine(store: store, config: config)
            bootStatus += "KB \(store.count) foods (\(store.activeModelVersion ?? "?")). "
        } catch {
            bootStatus += "KB load failed: \(error). "
        }
        var model: FoodSegModel?
        do {
            let loaded = try FoodSegModel()
            loaded.scoreThreshold = config.confThreshold   // B1: confidence floor (single source of truth)
            model = loaded
            bootStatus += "Model \(loaded.metadata.modelVersion)."
        } catch {
            bootStatus += "Model load failed (export FoodSeg.mlpackage): \(error)."
        }
        if let model, let engine {
            // FillDensityModel/VolumeModel are optional heads: nil when not bundled, and the
            // pipeline degrades gracefully (analytic ρ·(1−P) / scalar fallback respectively).
            self.inference = InferenceService(model: model, engine: engine,
                                              fillModel: FillDensityModel(), volumeModel: VolumeModel())
        } else {
            self.inference = nil
        }
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
        // Bundle the (non-Sendable) buffer into a Sendable wrapper *before* the Task, so
        // nothing non-Sendable is captured by the @Sendable Task closure.
        let input = FrameInput(pixelBuffer: pixelBuffer, context: context, depth: depth)
        Task { @MainActor in
            guard !inflight, let inference else { return }
            inflight = true
            let (results, scale) = await inference.process(input)
            labels = stabilizedLabels(results, cameraTransform: input.depth?.cameraTransform)
            calibrationSource = scale?.source.rawValue ?? "none"
            calibrationConfidence = scale?.confidence ?? 0
            status = results.isEmpty ? "No food detected" : "\(results.count) item(s)"
            // Depth is "degraded" whenever the high-confidence LiDAR depth path isn't the
            // active calibration source (sparse/lost depth → fallback).
            depthDegraded = (scale?.source != .depthIntrinsics)
            refreshGuidance()
            inflight = false
        }
    }
}
