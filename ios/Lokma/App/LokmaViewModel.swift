// LokmaViewModel — wires capture → segmentation → estimation and publishes the
// overlay state. It owns the only mutable state in the app; everything it calls
// (LokmaCore math, KnowledgeStore, EstimationEngine) is pure or read-only.
import Combine
import CoreVideo
import Foundation
import LokmaCore

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
    /// Temporal smoothing of displayed grams (presentation stability under motion).
    private let stabilizer = MassStabilizer()
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

    /// Re-label results with temporally smoothed grams (and the matching kcal),
    /// reusing `AnnotatedResult.label` so the overlay format stays in one place.
    private func stabilizedLabels(_ results: [AnnotatedResult]) -> [String] {
        results.map { r in
            guard let food = r.food, let mass = r.mass else { return r.label }
            let grams = stabilizer.smooth(className: r.detection.className, grams: mass.grams)
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

    public nonisolated func capture(_ controller: ARCaptureController, didProduce pixelBuffer: CVPixelBuffer, context: FrameContext) {
        Task { @MainActor in
            guard !busy, let model, let engine else { return }
            busy = true
            inferenceQueue.async { [weak self] in
                // predict center-crops to a square and reports the (S, S) frame the
                // detections live on; DetectionBuilder reuses that for native-grid area.
                let prediction = (try? model.predict(pixelBuffer: pixelBuffer))
                    ?? (detections: [RawDetection](), frameSize: (0, 0))
                let detections = prediction.detections.map {
                    DetectionBuilder.makeDetection(from: $0, frameSize: prediction.frameSize)
                }
                let (results, scale) = engine.process(detections: detections, context: context)
                Task { @MainActor in
                    self?.labels = self?.stabilizedLabels(results) ?? []
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
