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

    public let capture: ARCaptureController
    private let model: FoodSegModel?
    private let engine: EstimationEngine?
    private let config = AppConfig()
    private let inferenceQueue = DispatchQueue(label: "com.lokma.inference", qos: .userInitiated)
    private var busy = false

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
}

extension LokmaViewModel: ARCaptureDelegate {
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
                    self?.labels = results.map { $0.label }
                    self?.calibrationSource = scale?.source.rawValue ?? "none"
                    self?.calibrationConfidence = scale?.confidence ?? 0
                    self?.status = results.isEmpty ? "No food detected" : "\(results.count) item(s)"
                    self?.busy = false
                }
            }
        }
    }
}
