// InferenceService — an actor that owns the (non-Sendable) inference models and runs
// the per-frame pipeline off the main actor.
//
// Confining the models to an actor is the Swift-6-clean way to keep frame processing
// concurrent without data races. It replaces the old `Task{@MainActor} →
// DispatchQueue.async{@Sendable}` pattern, whose @Sendable closure captured non-Sendable
// state (FoodSegModel, CVPixelBuffer, FillDensityModel?, EstimationEngine) and warned.
import CoreVideo
import Foundation
import LokmaCore

/// One frame's inputs, bundled so they cross into the actor as a single Sendable value.
/// `@unchecked` is justified: ARKit hands us a fresh, read-only `capturedImage` per frame
/// and the pipeline never mutates it, so transferring it is safe.
public struct FrameInput: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    public let context: FrameContext
    public let depth: DepthCapture?

    public init(pixelBuffer: CVPixelBuffer, context: FrameContext, depth: DepthCapture?) {
        self.pixelBuffer = pixelBuffer
        self.context = context
        self.depth = depth
    }
}

public actor InferenceService {
    private let model: FoodSegModel
    private let engine: EstimationEngine
    private let fillModel: FillDensityModel?
    private let depthEngine = DepthVolumeEngine()

    public init(model: FoodSegModel, engine: EstimationEngine, fillModel: FillDensityModel?) {
        self.model = model
        self.engine = engine
        self.fillModel = fillModel
    }

    /// Segment → per-detection depth sample (+ fill-density when the head is bundled) →
    /// mass estimate. Serialized by the actor; the view model gates re-entry per frame.
    public func process(_ input: FrameInput) -> (results: [AnnotatedResult], scale: ScaleEstimate?) {
        let prediction = (try? model.predict(pixelBuffer: input.pixelBuffer))
            ?? (detections: [RawDetection](), frameSize: (0, 0))
        let detections = prediction.detections.map { raw -> Detection in
            let sample = input.depth.flatMap { DepthSampler.sample(detection: raw, capture: $0) }
            var fill: Double? = nil
            if let sample, let dv = depthEngine.integrate(sample) {
                fill = fillModel?.predict(capturedImage: input.pixelBuffer, raw: raw,
                                          volumeCm3: dv.volumeCm3, coverage: dv.coverage)
            }
            return DetectionBuilder.makeDetection(from: raw, frameSize: prediction.frameSize,
                                                  depthSample: sample, predictedFillDensity: fill)
        }
        return engine.process(detections: detections, context: input.context)
    }
}
