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
    private let volumeModel: VolumeModel?
    private let depthEngine = DepthVolumeEngine()

    public init(model: FoodSegModel, engine: EstimationEngine,
                fillModel: FillDensityModel?, volumeModel: VolumeModel?) {
        self.model = model
        self.engine = engine
        self.fillModel = fillModel
        self.volumeModel = volumeModel
    }

    /// Segment → per-detection volume + fill-density (LiDAR depth when available, else the
    /// RGB volume head) → mass estimate. Serialized by the actor; the view model gates
    /// re-entry per frame.
    public func process(_ input: FrameInput) -> (results: [AnnotatedResult], scale: ScaleEstimate?) {
        let prediction = (try? model.predict(pixelBuffer: input.pixelBuffer))
            ?? (detections: [RawDetection](), frameSize: (0, 0))
        let detections = prediction.detections.map { raw -> Detection in
            let sample = input.depth.flatMap { DepthSampler.sample(detection: raw, capture: $0) }
            var fill: Double? = nil
            var predictedVolume: Double? = nil
            if let sample, let dv = depthEngine.integrate(sample) {
                // LiDAR path: measured depth volume feeds the fill head (V·D).
                fill = fillModel?.predict(capturedImage: input.pixelBuffer, raw: raw,
                                          volumeCm3: dv.volumeCm3, coverage: dv.coverage)
            } else if let vp = volumeModel?.predict(capturedImage: input.pixelBuffer, raw: raw) {
                // Non-LiDAR fallback: predict V from RGB, then D from the fill head fed
                // V_pred + a coverage=1.0 proxy (no measured depth coverage exists) → V·D.
                predictedVolume = vp
                fill = fillModel?.predict(capturedImage: input.pixelBuffer, raw: raw,
                                          volumeCm3: vp, coverage: 1.0)
            }
            return DetectionBuilder.makeDetection(from: raw, frameSize: prediction.frameSize,
                                                  depthSample: sample, predictedFillDensity: fill,
                                                  predictedVolumeCm3: predictedVolume)
        }
        return engine.process(detections: detections, context: input.context)
    }
}
