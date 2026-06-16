// EstimationEngine — the iOS mirror of `InferencePipeline.process_frame` (the
// post-segmentation half). Detections + a FrameContext go in; AnnotatedResults
// come out. One calibration per frame, shared by every detection — identical to
// the Python contract, but every piece is the parity-tested LokmaCore code.
import Foundation
import LokmaCore

public final class EstimationEngine {
    private let store: KnowledgeStore
    private let resolver: CalibrationResolver
    private let strategy: MassEstimationStrategy
    private let volumeEngine = VolumeEngine()
    private let densityService = DensityService()
    private let config: AppConfig

    public init(store: KnowledgeStore, config: AppConfig = AppConfig()) {
        self.store = store
        self.config = config
        self.resolver = CalibrationResolver.makeDefault(config: config)
        self.strategy = (try? makeStrategy(config.strategy)) ?? AutoStrategy()
    }

    public func process(detections: [Detection], context: FrameContext) -> (results: [AnnotatedResult], scale: ScaleEstimate?) {
        let scale = resolver.estimateScale(context)
        let ctx = MassContext(scale: scale, volumeEngine: volumeEngine, densityService: densityService, config: config)

        var annotated: [AnnotatedResult] = []
        for detection in detections {
            guard let food = store.foodRecord(for: detection.className) else {
                annotated.append(AnnotatedResult(detection: detection))
                continue
            }
            guard let mass = try? strategy.estimate(detection, food, ctx) else {
                annotated.append(AnnotatedResult(detection: detection, food: food))
                continue
            }
            let nutrition = NutritionResult.fromFood(food, grams: mass.grams)
            annotated.append(AnnotatedResult(detection: detection, food: food, mass: mass, nutrition: nutrition))
        }
        return (annotated, scale)
    }
}
