// Mass-estimation strategies — mirror of `lokma/geometry/strategies.py`.
//
//  * PixelRatioStrategy — 2D heuristic (portion · area/ref_area).
//  * VolumetricStrategy — m = V·ρ from the per-frame ScaleEstimate + height prior.
//  * AutoStrategy — the default: volumetric when calibration confidence clears
//    the threshold, else pixel_ratio tagged ":uncalibrated".
import Foundation

public enum StrategyError: Error {
    case missingScale
    case unknownStrategy(String)
}

/// Per-frame dependencies for a strategy. `scale` is recomputed each frame.
public struct MassContext {
    public let scale: ScaleEstimate?
    public let volumeEngine: VolumeEngine
    public let densityService: DensityService
    public let depthEngine: DepthVolumeEngine
    public let config: AppConfig

    public init(scale: ScaleEstimate?, volumeEngine: VolumeEngine = VolumeEngine(),
                densityService: DensityService = DensityService(),
                depthEngine: DepthVolumeEngine = DepthVolumeEngine(), config: AppConfig = AppConfig()) {
        self.scale = scale
        self.volumeEngine = volumeEngine
        self.densityService = densityService
        self.depthEngine = depthEngine
        self.config = config
    }
}

public protocol MassEstimationStrategy {
    var name: String { get }
    func estimate(_ detection: Detection, _ food: FoodRecord, _ ctx: MassContext) throws -> MassEstimate
}

public struct PixelRatioStrategy: MassEstimationStrategy {
    public let name = "pixel_ratio"
    public init() {}

    public func estimate(_ detection: Detection, _ food: FoodRecord, _ ctx: MassContext) throws -> MassEstimate {
        let refArea = (food.refArea != nil && food.refArea! > 0) ? food.refArea! : ctx.config.defaultRefArea
        let scale = refArea > 0 ? detection.maskAreaPx / refArea : 0.0
        let grams = food.portionG * scale
        return MassEstimate(grams: grams, method: name, densityUsed: nil)
    }
}

public struct VolumetricStrategy: MassEstimationStrategy {
    public let name = "volumetric"
    public init() {}

    public func estimate(_ detection: Detection, _ food: FoodRecord, _ ctx: MassContext) throws -> MassEstimate {
        guard let scale = ctx.scale else { throw StrategyError.missingScale }
        // Native-frame area (square pixels) keeps the metric scaling correct;
        // 0.0 is falsy in Python, so fall back to the working-grid area.
        let areaPx = detection.maskAreaPxFrame != 0 ? detection.maskAreaPxFrame : detection.maskAreaPx
        let realAreaCm2 = scale.areaPxToCm2(areaPx)
        let (density, densitySource) = ctx.densityService.resolve(food)
        let shape = food.geometricShape ?? "prism"
        let heightCm = Categories.heightFor(food.className)
        let volume = ctx.volumeEngine.estimateVolume(realAreaCm2: realAreaCm2, shape: shape, heightCm: heightCm)
        let grams = volume.volumeCm3 * density
        return MassEstimate(
            grams: grams,
            method: "\(name):\(densitySource.rawValue)",
            densityUsed: density,
            volumeCm3: volume.volumeCm3,
            calibrationSource: scale.source.rawValue,
            calibrationConfidence: scale.confidence
        )
    }
}

public struct AutoStrategy: MassEstimationStrategy {
    public let name = "auto"
    private let volumetric = VolumetricStrategy()
    private let pixelRatio = PixelRatioStrategy()
    public init() {}

    /// Volumetric area scaling degrades past a grazing incidence; gate it off there.
    static func tiltTrustworthy(_ tiltDeg: Double?) -> Bool {
        guard let t = tiltDeg else { return true }
        return t <= ScaleEstimate.maxForeshorteningTiltDeg
    }

    public func estimate(_ detection: Detection, _ food: FoodRecord, _ ctx: MassContext) throws -> MassEstimate {
        // 1. Measured LiDAR depth volume — most accurate, and inherently tilt-robust
        //    (the fitted plane absorbs orientation), so it needs no tilt gate.
        if let sample = detection.depthSample, let dv = ctx.depthEngine.integrate(sample) {
            let (density, densitySource) = ctx.densityService.resolve(food)
            return MassEstimate(
                grams: dv.volumeCm3 * density,
                method: "volumetric_depth:\(densitySource.rawValue)",
                densityUsed: density,
                volumeCm3: dv.volumeCm3,
                calibrationSource: "depth_plane",
                calibrationConfidence: dv.coverage
            )
        }
        // 2. Scalar volumetric when calibration is confident and not too oblique.
        let scale = ctx.scale
        let threshold = ctx.config.calibrationConfidenceThreshold
        if let scale, scale.confidence >= threshold, Self.tiltTrustworthy(scale.tiltDeg) {
            return try volumetric.estimate(detection, food, ctx)
        }
        // 3. Not trustworthy enough (low confidence or too oblique) — fall back and mark it.
        let base = try pixelRatio.estimate(detection, food, ctx)
        return MassEstimate(
            grams: base.grams,
            method: "\(base.method):uncalibrated",
            densityUsed: base.densityUsed,
            volumeCm3: base.volumeCm3,
            calibrationSource: scale?.source.rawValue,
            calibrationConfidence: scale?.confidence
        )
    }
}

public func makeStrategy(_ name: String) throws -> MassEstimationStrategy {
    switch name {
    case "pixel_ratio": return PixelRatioStrategy()
    case "volumetric": return VolumetricStrategy()
    case "auto": return AutoStrategy()
    default: throw StrategyError.unknownStrategy(name)
    }
}
