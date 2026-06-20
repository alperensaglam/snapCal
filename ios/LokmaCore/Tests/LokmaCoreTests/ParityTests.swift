// ParityTests — assert the Swift port reproduces the Python reference exactly.
//
// Loads `Fixtures/golden_vectors.json` (emitted by scripts/dump_golden_vectors.py
// from the real `lokma/` classes) and checks every case. If Swift and Python ever
// diverge, a case here fails — this is the contract that lets the two coexist.
import XCTest
@testable import LokmaCore

private let EPS = 1e-9

final class ParityTests: XCTestCase {
    private var golden: Golden!

    override func setUpWithError() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "golden_vectors", withExtension: "json", subdirectory: "Fixtures"),
            "golden_vectors.json missing from test bundle"
        )
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        golden = try decoder.decode(Golden.self, from: Data(contentsOf: url))
    }

    // Guards against silent drift of the on-device constants.
    func testAppConfigConstantsMatchReference() {
        let c = AppConfig()
        XCTAssertEqual(c.focalLengthMm, 4.2, accuracy: EPS)
        XCTAssertEqual(c.pixelPitchMm, 0.0012, accuracy: EPS)
        XCTAssertEqual(c.defaultDistanceMm, 300.0, accuracy: EPS)
        XCTAssertEqual(c.defaultRefArea, 40_000.0, accuracy: EPS)
        XCTAssertEqual(c.calibrationConfidenceThreshold, 0.5, accuracy: EPS)
        XCTAssertEqual(c.defaultPlateDiameterCm, 27.0, accuracy: EPS)
        XCTAssertEqual(c.assumedPlateFrameFraction, 0.7, accuracy: EPS)
        XCTAssertEqual(VolumeEngine.flatLayerCm, 0.8, accuracy: EPS)
        XCTAssertEqual(Categories.waterDensity, 1.0, accuracy: EPS)
        XCTAssertEqual(Categories.defaultHeightCm, 2.5, accuracy: EPS)
    }

    func testIntrinsicsFromMM() {
        for c in golden.intrinsicsFromMm {
            let intr = CameraIntrinsics.fromMM(
                focalLengthMm: c.focalMm, pixelPitchMm: c.pitchMm,
                nativeResolution: (c.nativeW, c.nativeH), workingResolution: (c.workingW, c.workingH)
            )
            XCTAssertEqual(intr.focalPx, c.expectFocalPx, accuracy: EPS)
            XCTAssertEqual(intr.principalPoint!.0, c.expectCx, accuracy: EPS)
            XCTAssertEqual(intr.principalPoint!.1, c.expectCy, accuracy: EPS)
        }
    }

    func testAreaPxToCm2() {
        for c in golden.areaPxToCm2 {
            let se = ScaleEstimate(mmPerPx: c.mmPerPx, source: .depthIntrinsics, confidence: 0.9, tiltDeg: c.tiltDeg)
            XCTAssertEqual(se.mm2PerPx2, c.expectMm2PerPx2, accuracy: EPS)
            XCTAssertEqual(se.areaPxToCm2(c.areaPx), c.expectCm2, accuracy: EPS)
        }
    }

    // Standalone sanity for the tilt correction (independent of the generator).
    func testForeshorteningMath() {
        XCTAssertEqual(ScaleEstimate.foreshortening(nil), 1.0, accuracy: EPS)
        XCTAssertEqual(ScaleEstimate.foreshortening(0.0), 1.0, accuracy: EPS)
        XCTAssertEqual(ScaleEstimate.foreshortening(60.0), 0.5, accuracy: 1e-9)   // cos 60° = 0.5
        // 80° is clamped to the 65° ceiling.
        XCTAssertEqual(ScaleEstimate.foreshortening(80.0), cos(65.0 * .pi / 180.0), accuracy: EPS)
        // A 60° tilt doubles the recovered footprint area vs top-down.
        let topdown = ScaleEstimate(mmPerPx: 0.5, source: .depthIntrinsics, confidence: 0.9, tiltDeg: 0.0)
        let tilted = ScaleEstimate(mmPerPx: 0.5, source: .depthIntrinsics, confidence: 0.9, tiltDeg: 60.0)
        XCTAssertEqual(tilted.areaPxToCm2(50_000.0), 2.0 * topdown.areaPxToCm2(50_000.0), accuracy: 1e-9)
    }

    func testVolume() {
        let engine = VolumeEngine()
        for c in golden.volume {
            let v = engine.estimateVolume(realAreaCm2: c.areaCm2, shape: c.shape, heightCm: c.heightCm)
            XCTAssertEqual(v.volumeCm3, c.expectVolumeCm3, accuracy: EPS)
            XCTAssertEqual(v.shape, c.expectShape)
            XCTAssertEqual(v.heightCm, c.expectEffectiveHeightCm, accuracy: EPS)
        }
    }

    func testDensityLookup() {
        for c in golden.densityLookup {
            assertOptional(Categories.densityFor(c.className), c.expectDensity, c.className)
        }
    }

    func testHeightLookup() {
        for c in golden.heightLookup {
            XCTAssertEqual(Categories.heightFor(c.className), c.expectHeightCm, accuracy: EPS, c.className)
        }
    }

    func testPorosityLookup() {
        for c in golden.porosityLookup {
            XCTAssertEqual(Categories.porosityFor(c.className), c.expectPorosity, accuracy: EPS, c.className)
        }
    }

    func testDensityResolve() {
        let svc = DensityService()
        for c in golden.densityResolve {
            let food = makeFood(className: c.className, density: c.dbDensity)
            let (rho, src) = svc.resolve(food)
            XCTAssertEqual(rho, c.expectDensity, accuracy: EPS, c.className)
            XCTAssertEqual(src.rawValue, c.expectSource, c.className)
        }
    }

    func testNutrition() {
        for c in golden.nutrition {
            let food = makeFood(kcal: c.kcal100, protein: c.protein100, fat: c.fat100, carbs: c.carbs100)
            let n = NutritionResult.fromFood(food, grams: c.grams)
            XCTAssertEqual(n.calories, c.expectCalories, accuracy: EPS)
            XCTAssertEqual(n.protein, c.expectProtein, accuracy: EPS)
            XCTAssertEqual(n.fat, c.expectFat, accuracy: EPS)
            XCTAssertEqual(n.carbs, c.expectCarbs, accuracy: EPS)
        }
    }

    func testPixelRatio() throws {
        let strategy = PixelRatioStrategy()
        for c in golden.pixelRatio {
            let food = makeFood(portionG: c.portionG, refArea: c.refArea)
            let det = makeDetection(areaSq: c.maskAreaPx, areaFrame: c.maskAreaPx)
            let ctx = MassContext(scale: nil, config: AppConfig())
            let m = try strategy.estimate(det, food, ctx)
            XCTAssertEqual(m.grams, c.expectGrams, accuracy: EPS)
            XCTAssertEqual(m.method, c.expectMethod)
        }
    }

    func testVolumetric() throws {
        let strategy = VolumetricStrategy()
        for c in golden.volumetric {
            let se = ScaleEstimate(mmPerPx: c.mmPerPx, source: .depthIntrinsics, confidence: 0.9)
            let food = makeFood(className: c.className, density: c.dbDensity, geometricShape: c.geometricShape)
            let det = makeDetection(areaSq: c.maskAreaPx, areaFrame: c.maskAreaPxFrame)
            let m = try strategy.estimate(det, food, MassContext(scale: se))
            XCTAssertEqual(m.grams, c.expectGrams, accuracy: 1e-6, c.className)
            XCTAssertEqual(m.volumeCm3!, c.expectVolumeCm3, accuracy: 1e-6, c.className)
            XCTAssertEqual(m.densityUsed!, c.expectDensityUsed, accuracy: EPS, c.className)
        }
    }

    func testAutoStrategy() throws {
        let strategy = AutoStrategy()
        for c in golden.autoStrategy {
            let se = c.scaleConfidence.map { ScaleEstimate(mmPerPx: 0.5, source: .depthIntrinsics, confidence: $0, tiltDeg: c.tiltDeg) }
            let food = makeFood(className: "baklava", portionG: 150.0, refArea: 50_000.0,
                                density: 1.2, geometricShape: "prism")
            let det = makeDetection(areaSq: 60_000.0, areaFrame: 90_000.0)
            let m = try strategy.estimate(det, food, MassContext(scale: se))
            XCTAssertEqual(m.grams, c.expectGrams, accuracy: 1e-6)
            XCTAssertEqual(m.method, c.expectMethod)
        }
    }

    func testCalibrationLevels() {
        let cfg = AppConfig()
        let intr = CameraIntrinsics.fromMM(focalLengthMm: cfg.focalLengthMm, pixelPitchMm: cfg.pixelPitchMm,
                                           nativeResolution: cfg.nativeResolution, workingResolution: cfg.workingResolution)
        for c in golden.calibrationLevels {
            let est: ScaleEstimate?
            switch c.level {
            case "depth_intrinsics":
                let ctx = FrameContext(intrinsics: intr, tiltDeg: 5.0, depthMm: c.depthMm)
                est = IntrinsicDepthCalibration().estimateScale(ctx)
            case "intrinsics_plane":
                let ctx = FrameContext(intrinsics: intr, tiltDeg: 10.0, depthMm: c.depthMm)
                est = IntrinsicPlaneCalibration(config: cfg).estimateScale(ctx)
            case "static_default":
                est = StaticCalibration.fromConfig(cfg).estimateScale(FrameContext.simulatedTopdown())
            default:
                est = nil
            }
            let e = est!
            XCTAssertEqual(e.mmPerPx, c.expectMmPerPx, accuracy: EPS, c.level)
            XCTAssertEqual(e.source.rawValue, c.expectSource, c.level)
            XCTAssertEqual(e.confidence, c.expectConfidence, accuracy: EPS, c.level)
        }
    }

    func testAssumedPlate() {
        let cfg = AppConfig()
        for c in golden.assumedPlate {
            let ctx = FrameContext(frameSize: (c.frameW, c.frameH))
            let est = NaturalAnchorCalibration(config: cfg).estimateScale(ctx)!
            XCTAssertEqual(est.mmPerPx, c.expectMmPerPx, accuracy: EPS)
            XCTAssertEqual(est.confidence, c.expectConfidence, accuracy: EPS)
            XCTAssertEqual(est.source.rawValue, c.expectSource)
        }
    }

    func testResolverSelection() {
        let cfg = AppConfig()
        let intr = CameraIntrinsics.fromMM(focalLengthMm: cfg.focalLengthMm, pixelPitchMm: cfg.pixelPitchMm,
                                           nativeResolution: cfg.nativeResolution, workingResolution: cfg.workingResolution)
        let resolver = CalibrationResolver([
            IntrinsicDepthCalibration(),
            IntrinsicPlaneCalibration(config: cfg),
            StaticCalibration.fromConfig(cfg),
        ])
        for c in golden.resolverSelection {
            let ctx: FrameContext
            switch c.scenario {
            case "depth+intrinsics": ctx = FrameContext(intrinsics: intr, tiltDeg: 5.0, depthMm: 300.0)
            case "intrinsics+tilt":  ctx = FrameContext(intrinsics: intr, tiltDeg: 8.0)
            default:                 ctx = FrameContext.simulatedTopdown()
            }
            let best = resolver.estimateScale(ctx)!
            XCTAssertEqual(best.source.rawValue, c.expectSource, c.scenario)
            XCTAssertEqual(best.confidence, c.expectConfidence, accuracy: EPS, c.scenario)
            XCTAssertEqual(best.mmPerPx, c.expectMmPerPx, accuracy: EPS, c.scenario)
        }
    }

    func testDepthVolume() throws {
        let engine = DepthVolumeEngine()
        for c in golden.depthVolume {
            let res = engine.integrate(depthMm: c.depth, mask: c.mask, width: c.width, height: c.height,
                                       fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy)
            if let expectV = c.expectVolumeCm3 {
                let r = try XCTUnwrap(res, c.name)
                XCTAssertEqual(r.volumeCm3, expectV, accuracy: max(1e-6, abs(expectV) * 1e-6), c.name)
                if let cov = c.expectCoverage { XCTAssertEqual(r.coverage, cov, accuracy: 1e-9, c.name) }
                XCTAssertEqual(r.method, c.expectMethod, c.name)
            } else {
                XCTAssertNil(res, c.name)
            }
        }
    }

    func testAutoDepth() throws {
        let strategy = AutoStrategy()
        for c in golden.autoDepth {
            let sample = DepthSample(depthMm: c.depth, mask: c.mask, width: c.width, height: c.height,
                                     fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy)
            let food = makeFood(className: c.className, density: c.dbDensity)
            let det = Detection(classId: 0, className: c.className, confidence: 0.9,
                                maskAreaPx: 60_000.0, maskAreaPxFrame: 90_000.0,
                                frameSize: (640, 480), depthSample: sample,
                                predictedPorosity: c.predictedPorosity)
            let m = try strategy.estimate(det, food, MassContext(scale: nil))
            XCTAssertEqual(m.grams, c.expectGrams, accuracy: max(1e-6, abs(c.expectGrams) * 1e-6))
            XCTAssertEqual(m.volumeCm3!, c.expectVolumeCm3, accuracy: 1e-6)
            XCTAssertEqual(m.method, c.expectMethod)
        }
    }

    // Independent correctness check (not generator-derived): a flat top-down box
    // must integrate to height × world footprint area.
    func testDepthVolumeFlatClosedForm() throws {
        let W = 24, H = 18, fx = 30.0, fy = 30.0, cx = 12.0, cy = 9.0
        let z0 = 300.0, hBox = 20.0
        var depth = [Double](repeating: z0, count: W * H)
        var mask = [Double](repeating: 0, count: W * H)
        var nBox = 0
        for v in 6...11 {
            for u in 8...15 {
                depth[v * W + u] = z0 - hBox
                mask[v * W + u] = 1.0
                nBox += 1
            }
        }
        let res = try XCTUnwrap(DepthVolumeEngine().integrate(
            depthMm: depth, mask: mask, width: W, height: H, fx: fx, fy: fy, cx: cx, cy: cy))
        let footprintMm2 = Double(nBox) * (z0 - hBox) * (z0 - hBox) / (fx * fy)
        let expectedCm3 = hBox * footprintMm2 / 1000.0
        XCTAssertEqual(res.volumeCm3, expectedCm3, accuracy: 1e-6)
        XCTAssertEqual(res.coverage, 1.0, accuracy: 1e-12)
    }

    // MARK: - helpers

    private func assertOptional(_ got: Double?, _ want: Double?, _ msg: String) {
        switch (got, want) {
        case let (g?, w?): XCTAssertEqual(g, w, accuracy: EPS, msg)
        case (nil, nil): break
        default: XCTFail("optional mismatch for \(msg): got \(String(describing: got)), want \(String(describing: want))")
        }
    }

    private func makeFood(
        className: String = "generic", kcal: Double = 200, protein: Double = 5, fat: Double = 8,
        carbs: Double = 25, portionG: Double = 150, refArea: Double? = 50_000,
        density: Double? = nil, geometricShape: String? = "prism"
    ) -> FoodRecord {
        FoodRecord(className: className, caloriesPer100g: kcal, proteinPer100g: protein,
                   fatPer100g: fat, carbsPer100g: carbs, portionG: portionG,
                   refArea: refArea, density: density, geometricShape: geometricShape)
    }

    private func makeDetection(areaSq: Double, areaFrame: Double) -> Detection {
        Detection(classId: 0, className: "generic", confidence: 0.9,
                  maskAreaPx: areaSq, maskAreaPxFrame: areaFrame, frameSize: (640, 480))
    }
}

// MARK: - Decodable mirror of golden_vectors.json (keyDecodingStrategy = convertFromSnakeCase)

private struct Golden: Decodable {
    let intrinsicsFromMm: [IntrinsicsCase]
    let areaPxToCm2: [AreaCase]
    let volume: [VolumeCase]
    let densityLookup: [DensityLookupCase]
    let heightLookup: [HeightLookupCase]
    let porosityLookup: [PorosityLookupCase]
    let densityResolve: [DensityResolveCase]
    let nutrition: [NutritionCase]
    let pixelRatio: [PixelRatioCase]
    let volumetric: [VolumetricCase]
    let autoStrategy: [AutoCase]
    let calibrationLevels: [CalibrationCase]
    let assumedPlate: [AssumedPlateCase]
    let resolverSelection: [ResolverCase]
    let depthVolume: [DepthVolumeCase]
    let autoDepth: [AutoDepthCase]

    struct IntrinsicsCase: Decodable {
        let focalMm, pitchMm: Double
        let nativeW, nativeH, workingW, workingH: Int
        let expectFocalPx, expectCx, expectCy: Double
    }
    struct AreaCase: Decodable {
        let mmPerPx, areaPx: Double
        let tiltDeg: Double?
        let expectMm2PerPx2, expectCm2: Double
    }
    struct VolumeCase: Decodable {
        let areaCm2: Double; let shape: String; let heightCm, expectVolumeCm3: Double
        let expectShape: String; let expectEffectiveHeightCm: Double
    }
    struct DensityLookupCase: Decodable { let className: String; let expectDensity: Double? }
    struct HeightLookupCase: Decodable { let className: String; let expectHeightCm: Double }
    struct PorosityLookupCase: Decodable { let className: String; let expectPorosity: Double }
    struct DensityResolveCase: Decodable {
        let className: String; let dbDensity: Double?; let expectDensity: Double; let expectSource: String
    }
    struct NutritionCase: Decodable {
        let kcal100, protein100, fat100, carbs100, grams: Double
        let expectCalories, expectProtein, expectFat, expectCarbs: Double
    }
    struct PixelRatioCase: Decodable {
        let maskAreaPx: Double; let refArea: Double?; let portionG, defaultRefArea, expectGrams: Double
        let expectMethod: String
    }
    struct VolumetricCase: Decodable {
        let mmPerPx, maskAreaPxFrame, maskAreaPx: Double; let className: String
        let dbDensity: Double?; let geometricShape: String
        let expectGrams, expectVolumeCm3, expectDensityUsed: Double
    }
    struct AutoCase: Decodable {
        let scaleConfidence, tiltDeg: Double?; let threshold, expectGrams: Double; let expectMethod: String
    }
    struct CalibrationCase: Decodable {
        let level: String; let focalPx: Double; let depthMm, defaultDistanceMm: Double?
        let expectMmPerPx: Double; let expectSource: String; let expectConfidence: Double
    }
    struct AssumedPlateCase: Decodable {
        let frameW, frameH: Int; let plateDiameterCm, assumedFraction, expectMmPerPx, expectConfidence: Double
        let expectSource: String
    }
    struct ResolverCase: Decodable {
        let scenario, expectSource: String; let expectConfidence, expectMmPerPx: Double
    }
    struct DepthVolumeCase: Decodable {
        let name: String
        let width, height: Int
        let fx, fy, cx, cy: Double
        let depth, mask: [Double]
        let expectVolumeCm3, expectCoverage: Double?
        let expectMethod: String?
    }
    struct AutoDepthCase: Decodable {
        let width, height: Int
        let fx, fy, cx, cy: Double
        let depth, mask: [Double]
        let className: String
        let dbDensity: Double?
        let predictedPorosity: Double?
        let expectGrams, expectVolumeCm3: Double
        let expectMethod: String
    }
}
