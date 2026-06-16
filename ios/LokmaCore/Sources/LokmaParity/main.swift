// LokmaParity — runnable parity check (no XCTest), for Command-Line-Tools machines.
//
// Loads the golden vectors emitted by scripts/dump_golden_vectors.py and asserts
// the LokmaCore Swift port reproduces the Python reference. Mirrors the XCTest
// target one-for-one; exits non-zero on any mismatch so it can gate CI.
//
//     swift run LokmaParity
import Foundation
import LokmaCore

let EPS = 1e-9

// MARK: - tiny assertion harness

var failures: [String] = []
var checks = 0

func approx(_ a: Double, _ b: Double, _ eps: Double = EPS) -> Bool { abs(a - b) <= eps }

func expect(_ cond: Bool, _ msg: @autoclosure () -> String) {
    checks += 1
    if !cond { failures.append(msg()) }
}

func expectEq(_ a: Double, _ b: Double, _ msg: String, _ eps: Double = EPS) {
    expect(approx(a, b, eps), "\(msg): \(a) != \(b)")
}

func expectEq(_ a: String, _ b: String, _ msg: String) {
    expect(a == b, "\(msg): \"\(a)\" != \"\(b)\"")
}

func expectOptional(_ got: Double?, _ want: Double?, _ msg: String) {
    switch (got, want) {
    case let (g?, w?): expectEq(g, w, msg)
    case (nil, nil): checks += 1
    default: expect(false, "\(msg): optional mismatch got \(String(describing: got)) want \(String(describing: want))")
    }
}

// MARK: - locate + decode fixture (relative to this source file; no bundle needed)

let fixtureURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Tests/LokmaCoreTests/Fixtures/golden_vectors.json")

guard let data = try? Data(contentsOf: fixtureURL) else {
    FileHandle.standardError.write(Data("FATAL: cannot read \(fixtureURL.path)\n".utf8))
    exit(2)
}

let decoder = JSONDecoder()
decoder.keyDecodingStrategy = .convertFromSnakeCase
let golden: Golden
do {
    golden = try decoder.decode(Golden.self, from: data)
} catch {
    FileHandle.standardError.write(Data("FATAL: decode failed: \(error)\n".utf8))
    exit(2)
}

// MARK: - helpers

func makeFood(
    className: String = "generic", kcal: Double = 200, protein: Double = 5, fat: Double = 8,
    carbs: Double = 25, portionG: Double = 150, refArea: Double? = 50_000,
    density: Double? = nil, geometricShape: String? = "prism"
) -> FoodRecord {
    FoodRecord(className: className, caloriesPer100g: kcal, proteinPer100g: protein,
               fatPer100g: fat, carbsPer100g: carbs, portionG: portionG,
               refArea: refArea, density: density, geometricShape: geometricShape)
}

func makeDetection(areaSq: Double, areaFrame: Double) -> Detection {
    Detection(classId: 0, className: "generic", confidence: 0.9,
              maskAreaPx: areaSq, maskAreaPxFrame: areaFrame, frameSize: (640, 480))
}

let cfg = AppConfig()
let intr = CameraIntrinsics.fromMM(focalLengthMm: cfg.focalLengthMm, pixelPitchMm: cfg.pixelPitchMm,
                                   nativeResolution: cfg.nativeResolution, workingResolution: cfg.workingResolution)

// MARK: - config drift guard

expectEq(cfg.focalLengthMm, 4.2, "config.focalLengthMm")
expectEq(cfg.pixelPitchMm, 0.0012, "config.pixelPitchMm")
expectEq(cfg.defaultDistanceMm, 300.0, "config.defaultDistanceMm")
expectEq(cfg.defaultRefArea, 40_000.0, "config.defaultRefArea")
expectEq(cfg.calibrationConfidenceThreshold, 0.5, "config.calibrationConfidenceThreshold")
expectEq(VolumeEngine.flatLayerCm, 0.8, "VolumeEngine.flatLayerCm")
expectEq(Categories.waterDensity, 1.0, "Categories.waterDensity")
expectEq(Categories.defaultHeightCm, 2.5, "Categories.defaultHeightCm")

// MARK: - groups

for c in golden.intrinsicsFromMm {
    let i = CameraIntrinsics.fromMM(focalLengthMm: c.focalMm, pixelPitchMm: c.pitchMm,
                                    nativeResolution: (c.nativeW, c.nativeH), workingResolution: (c.workingW, c.workingH))
    expectEq(i.focalPx, c.expectFocalPx, "intrinsics.focalPx")
    expectEq(i.principalPoint!.0, c.expectCx, "intrinsics.cx")
    expectEq(i.principalPoint!.1, c.expectCy, "intrinsics.cy")
}

for c in golden.areaPxToCm2 {
    let se = ScaleEstimate(mmPerPx: c.mmPerPx, source: .depthIntrinsics, confidence: 0.9)
    expectEq(se.mm2PerPx2, c.expectMm2PerPx2, "mm2PerPx2")
    expectEq(se.areaPxToCm2(c.areaPx), c.expectCm2, "areaPxToCm2")
}

let engine = VolumeEngine()
for c in golden.volume {
    let v = engine.estimateVolume(realAreaCm2: c.areaCm2, shape: c.shape, heightCm: c.heightCm)
    expectEq(v.volumeCm3, c.expectVolumeCm3, "volume[\(c.shape)].cm3")
    expectEq(v.shape, c.expectShape, "volume.shape")
    expectEq(v.heightCm, c.expectEffectiveHeightCm, "volume.effectiveHeight")
}

for c in golden.densityLookup {
    expectOptional(Categories.densityFor(c.className), c.expectDensity, "densityFor(\(c.className))")
}

for c in golden.heightLookup {
    expectEq(Categories.heightFor(c.className), c.expectHeightCm, "heightFor(\(c.className))")
}

let densitySvc = DensityService()
for c in golden.densityResolve {
    let (rho, src) = densitySvc.resolve(makeFood(className: c.className, density: c.dbDensity))
    expectEq(rho, c.expectDensity, "densityResolve(\(c.className)).rho")
    expectEq(src.rawValue, c.expectSource, "densityResolve(\(c.className)).source")
}

for c in golden.nutrition {
    let n = NutritionResult.fromFood(makeFood(kcal: c.kcal100, protein: c.protein100, fat: c.fat100, carbs: c.carbs100), grams: c.grams)
    expectEq(n.calories, c.expectCalories, "nutrition.calories")
    expectEq(n.protein, c.expectProtein, "nutrition.protein")
    expectEq(n.fat, c.expectFat, "nutrition.fat")
    expectEq(n.carbs, c.expectCarbs, "nutrition.carbs")
}

do {
    let pr = PixelRatioStrategy()
    for c in golden.pixelRatio {
        let food = makeFood(portionG: c.portionG, refArea: c.refArea)
        let m = try pr.estimate(makeDetection(areaSq: c.maskAreaPx, areaFrame: c.maskAreaPx), food, MassContext(scale: nil))
        expectEq(m.grams, c.expectGrams, "pixelRatio.grams")
        expectEq(m.method, c.expectMethod, "pixelRatio.method")
    }

    let vs = VolumetricStrategy()
    for c in golden.volumetric {
        let se = ScaleEstimate(mmPerPx: c.mmPerPx, source: .depthIntrinsics, confidence: 0.9)
        let food = makeFood(className: c.className, density: c.dbDensity, geometricShape: c.geometricShape)
        let m = try vs.estimate(makeDetection(areaSq: c.maskAreaPx, areaFrame: c.maskAreaPxFrame), food, MassContext(scale: se))
        expectEq(m.grams, c.expectGrams, "volumetric(\(c.className)).grams", 1e-6)
        expectEq(m.volumeCm3!, c.expectVolumeCm3, "volumetric(\(c.className)).volume", 1e-6)
        expectEq(m.densityUsed!, c.expectDensityUsed, "volumetric(\(c.className)).density")
    }

    let auto = AutoStrategy()
    for c in golden.autoStrategy {
        let se = c.scaleConfidence.map { ScaleEstimate(mmPerPx: 0.5, source: .depthIntrinsics, confidence: $0) }
        let food = makeFood(className: "baklava", portionG: 150, refArea: 50_000, density: 1.2, geometricShape: "prism")
        let m = try auto.estimate(makeDetection(areaSq: 60_000, areaFrame: 90_000), food, MassContext(scale: se))
        expectEq(m.grams, c.expectGrams, "auto.grams", 1e-6)
        expectEq(m.method, c.expectMethod, "auto.method")
    }
} catch {
    FileHandle.standardError.write(Data("FATAL: strategy threw: \(error)\n".utf8))
    exit(2)
}

for c in golden.calibrationLevels {
    let est: ScaleEstimate?
    switch c.level {
    case "depth_intrinsics":
        est = IntrinsicDepthCalibration().estimateScale(FrameContext(intrinsics: intr, tiltDeg: 5.0, depthMm: c.depthMm))
    case "intrinsics_plane":
        est = IntrinsicPlaneCalibration(config: cfg).estimateScale(FrameContext(intrinsics: intr, tiltDeg: 10.0, depthMm: c.depthMm))
    case "static_default":
        est = StaticCalibration.fromConfig(cfg).estimateScale(FrameContext.simulatedTopdown())
    default:
        est = nil
    }
    if let e = est {
        expectEq(e.mmPerPx, c.expectMmPerPx, "calib[\(c.level)].mmPerPx")
        expectEq(e.source.rawValue, c.expectSource, "calib[\(c.level)].source")
        expectEq(e.confidence, c.expectConfidence, "calib[\(c.level)].confidence")
    } else {
        expect(false, "calib[\(c.level)] produced nil")
    }
}

for c in golden.assumedPlate {
    if let e = NaturalAnchorCalibration(config: cfg).estimateScale(FrameContext(frameSize: (c.frameW, c.frameH))) {
        expectEq(e.mmPerPx, c.expectMmPerPx, "assumedPlate.mmPerPx")
        expectEq(e.confidence, c.expectConfidence, "assumedPlate.confidence")
        expectEq(e.source.rawValue, c.expectSource, "assumedPlate.source")
    } else {
        expect(false, "assumedPlate produced nil")
    }
}

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
    if let best = resolver.estimateScale(ctx) {
        expectEq(best.source.rawValue, c.expectSource, "resolver[\(c.scenario)].source")
        expectEq(best.confidence, c.expectConfidence, "resolver[\(c.scenario)].confidence")
        expectEq(best.mmPerPx, c.expectMmPerPx, "resolver[\(c.scenario)].mmPerPx")
    } else {
        expect(false, "resolver[\(c.scenario)] produced nil")
    }
}

// MARK: - report

if failures.isEmpty {
    print("✅ LokmaParity: all \(checks) checks passed (Swift port == Python reference)")
    exit(0)
} else {
    print("❌ LokmaParity: \(failures.count)/\(checks) checks FAILED:")
    for f in failures { print("   - \(f)") }
    exit(1)
}

// MARK: - Decodable mirror of golden_vectors.json (keyDecodingStrategy = convertFromSnakeCase)

struct Golden: Decodable {
    let intrinsicsFromMm: [IntrinsicsCase]
    let areaPxToCm2: [AreaCase]
    let volume: [VolumeCase]
    let densityLookup: [DensityLookupCase]
    let heightLookup: [HeightLookupCase]
    let densityResolve: [DensityResolveCase]
    let nutrition: [NutritionCase]
    let pixelRatio: [PixelRatioCase]
    let volumetric: [VolumetricCase]
    let autoStrategy: [AutoCase]
    let calibrationLevels: [CalibrationCase]
    let assumedPlate: [AssumedPlateCase]
    let resolverSelection: [ResolverCase]

    struct IntrinsicsCase: Decodable {
        let focalMm, pitchMm: Double
        let nativeW, nativeH, workingW, workingH: Int
        let expectFocalPx, expectCx, expectCy: Double
    }
    struct AreaCase: Decodable { let mmPerPx, areaPx, expectMm2PerPx2, expectCm2: Double }
    struct VolumeCase: Decodable {
        let areaCm2: Double; let shape: String; let heightCm, expectVolumeCm3: Double
        let expectShape: String; let expectEffectiveHeightCm: Double
    }
    struct DensityLookupCase: Decodable { let className: String; let expectDensity: Double? }
    struct HeightLookupCase: Decodable { let className: String; let expectHeightCm: Double }
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
        let scaleConfidence: Double?; let threshold, expectGrams: Double; let expectMethod: String
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
}
