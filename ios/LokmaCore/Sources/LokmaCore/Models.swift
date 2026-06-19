// Typed models — the Swift mirror of `lokma/core/models.py`.
//
// Pure value types. The raw segmentation mask (an image buffer) is intentionally
// *not* carried here; the app's DetectionBuilder reduces it to the two pixel-area
// scalars below before it crosses into LokmaCore.
import Foundation

// MARK: - Geometry / camera

public struct CameraIntrinsics: Sendable, Equatable {
    /// Focal length in *pixels of the working grid* (folds sensor pitch + resize).
    public let focalPx: Double
    public let imageSize: (Int, Int)
    public let principalPoint: (Double, Double)?

    public init(focalPx: Double, imageSize: (Int, Int), principalPoint: (Double, Double)? = nil) {
        self.focalPx = focalPx
        self.imageSize = imageSize
        self.principalPoint = principalPoint
    }

    public static func == (lhs: CameraIntrinsics, rhs: CameraIntrinsics) -> Bool {
        lhs.focalPx == rhs.focalPx && lhs.imageSize == rhs.imageSize
            && lhs.principalPoint?.0 == rhs.principalPoint?.0
            && lhs.principalPoint?.1 == rhs.principalPoint?.1
    }

    /// Derive working-grid intrinsics from physical sensor parameters.
    ///
    /// `focalPx(native) = focalMm / pitchMm`, then scaled by the native→working
    /// width ratio so the focal length matches the grid areas are counted on.
    /// Mirrors `CameraIntrinsics.from_mm` (models.py:33-57).
    public static func fromMM(
        focalLengthMm: Double,
        pixelPitchMm: Double,
        nativeResolution: (Int, Int),
        workingResolution: (Int, Int)
    ) -> CameraIntrinsics {
        precondition(pixelPitchMm > 0, "pixelPitchMm must be positive")
        precondition(nativeResolution.0 > 0, "nativeResolution width must be positive")
        let focalPxNative = focalLengthMm / pixelPitchMm
        let scale = Double(workingResolution.0) / Double(nativeResolution.0)
        let focalPx = focalPxNative * scale
        let cx = Double(workingResolution.0) / 2.0
        let cy = Double(workingResolution.1) / 2.0
        return CameraIntrinsics(focalPx: focalPx, imageSize: workingResolution, principalPoint: (cx, cy))
    }
}

public struct VolumeEstimate: Sendable, Equatable {
    public let volumeCm3: Double
    public let shape: String
    public let heightCm: Double
    public let realAreaCm2: Double
    public let method: String
}

// MARK: - Knowledge base

/// One row of the `nutrition` compatibility view, macros stored per 100 g.
public struct FoodRecord: Sendable, Equatable {
    public let className: String
    public let source: String?
    public let usdaDesc: String?
    public let caloriesPer100g: Double
    public let proteinPer100g: Double
    public let fatPer100g: Double
    public let carbsPer100g: Double
    public let portionG: Double
    public let refArea: Double?
    public let density: Double?
    public let geometricShape: String?

    public init(
        className: String, source: String? = nil, usdaDesc: String? = nil,
        caloriesPer100g: Double, proteinPer100g: Double, fatPer100g: Double, carbsPer100g: Double,
        portionG: Double, refArea: Double? = nil, density: Double? = nil, geometricShape: String? = nil
    ) {
        self.className = className
        self.source = source
        self.usdaDesc = usdaDesc
        self.caloriesPer100g = caloriesPer100g
        self.proteinPer100g = proteinPer100g
        self.fatPer100g = fatPer100g
        self.carbsPer100g = carbsPer100g
        self.portionG = portionG
        self.refArea = refArea
        self.density = density
        self.geometricShape = geometricShape
    }
}

// MARK: - Inference results

/// One segmented food instance, reduced to the two pixel-area scalars the
/// estimators consume (mirrors the area fields of `core/models.py:Detection`).
public struct Detection: Sendable, Equatable {
    public let classId: Int
    public let className: String
    public let confidence: Double
    public let bbox: (Double, Double, Double, Double)
    public let maskAreaPx: Double         // working (640²) grid — pixel_ratio parity
    public let maskAreaPxFrame: Double    // native frame grid — volumetric/scale
    public let frameSize: (Int, Int)

    public init(
        classId: Int, className: String, confidence: Double,
        bbox: (Double, Double, Double, Double) = (0, 0, 0, 0),
        maskAreaPx: Double, maskAreaPxFrame: Double = 0.0, frameSize: (Int, Int) = (0, 0)
    ) {
        self.classId = classId
        self.className = className
        self.confidence = confidence
        self.bbox = bbox
        self.maskAreaPx = maskAreaPx
        self.maskAreaPxFrame = maskAreaPxFrame
        self.frameSize = frameSize
    }

    public static func == (lhs: Detection, rhs: Detection) -> Bool {
        lhs.classId == rhs.classId && lhs.className == rhs.className
            && lhs.confidence == rhs.confidence && lhs.maskAreaPx == rhs.maskAreaPx
            && lhs.maskAreaPxFrame == rhs.maskAreaPxFrame && lhs.frameSize == rhs.frameSize
    }
}

public struct MassEstimate: Sendable, Equatable {
    public let grams: Double
    public let method: String
    public let densityUsed: Double?
    public let volumeCm3: Double?
    public let calibrationSource: String?
    public let calibrationConfidence: Double?

    public init(
        grams: Double, method: String, densityUsed: Double? = nil, volumeCm3: Double? = nil,
        calibrationSource: String? = nil, calibrationConfidence: Double? = nil
    ) {
        self.grams = grams
        self.method = method
        self.densityUsed = densityUsed
        self.volumeCm3 = volumeCm3
        self.calibrationSource = calibrationSource
        self.calibrationConfidence = calibrationConfidence
    }
}

public struct NutritionResult: Sendable, Equatable {
    public let calories: Double
    public let protein: Double
    public let fat: Double
    public let carbs: Double

    public init(calories: Double, protein: Double, fat: Double, carbs: Double) {
        self.calories = calories
        self.protein = protein
        self.fat = fat
        self.carbs = carbs
    }

    /// Scale per-100g macros to an estimated portion mass (models.py:151-159).
    public static func fromFood(_ food: FoodRecord, grams: Double) -> NutritionResult {
        let factor = grams / 100.0
        return NutritionResult(
            calories: food.caloriesPer100g * factor,
            protein: food.proteinPer100g * factor,
            fat: food.fatPer100g * factor,
            carbs: food.carbsPer100g * factor
        )
    }
}

public struct AnnotatedResult: Sendable {
    public let detection: Detection
    public let food: FoodRecord?
    public let mass: MassEstimate?
    public let nutrition: NutritionResult?

    public init(detection: Detection, food: FoodRecord? = nil, mass: MassEstimate? = nil, nutrition: NutritionResult? = nil) {
        self.detection = detection
        self.food = food
        self.mass = mass
        self.nutrition = nutrition
    }

    public var isIdentified: Bool { food != nil && mass != nil && nutrition != nil }

    public var label: String {
        let name = detection.className.uppercased()
        if isIdentified, let mass, let nutrition {
            return "\(name): ~\(Int(mass.grams.rounded()))g | \(Int(nutrition.calories.rounded())) kcal"
        }
        return "\(name): (no DB match)"
    }
}

// MARK: - Calibration

public enum CalibrationSource: String, Sendable {
    case depthIntrinsics = "depth_intrinsics"
    case intrinsicsPlane = "intrinsics_plane"
    case naturalAnchor = "natural_anchor"
    case staticDefault = "static_default"
}

/// A natural anchor (plate/utensil) of known real size found in the frame.
public struct AnchorDetection: Sendable {
    public let kind: String
    public let pixelSize: Double          // major axis (plate) / length (utensil), frame px
    public let assumedRealMm: Double
    public let confidence: Double
    public let tiltDeg: Double?

    public init(kind: String, pixelSize: Double, assumedRealMm: Double, confidence: Double, tiltDeg: Double? = nil) {
        self.kind = kind
        self.pixelSize = pixelSize
        self.assumedRealMm = assumedRealMm
        self.confidence = confidence
        self.tiltDeg = tiltDeg
    }
}

/// Per-frame calibration: how many millimetres a pixel spans.
public struct ScaleEstimate: Sendable, Equatable {
    public let mmPerPx: Double
    public let source: CalibrationSource
    public let confidence: Double
    public let tiltDeg: Double?
    public let workingSize: (Int, Int)?

    public init(mmPerPx: Double, source: CalibrationSource, confidence: Double, tiltDeg: Double? = nil, workingSize: (Int, Int)? = nil) {
        self.mmPerPx = mmPerPx
        self.source = source
        self.confidence = confidence
        self.tiltDeg = tiltDeg
        self.workingSize = workingSize
    }

    public var mm2PerPx2: Double { mmPerPx * mmPerPx }

    /// Beyond this incidence angle the 1/cos(θ) area correction is clamped and the
    /// volumetric path is gated off (AutoStrategy) — grazing views aren't trustworthy.
    public static let maxForeshorteningTiltDeg = 65.0

    /// Area foreshortening factor cos(θ) for a plane viewed at incidence θ. Unknown
    /// tilt → 1.0 (top-down); clamped so the reciprocal can't blow up at grazing
    /// angles. Mirrors `_foreshortening` (models.py).
    public static func foreshortening(_ tiltDeg: Double?) -> Double {
        guard let t = tiltDeg else { return 1.0 }
        let clamped = min(max(t, 0.0), maxForeshorteningTiltDeg)
        return max(cos(clamped * .pi / 180.0), 1e-3)
    }

    /// Metric footprint area from pixel area, tilt-corrected (models.py:area_px_to_cm2).
    /// Top-down scaling is exact only for a perpendicular optical axis; divide by
    /// cos(tilt) to undo plane foreshortening at oblique holding angles.
    public func areaPxToCm2(_ areaPx: Double) -> Double {
        let topdown = areaPx * mm2PerPx2 / 100.0
        return topdown / Self.foreshortening(tiltDeg)
    }

    public static func == (lhs: ScaleEstimate, rhs: ScaleEstimate) -> Bool {
        lhs.mmPerPx == rhs.mmPerPx && lhs.source == rhs.source && lhs.confidence == rhs.confidence
    }
}

/// Per-frame capture bundle. The ARKit/AVFoundation adapter (app layer) fills
/// these from `ARFrame`; LokmaCore's calibration consumes whatever is present.
/// Mirrors `FrameContext.from_device` — minus the raw image/depth buffers, which
/// the app reduces to `depthMm`/`frameSize` before constructing this.
public struct FrameContext: Sendable {
    public let intrinsics: CameraIntrinsics?
    public let tiltDeg: Double?
    public let gravity: (Double, Double, Double)?
    public let depthMm: Double?
    public let frameSize: (Int, Int)?
    public let anchor: AnchorDetection?

    public init(
        intrinsics: CameraIntrinsics? = nil, tiltDeg: Double? = nil,
        gravity: (Double, Double, Double)? = nil, depthMm: Double? = nil,
        frameSize: (Int, Int)? = nil, anchor: AnchorDetection? = nil
    ) {
        self.intrinsics = intrinsics
        self.tiltDeg = tiltDeg
        self.gravity = gravity
        self.depthMm = depthMm
        self.frameSize = frameSize
        self.anchor = anchor
    }

    /// Desktop/simulator analogue of `FrameContext.simulated_topdown`.
    public static func simulatedTopdown(frameSize: (Int, Int)? = nil) -> FrameContext {
        FrameContext(intrinsics: nil, tiltDeg: 0.0, frameSize: frameSize)
    }
}
