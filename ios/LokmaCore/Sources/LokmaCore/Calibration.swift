// Calibration — mirror of `lokma/geometry/calibration_service.py`.
//
// Per frame, each source returns a ScaleEstimate (mm per pixel) or nil; the
// resolver keeps the most confident. The image-dependent plate/utensil detector
// lives in the app layer (it needs Vision); it injects an `AnchorDetection` onto
// the FrameContext, which `NaturalAnchorCalibration` below consumes. The pure
// "assume a standard plate" fallback is kept here.
import Foundation

public protocol CalibrationService: Sendable {
    /// Returns a scale for this frame, or nil when the source does not apply.
    func estimateScale(_ context: FrameContext) -> ScaleEstimate?
}

/// Level 1: exact scale from device intrinsics + metric depth (ARKit/LiDAR).
public struct IntrinsicDepthCalibration: CalibrationService {
    public static let confidence = 0.90
    public init() {}

    public func estimateScale(_ context: FrameContext) -> ScaleEstimate? {
        guard let intr = context.intrinsics, intr.focalPx > 0 else { return nil }
        guard let depthMm = context.depthMm, depthMm > 0 else { return nil }
        return ScaleEstimate(
            mmPerPx: depthMm / intr.focalPx,
            source: .depthIntrinsics,
            confidence: Self.confidence,
            tiltDeg: context.tiltDeg,
            workingSize: intr.imageSize
        )
    }
}

/// Level 2: intrinsics + tilt + a plane distance (AR plane tracking, no LiDAR).
public struct IntrinsicPlaneCalibration: CalibrationService {
    public static let confidence = 0.70
    public let config: AppConfig
    public init(config: AppConfig) { self.config = config }

    public func estimateScale(_ context: FrameContext) -> ScaleEstimate? {
        guard let intr = context.intrinsics, intr.focalPx > 0, context.tiltDeg != nil else { return nil }
        let distanceMm = context.depthMm ?? config.defaultDistanceMm
        return ScaleEstimate(
            mmPerPx: distanceMm / intr.focalPx,
            source: .intrinsicsPlane,
            confidence: Self.confidence,
            tiltDeg: context.tiltDeg,
            workingSize: intr.imageSize
        )
    }
}

/// Level 3: a plate/utensil of known real size detected in the RGB frame.
/// The detection itself is done in the app (Vision) and attached to the context;
/// the low-confidence "assume a standard plate" branch is pure and lives here.
public struct NaturalAnchorCalibration: CalibrationService {
    public let config: AppConfig
    public init(config: AppConfig) { self.config = config }

    public func estimateScale(_ context: FrameContext) -> ScaleEstimate? {
        if let anchor = context.anchor, anchor.pixelSize > 0 {
            return ScaleEstimate(
                mmPerPx: anchor.assumedRealMm / anchor.pixelSize,
                source: .naturalAnchor,
                confidence: anchor.confidence,
                tiltDeg: anchor.tiltDeg
            )
        }
        // Last-ditch: assume a plate spanning a fixed fraction of the frame.
        if config.assumeDefaultPlate, let (w, h) = context.frameSize, w > 0, h > 0 {
            let pixelSize = config.assumedPlateFrameFraction * Double(max(w, h))
            if pixelSize > 0 {
                return ScaleEstimate(
                    mmPerPx: (config.defaultPlateDiameterCm * 10.0) / pixelSize,
                    source: .naturalAnchor,
                    confidence: 0.30,
                    tiltDeg: 0.0
                )
            }
        }
        return nil
    }
}

/// Level 4: assumed top-down + config focal length & distance (always returns).
public struct StaticCalibration: CalibrationService {
    public static let confidence = 0.20
    public let intrinsics: CameraIntrinsics
    public let distanceMm: Double

    public init(intrinsics: CameraIntrinsics, distanceMm: Double) {
        self.intrinsics = intrinsics
        self.distanceMm = distanceMm
    }

    public func estimateScale(_ context: FrameContext) -> ScaleEstimate? {
        guard intrinsics.focalPx > 0 else { return nil }
        return ScaleEstimate(
            mmPerPx: distanceMm / intrinsics.focalPx,
            source: .staticDefault,
            confidence: Self.confidence,
            tiltDeg: 0.0,
            workingSize: intrinsics.imageSize
        )
    }

    public static func fromConfig(_ config: AppConfig) -> StaticCalibration {
        let intrinsics = CameraIntrinsics.fromMM(
            focalLengthMm: config.focalLengthMm,
            pixelPitchMm: config.pixelPitchMm,
            nativeResolution: config.nativeResolution,
            workingResolution: config.workingResolution
        )
        return StaticCalibration(intrinsics: intrinsics, distanceMm: config.defaultDistanceMm)
    }
}

/// Tries calibration sources and returns the most confident valid ScaleEstimate.
public struct CalibrationResolver: Sendable {
    public let services: [CalibrationService]

    public init(_ services: [CalibrationService]) {
        precondition(!services.isEmpty, "CalibrationResolver needs at least one service")
        self.services = services
    }

    public func estimateScale(_ context: FrameContext) -> ScaleEstimate? {
        var best: ScaleEstimate?
        for service in services {
            guard let estimate = service.estimateScale(context) else { continue }
            if best == nil || estimate.confidence > best!.confidence {
                best = estimate
            }
        }
        return best
    }

    /// The default device chain: depth → plane → natural anchor → static.
    public static func makeDefault(config: AppConfig) -> CalibrationResolver {
        CalibrationResolver([
            IntrinsicDepthCalibration(),
            IntrinsicPlaneCalibration(config: config),
            NaturalAnchorCalibration(config: config),
            StaticCalibration.fromConfig(config),
        ])
    }
}
