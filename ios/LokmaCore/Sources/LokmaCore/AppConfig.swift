// AppConfig — the on-device subset of `lokma/config.py:AppConfig`.
//
// Only the constants the runtime math consumes are mirrored here. Values must
// stay in lockstep with the Python defaults; the parity fixture's `_meta.config`
// block records the source-of-truth values and `ParityTests` asserts they match.
import Foundation

public struct AppConfig: Sendable {
    // Inference
    public var maskResolution: Int = 640
    public var maskThreshold: Double = 0.5
    public var confThreshold: Double = 0.7
    public var defaultRefArea: Double = 40_000.0
    public var strategy: String = "auto"

    // Calibration (pinhole) defaults
    public var focalLengthMm: Double = 4.2
    public var pixelPitchMm: Double = 0.0012          // ~1.2 µm smartphone sensor
    public var nativeResolution: (Int, Int) = (4032, 3024)
    public var defaultDistanceMm: Double = 300.0
    public var defaultHeightCm: Double = 2.5

    // Frictionless calibration (Phase 2)
    public var defaultPlateDiameterCm: Double = 27.0
    public var calibrationConfidenceThreshold: Double = 0.5
    public var assumeDefaultPlate: Bool = true
    public var assumedPlateFrameFraction: Double = 0.70

    // Global V→mass scale, fit on-device against weighed references (Phase 8). 1.0 = no-op.
    public var massCalibrationConstant: Double = 1.0
    /// Suppress a detection's mass when it exceeds this (g) — a safety cap so false
    /// positives / runaway volumes show no calories instead of an impossible number.
    public var maxPlausibleGrams: Double = 2500.0

    // Active visual model version (keys the bundled DB's class_map).
    public var modelVersion: String = "foodyolo_v1"

    public init() {}

    /// Square working resolution at which mask areas are measured.
    public var workingResolution: (Int, Int) { (maskResolution, maskResolution) }
}
