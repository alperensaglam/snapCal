// VolumeEngine — mirror of `lokma/geometry/volume_engine.py`.
//
// Pure, stateless geometry: a real-world footprint area extruded by an assumed
// height according to the food's shape prior.
import Foundation

public struct VolumeEngine: Sendable {
    /// Assumed thickness for a thin/flat layer (e.g. pizza, carpaccio), in cm.
    public static let flatLayerCm: Double = 0.8

    public init() {}

    public func estimateVolume(realAreaCm2: Double, shape: String, heightCm: Double) -> VolumeEstimate {
        let shapeKey = (shape.isEmpty ? "prism" : shape).lowercased()
        let volume: Double
        let effectiveHeight: Double

        switch shapeKey {
        case "prism", "cylinder":
            volume = realAreaCm2 * heightCm
            effectiveHeight = heightCm
        case "paraboloid":
            volume = 0.5 * realAreaCm2 * heightCm
            effectiveHeight = heightCm
        case "flat":
            volume = realAreaCm2 * Self.flatLayerCm
            effectiveHeight = Self.flatLayerCm
        default:  // unknown shape -> simple extrusion fallback
            volume = realAreaCm2 * heightCm
            effectiveHeight = heightCm
        }

        return VolumeEstimate(
            volumeCm3: volume, shape: shapeKey, heightCm: effectiveHeight,
            realAreaCm2: realAreaCm2, method: "geometric:\(shapeKey)"
        )
    }
}
