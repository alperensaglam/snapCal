// VolumeModel — runs the Phase-9 non-LiDAR volume head on-device.
//
// Loads HeightHead.mlpackage (exported by `scripts/export_height_head.py`) and, per
// detection, predicts the food's volume V (cm³) from the masked RGB crop alone — the
// learned replacement for the static area×height prior when no LiDAR depth exists.
// Optional: if the model isn't bundled, `init?` returns nil and the pipeline keeps its
// existing scalar/pixel-ratio fallback — no regression.
//
// FEATURE CONTRACT (must match training preprocessing in lokma/training/
// height_regressor.py exactly — see HeightHead.metadata.json):
//   • image   : a *masked* 96×96 RGB crop of the food (0–255; /255 baked into the model).
//   • scalars : [mask_area_fraction]   (RGB only — no depth).
//   • output  : volume_log1p; we return expm1(volume_log1p) → V in cm³.
//
// The crop geometry is shared with FillDensityModel via MaskedCropExtractor, so both
// heads see byte-identical inputs.
import CoreML
import CoreVideo
import Foundation
import LokmaCore

public final class VolumeModel {
    private let model: MLModel
    private let cropper = MaskedCropExtractor()

    public init?(bundle: Bundle = .main, resource: String = "HeightHead") {
        guard let url = bundle.url(forResource: resource, withExtension: "mlmodelc") else { return nil }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        guard let m = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }
        self.model = m
    }

    /// Predict volume V (cm³) for one detection from RGB alone, or nil (→ caller keeps
    /// the scalar/pixel-ratio fallback) if features can't be built or the model errors.
    public func predict(capturedImage: CVPixelBuffer, raw: RawDetection) -> Double? {
        guard let crop = cropper.maskedCrop(capturedImage, raw: raw),
              let scalars = try? MLMultiArray(shape: [1, 1], dataType: .float32) else { return nil }
        scalars[0] = NSNumber(value: Float(cropper.maskAreaFraction(raw)))

        guard let provider = try? MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: crop),
            "scalars": MLFeatureValue(multiArray: scalars),
        ]),
        let out = try? model.prediction(from: provider),
        let v = out.featureValue(for: "volume_log1p")?.multiArrayValue, v.count > 0 else { return nil }
        // The head regresses log1p(V) for numerical stability; invert to cm³.
        let volume = expm1(v[0].doubleValue)
        return volume > 0 ? volume : nil
    }
}
