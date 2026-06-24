// FillDensityModel — runs the Phase-7 fill-density head on-device (Phase 8B).
//
// Loads FillHead.mlpackage (exported by `scripts/export_fill_head.py`) and, per
// detection, predicts D = ρ·(1−P) so the estimator can use mass = V·D (Stage A).
// Optional: if the model isn't bundled, `init?` returns nil and the pipeline keeps
// the analytic ρ·(1−P) path — no regression.
//
// FEATURE CONTRACT (must match training preprocessing in lokma/training/
// fill_regressor.py exactly — see FillHead.metadata.json):
//   • image   : a *masked* 96×96 RGB crop of the food (0–255; the /255 is baked into
//               the CoreML model). Crop = food bbox → resize 96 → × mask.
//   • scalars : [log1p(volume_cm3), coverage, mask_area_fraction].
//
// The CoreImage crop geometry (esp. the CIImage y-up flip) is the #1 thing to
// validate on-device — preview the masked crop and confirm it frames the food.
import CoreML
import CoreVideo
import Foundation
import LokmaCore

public final class FillDensityModel {
    public var imgSize: Int { cropper.imgSize }
    private let model: MLModel
    private let cropper = MaskedCropExtractor()

    public init?(bundle: Bundle = .main, resource: String = "FillHead") {
        guard let url = bundle.url(forResource: resource, withExtension: "mlmodelc") else { return nil }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        guard let m = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }
        self.model = m
    }

    /// Predict fill-density D for one detection, or nil (→ caller keeps the analytic
    /// ρ·(1−P) path) if features can't be built or the model errors.
    public func predict(capturedImage: CVPixelBuffer, raw: RawDetection,
                        volumeCm3: Double, coverage: Double) -> Double? {
        guard volumeCm3 > 0,
              let crop = cropper.maskedCrop(capturedImage, raw: raw),
              let scalars = try? MLMultiArray(shape: [1, 3], dataType: .float32) else { return nil }
        scalars[0] = NSNumber(value: Float(log(1.0 + volumeCm3)))
        scalars[1] = NSNumber(value: Float(coverage))
        scalars[2] = NSNumber(value: Float(cropper.maskAreaFraction(raw)))

        guard let provider = try? MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: crop),
            "scalars": MLFeatureValue(multiArray: scalars),
        ]),
        let out = try? model.prediction(from: provider),
        let d = out.featureValue(for: "fill_density")?.multiArrayValue, d.count > 0 else { return nil }
        return d[0].doubleValue
    }
}
