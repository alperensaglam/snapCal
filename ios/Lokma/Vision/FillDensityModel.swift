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
import CoreImage
import CoreML
import CoreVideo
import Foundation
import LokmaCore

public final class FillDensityModel {
    public let imgSize = 96
    private let model: MLModel
    private let ciContext: CIContext

    public init?(bundle: Bundle = .main, resource: String = "FillHead") {
        guard let url = bundle.url(forResource: resource, withExtension: "mlmodelc") else { return nil }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        guard let m = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }
        self.model = m
        self.ciContext = CIContext(options: [.useSoftwareRenderer: false])
    }

    /// Predict fill-density D for one detection, or nil (→ caller keeps the analytic
    /// ρ·(1−P) path) if features can't be built or the model errors.
    public func predict(capturedImage: CVPixelBuffer, raw: RawDetection,
                        volumeCm3: Double, coverage: Double) -> Double? {
        guard volumeCm3 > 0,
              let crop = maskedCrop(capturedImage, raw: raw),
              let scalars = try? MLMultiArray(shape: [1, 3], dataType: .float32) else { return nil }
        scalars[0] = NSNumber(value: Float(log(1.0 + volumeCm3)))
        scalars[1] = NSNumber(value: Float(coverage))
        scalars[2] = NSNumber(value: Float(maskAreaFraction(raw)))

        guard let provider = try? MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: crop),
            "scalars": MLFeatureValue(multiArray: scalars),
        ]),
        let out = try? model.prediction(from: provider),
        let d = out.featureValue(for: "fill_density")?.multiArrayValue, d.count > 0 else { return nil }
        return d[0].doubleValue
    }

    // MARK: - feature extraction

    /// Thresholded coverage of the per-instance mask (on-device analog of the training
    /// `mask.mean()`; note training averages over the whole Nutrition5K frame, here
    /// over the 160-grid — an accepted bootstrap mismatch).
    private func maskAreaFraction(_ raw: RawDetection) -> Double {
        let total = max(1, raw.maskWidth * raw.maskHeight)
        var above = 0
        for v in raw.mask where v > 0.5 { above += 1 }
        return Double(above) / Double(total)
    }

    /// Build the masked 96×96 RGB crop. The mask (160-grid, in the S×S model frame) and
    /// the bbox (S×S frame px) are mapped into the captured-image buffer, the food bbox
    /// is cropped and scaled to 96, and the mask is multiplied in (background → black).
    private func maskedCrop(_ captured: CVPixelBuffer, raw: RawDetection) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(captured), h = CVPixelBufferGetHeight(captured)
        let s = CGFloat(min(w, h))
        let originX = CGFloat((w - Int(s)) / 2), originY = CGFloat((h - Int(s)) / 2)
        guard let maskBuf = maskBuffer(raw) else { return nil }

        // Mask 160-grid → S×S frame → captured pixels (top-left origin throughout).
        let maskScale = s / CGFloat(raw.maskWidth)            // 160 → S
        let base = CIImage(cvPixelBuffer: captured)
        let mask = CIImage(cvPixelBuffer: maskBuf)
            .transformed(by: CGAffineTransform(scaleX: maskScale, y: maskScale))
            .transformed(by: CGAffineTransform(translationX: originX, y: originY))

        // Multiply RGB by the mask (background → black), matching `rgb * mask`.
        guard let mult = CIFilter(name: "CIMultiplyCompositing") else { return nil }
        mult.setValue(mask, forKey: kCIInputImageKey)
        mult.setValue(base, forKey: kCIInputBackgroundImageKey)
        guard let masked = mult.outputImage else { return nil }

        // Food bbox (S×S frame) → captured pixels, then a top-left-origin crop rect.
        // NOTE: CIImage is y-up; flip the bbox vertically against the buffer height.
        let (x1, y1, x2, y2) = raw.bbox
        let cx0 = CGFloat(x1) + originX, cx1 = CGFloat(x2) + originX
        let cy0 = CGFloat(y1) + originY, cy1 = CGFloat(y2) + originY
        let cropW = max(1, cx1 - cx0), cropH = max(1, cy1 - cy0)
        let rect = CGRect(x: cx0, y: CGFloat(h) - cy1, width: cropW, height: cropH)   // y-flip
        let cropped = masked.cropped(to: rect)
            .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(imgSize) / cropW, y: CGFloat(imgSize) / cropH))

        var dst: CVPixelBuffer?
        CVPixelBufferCreate(nil, imgSize, imgSize, kCVPixelFormatType_32BGRA, nil, &dst)
        guard let out = dst else { return nil }
        ciContext.render(cropped, to: out,
                         bounds: CGRect(x: 0, y: 0, width: imgSize, height: imgSize),
                         colorSpace: CGColorSpaceCreateDeviceRGB())
        return out
    }

    /// OneComponent8 mask buffer from the 160-grid sigmoid mask.
    private func maskBuffer(_ raw: RawDetection) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, raw.maskWidth, raw.maskHeight,
                            kCVPixelFormatType_OneComponent8, nil, &pb)
        guard let buf = pb else { return nil }
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(buf)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<raw.maskHeight {
            for x in 0..<raw.maskWidth {
                let v = raw.mask[y * raw.maskWidth + x]
                ptr[y * stride + x] = UInt8(max(0, min(255, v * 255)))
            }
        }
        return buf
    }
}
