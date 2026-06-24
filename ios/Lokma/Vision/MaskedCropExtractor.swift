// MaskedCropExtractor — shared on-device preprocessing for the per-instance
// regression heads (fill-density + volume). It produces the two inputs both heads
// consume — a masked imgSize×imgSize RGB crop and the mask-area-fraction scalar —
// so their feature pipelines stay byte-for-byte identical (and match the training
// preprocessing in lokma/training/*_regressor.py). Extracted from FillDensityModel
// so the volume head can reuse the exact same crop geometry (esp. the CIImage y-up
// flip, the #1 thing to validate on-device).
import CoreImage
import CoreVideo
import Foundation
import LokmaCore

public struct MaskedCropExtractor {
    public let imgSize: Int
    private let ciContext: CIContext

    public init(imgSize: Int = 96) {
        self.imgSize = imgSize
        self.ciContext = CIContext(options: [.useSoftwareRenderer: false])
    }

    /// Thresholded coverage of the per-instance mask (on-device analog of the training
    /// `mask.mean()`; note training averages over the whole Nutrition5K frame, here
    /// over the 160-grid — an accepted bootstrap mismatch).
    public func maskAreaFraction(_ raw: RawDetection) -> Double {
        let total = max(1, raw.maskWidth * raw.maskHeight)
        var above = 0
        for v in raw.mask where v > 0.5 { above += 1 }
        return Double(above) / Double(total)
    }

    /// Build the masked imgSize×imgSize RGB crop. The mask (160-grid, in the S×S model
    /// frame) and the bbox (S×S frame px) are mapped into the captured-image buffer, the
    /// food bbox is cropped and scaled to imgSize, and the mask is multiplied in
    /// (background → black).
    public func maskedCrop(_ captured: CVPixelBuffer, raw: RawDetection) -> CVPixelBuffer? {
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
