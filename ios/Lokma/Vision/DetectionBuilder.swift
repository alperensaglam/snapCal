// DetectionBuilder — mirror of `InferencePipeline._build_detection` (lines 117-144).
//
// Turns a raw per-instance segmentation mask into a `LokmaCore.Detection` carrying
// the two thresholded pixel-area scalars the estimators consume:
//   * maskAreaPx       — area on the 640² working grid (pixel_ratio parity)
//   * maskAreaPxFrame  — area on the native frame grid (volumetric / scale parity)
//
// Python resizes the soft mask to each grid then thresholds at 0.5. We compute the
// thresholded coverage *fraction* on the mask's own grid and scale it to each
// target grid — an area-preserving equivalent that avoids a second resample.
// (Exact pixel-for-pixel agreement with cv2.resize is confirmed separately by the
// cross-impl device↔Python check, verification step 5.)
import Foundation
import LokmaCore

public struct RawDetection {
    public let classId: Int
    public let className: String
    public let confidence: Double
    public let bbox: (Double, Double, Double, Double)   // in frame pixels (x1,y1,x2,y2)
    /// Per-instance mask, row-major, values in [0,1], at `maskWidth`×`maskHeight`.
    public let mask: [Float]
    public let maskWidth: Int
    public let maskHeight: Int

    public init(classId: Int, className: String, confidence: Double,
                bbox: (Double, Double, Double, Double), mask: [Float], maskWidth: Int, maskHeight: Int) {
        self.classId = classId
        self.className = className
        self.confidence = confidence
        self.bbox = bbox
        self.mask = mask
        self.maskWidth = maskWidth
        self.maskHeight = maskHeight
    }
}

public enum DetectionBuilder {
    public static func makeDetection(
        from raw: RawDetection, frameSize: (Int, Int), config: AppConfig = AppConfig(),
        depthSample: DepthSample? = nil
    ) -> Detection {
        let threshold = Float(config.maskThreshold)
        let total = max(1, raw.maskWidth * raw.maskHeight)
        var above = 0
        for v in raw.mask where v > threshold { above += 1 }
        let coverage = Double(above) / Double(total)

        let work = config.workingResolution
        let areaSq = coverage * Double(work.0) * Double(work.1)
        let areaFrame = coverage * Double(frameSize.0) * Double(frameSize.1)

        return Detection(
            classId: raw.classId,
            className: raw.className,
            confidence: raw.confidence,
            bbox: raw.bbox,
            maskAreaPx: areaSq,
            maskAreaPxFrame: areaFrame,
            frameSize: frameSize,
            depthSample: depthSample
        )
    }
}
