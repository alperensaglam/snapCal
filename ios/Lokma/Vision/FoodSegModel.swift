// FoodSegModel — loads the exported CoreML segmentation package, preprocesses the
// camera frame, and decodes the raw output into `RawDetection`s.
//
// The model + sidecar are produced by `python scripts/export_coreml.py`:
//   • FoodSeg.mlpackage        (Xcode compiles this to FoodSeg.mlmodelc)
//   • FoodSeg.metadata.json    (model_version, imgsz, class names by id)
//
// Confirmed by verification Step 1 (real v1 export): the CoreML input is a *fixed*
// `image 640×640`; outputs are `[1, 46, 8400]` (box+scores+coeffs) and
// `[1, 32, 160, 160]` (proto), both Float32. The decoder resolves them by rank, so
// no `outputKey0/1` override is needed — but keep the seams in case a future export
// renames/reshapes them.
//
// Preprocessing: ARKit hands us a landscape buffer (e.g. 1920×1440). We center-crop
// it to a square (side = min(w,h)) and scale that to 640×640 through a reused,
// GPU-backed `CIContext` writing into a `CVPixelBufferPool` — no per-frame
// allocation, so it stays smooth at the capped capture cadence. Aspect-fill (not
// stretch, not letterbox) keeps the food undistorted for the CNN and matches the
// "point at your plate" capture UX.
//
// Because the model sees the square crop, the effective frame the detections live
// on is `(S, S)` with square pixels — that is what `predict` returns and what the
// decoder + `DetectionBuilder` use (uniform `S/640` scale-back, native-grid area).
// `ARCaptureController` scales the working-grid focal length by the same crop side,
// so capture and inference agree on one square frame.
import CoreImage
import CoreML
import CoreVideo
import Foundation

public struct FoodSegMetadata: Decodable {
    public let modelVersion: String
    public let imgsz: Int
    public let names: [String: String]   // "0":"apple_pie", ...

    enum CodingKeys: String, CodingKey { case modelVersion = "model_version", imgsz, names }

    public func name(for classId: Int) -> String { names[String(classId)] ?? "class_\(classId)" }
}

public final class FoodSegModel {
    public let metadata: FoodSegMetadata
    public let inputSize: Int
    private let model: MLModel
    private let ciContext: CIContext
    private var bufferPool: CVPixelBufferPool?

    // Adjust to match the exported feature names if they differ (see header note).
    public var outputKey0: String?       // detections (box+scores+coeffs); nil => first rank-3
    public var outputKey1: String?       // proto masks; nil => first rank-4

    public var scoreThreshold: Double = 0.25
    public var iouThreshold: Double = 0.45

    public init(bundle: Bundle = .main, resource: String = "FoodSeg") throws {
        guard let modelURL = bundle.url(forResource: resource, withExtension: "mlmodelc") else {
            throw VisionError.missingModel("\(resource).mlmodelc not in bundle (build compiles .mlpackage)")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all                  // ANE + GPU + CPU
        self.model = try MLModel(contentsOf: modelURL, configuration: cfg)
        // One reused GPU context for every frame's crop+scale render.
        self.ciContext = CIContext(options: [.useSoftwareRenderer: false])

        if let metaURL = bundle.url(forResource: resource, withExtension: "metadata.json"),
           let data = try? Data(contentsOf: metaURL),
           let meta = try? JSONDecoder().decode(FoodSegMetadata.self, from: data) {
            self.metadata = meta
        } else {
            // Fall back to names embedded in the .mlmodelc by ultralytics.
            self.metadata = FoodSegMetadata(modelVersion: "unknown", imgsz: 640, names: [:])
        }
        self.inputSize = metadata.imgsz
    }

    /// Run inference on a captured pixel buffer. Returns the decoded instances and the
    /// effective `(S, S)` square frame they are expressed in (`S = min(w, h)` of the
    /// input, the side of the center crop fed to the model).
    public func predict(pixelBuffer: CVPixelBuffer) throws -> (detections: [RawDetection], frameSize: (Int, Int)) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let square = try squareCropAndScale(pixelBuffer)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: square),
        ])
        let out = try model.prediction(from: input)
        let side = min(width, height)
        let frameSize = (side, side)
        let raws = YOLOSegDecoder(
            metadata: metadata, scoreThreshold: scoreThreshold, iouThreshold: iouThreshold,
            outputKey0: outputKey0, outputKey1: outputKey1
        ).decode(out, inputSize: inputSize, frameSize: frameSize)
        return (raws, frameSize)
    }

    /// Center-crop the camera buffer to a square and scale it to `inputSize`²,
    /// rendering through the reused GPU `CIContext` into a pooled BGRA buffer.
    private func squareCropAndScale(_ src: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(src)
        let height = CVPixelBufferGetHeight(src)
        let side = min(width, height)
        let originX = CGFloat((width - side) / 2)
        let originY = CGFloat((height - side) / 2)
        let scale = CGFloat(inputSize) / CGFloat(side)

        let image = CIImage(cvPixelBuffer: src)
            .cropped(to: CGRect(x: originX, y: originY, width: CGFloat(side), height: CGFloat(side)))
            .transformed(by: CGAffineTransform(translationX: -originX, y: -originY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        let dst = try pooledSquareBuffer()
        ciContext.render(
            image, to: dst,
            bounds: CGRect(x: 0, y: 0, width: inputSize, height: inputSize),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )
        return dst
    }

    /// Vend a reusable `inputSize`² BGRA buffer from a lazily-created pool.
    private func pooledSquareBuffer() throws -> CVPixelBuffer {
        if bufferPool == nil {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: inputSize,
                kCVPixelBufferHeightKey as String: inputSize,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),  // GPU/CoreML-friendly
            ]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool)
            bufferPool = pool
        }
        guard let pool = bufferPool else {
            throw VisionError.badOutput("failed to create pixel buffer pool")
        }
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        guard status == kCVReturnSuccess, let result = buffer else {
            throw VisionError.badOutput("pixel buffer allocation failed (status \(status))")
        }
        return result
    }
}

public enum VisionError: Error {
    case missingModel(String)
    case badOutput(String)
}
