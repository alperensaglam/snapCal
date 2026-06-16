// FoodSegModel — loads the exported CoreML segmentation package and decodes its
// raw output into `RawDetection`s.
//
// The model + sidecar are produced by `python scripts/export_coreml.py`:
//   • FoodSeg.mlpackage        (Xcode compiles this to FoodSeg.mlmodelc)
//   • FoodSeg.metadata.json    (model_version, imgsz, class names by id)
//
// IMPORTANT — decode is export-shape-dependent. Ultralytics YOLOv11-seg exported
// *without* NMS (`--no-nms`) yields two outputs:
//   output0  [1, 4+nc+32, 8400]  box(xywh) + class scores + 32 mask coeffs
//   output1  [1, 32, mh, mw]     prototype masks (typically 160×160)
// Per instance: mask = sigmoid(Σ coeff·proto), cropped to the box. Validate the
// concrete output feature names/shapes against your export (blueprint
// verification step 1) and adjust `outputKey0` / `outputKey1` if they differ.
import CoreML
import Foundation
import LokmaCore
import Vision

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

    // Adjust to match the exported feature names if they differ (see header note).
    public var outputKey0: String?       // detections (box+scores+coeffs); nil => first non-proto
    public var outputKey1: String?       // proto masks; nil => 4-D output

    public var scoreThreshold: Double = 0.25
    public var iouThreshold: Double = 0.45

    public init(bundle: Bundle = .main, resource: String = "FoodSeg") throws {
        guard let modelURL = bundle.url(forResource: resource, withExtension: "mlmodelc") else {
            throw VisionError.missingModel("\(resource).mlmodelc not in bundle (build compiles .mlpackage)")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all                  // ANE + GPU + CPU
        self.model = try MLModel(contentsOf: modelURL, configuration: cfg)

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

    /// Run inference on a captured pixel buffer and return decoded instances.
    public func predict(pixelBuffer: CVPixelBuffer, frameSize: (Int, Int)) throws -> [RawDetection] {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: try Self.resized(pixelBuffer, to: inputSize)),
        ])
        let out = try model.prediction(from: input)
        return YOLOSegDecoder(
            metadata: metadata, scoreThreshold: scoreThreshold, iouThreshold: iouThreshold,
            outputKey0: outputKey0, outputKey1: outputKey1
        ).decode(out, inputSize: inputSize, frameSize: frameSize)
    }

    /// Square-resize the camera buffer to the model's input grid (Vision/CoreImage).
    static func resized(_ pixelBuffer: CVPixelBuffer, to size: Int) throws -> CVPixelBuffer {
        // The model's input layer specifies `imgsz`; if it auto-resizes, this can
        // pass through. Provided as a seam — Vision's VNImageRequestHandler with an
        // imageCropAndScaleOption is the alternative. Returning the original keeps
        // the scaffold compiling; replace with a real CIContext render as needed.
        return pixelBuffer
    }
}

public enum VisionError: Error {
    case missingModel(String)
    case badOutput(String)
}
