// YOLOSegDecoder — decode raw YOLOv11-seg CoreML outputs into RawDetections.
//
// ⚠️ EXPORT-DEPENDENT. Written for the raw (`--no-nms`) ultralytics layout:
//     output0 [1, 4+nc+32, N]  : box(cx,cy,w,h in input px) + class scores + 32 coeffs
//     output1 [1, 32, mh, mw]  : prototype masks
// After your first `scripts/export_coreml.py`, confirm the feature names + shapes
// (blueprint verification step 1) and set FoodSegModel.outputKey0/1 if needed.
// Assumes Float32 multiarrays (export with `--no-half` while validating).
import CoreML
import Foundation

struct YOLOSegDecoder {
    let metadata: FoodSegMetadata
    let scoreThreshold: Double
    let iouThreshold: Double
    let outputKey0: String?
    let outputKey1: String?

    func decode(_ out: MLFeatureProvider, inputSize: Int, frameSize: (Int, Int)) -> [RawDetection] {
        guard let (dets, proto) = resolveOutputs(out) else { return [] }
        guard dets.dataType == .float32, proto.dataType == .float32 else { return [] }

        // dets: [1, C, N]; proto: [1, 32, mh, mw]
        let C = dets.shape[1].intValue
        let N = dets.shape[2].intValue
        let coeffCount = 32
        let nc = C - 4 - coeffCount
        guard nc > 0 else { return [] }

        let mh = proto.shape[2].intValue, mw = proto.shape[3].intValue
        let dp = dets.dataPointer.assumingMemoryBound(to: Float.self)
        let ds = dets.strides.map { $0.intValue }       // [s0, sC, sN]
        @inline(__always) func d(_ c: Int, _ i: Int) -> Float { dp[ds[1] * c + ds[2] * i] }

        // 1. Per-anchor best class above threshold.
        var cand: [(box: (Double, Double, Double, Double), score: Double, cls: Int, coeffs: [Float])] = []
        let scaleX = Double(frameSize.0) / Double(inputSize)
        let scaleY = Double(frameSize.1) / Double(inputSize)
        for i in 0..<N {
            var bestId = 0, bestScore: Float = 0
            for k in 0..<nc {
                let s = d(4 + k, i)
                if s > bestScore { bestScore = s; bestId = k }
            }
            guard Double(bestScore) >= scoreThreshold else { continue }
            let cx = Double(d(0, i)), cy = Double(d(1, i)), w = Double(d(2, i)), h = Double(d(3, i))
            let x1 = (cx - w / 2) * scaleX, y1 = (cy - h / 2) * scaleY
            let x2 = (cx + w / 2) * scaleX, y2 = (cy + h / 2) * scaleY
            var coeffs = [Float](repeating: 0, count: coeffCount)
            for k in 0..<coeffCount { coeffs[k] = d(4 + nc + k, i) }
            cand.append(((x1, y1, x2, y2), Double(bestScore), bestId, coeffs))
        }

        // 2. Greedy NMS (global).
        let kept = nms(cand.map { ($0.box, $0.score) }, iou: iouThreshold).map { cand[$0] }

        // 3. Assemble per-instance masks from the prototypes.
        let pp = proto.dataPointer.assumingMemoryBound(to: Float.self)
        let ps = proto.strides.map { $0.intValue }      // [s0, sK, sY, sX]
        @inline(__always) func p(_ k: Int, _ y: Int, _ x: Int) -> Float { pp[ps[1] * k + ps[2] * y + ps[3] * x] }

        var results: [RawDetection] = []
        for c in kept {
            var mask = [Float](repeating: 0, count: mh * mw)
            for y in 0..<mh {
                for x in 0..<mw {
                    var acc: Float = 0
                    for k in 0..<coeffCount { acc += c.coeffs[k] * p(k, y, x) }
                    mask[y * mw + x] = sigmoid(acc)
                }
            }
            zeroOutsideBox(&mask, mw: mw, mh: mh, box: c.box, frameSize: frameSize)
            results.append(RawDetection(
                classId: c.cls, className: metadata.name(for: c.cls), confidence: c.score,
                bbox: c.box, mask: mask, maskWidth: mw, maskHeight: mh
            ))
        }
        return results
    }

    // MARK: - helpers

    private func resolveOutputs(_ out: MLFeatureProvider) -> (MLMultiArray, MLMultiArray)? {
        var proto: MLMultiArray?
        var dets: MLMultiArray?
        if let k0 = outputKey0, let a = out.featureValue(for: k0)?.multiArrayValue { dets = a }
        if let k1 = outputKey1, let a = out.featureValue(for: k1)?.multiArrayValue { proto = a }
        if dets == nil || proto == nil {
            for name in out.featureNames {
                guard let a = out.featureValue(for: name)?.multiArrayValue else { continue }
                if a.shape.count == 4, proto == nil { proto = a }          // [1,32,mh,mw]
                else if a.shape.count == 3, dets == nil { dets = a }        // [1,C,N]
            }
        }
        if let d = dets, let p = proto { return (d, p) }
        return nil
    }

    private func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }

    private func zeroOutsideBox(_ mask: inout [Float], mw: Int, mh: Int,
                                box: (Double, Double, Double, Double), frameSize: (Int, Int)) {
        // Box is in frame px; the proto grid spans the input image, so map the box
        // back to [0,1] of the frame and onto the mask grid.
        let bx0 = max(0, Int((box.0 / Double(frameSize.0)) * Double(mw)))
        let by0 = max(0, Int((box.1 / Double(frameSize.1)) * Double(mh)))
        let bx1 = min(mw, Int((box.2 / Double(frameSize.0)) * Double(mw)))
        let by1 = min(mh, Int((box.3 / Double(frameSize.1)) * Double(mh)))
        for y in 0..<mh {
            for x in 0..<mw where !(x >= bx0 && x < bx1 && y >= by0 && y < by1) {
                mask[y * mw + x] = 0
            }
        }
    }

    private func nms(_ boxes: [((Double, Double, Double, Double), Double)], iou threshold: Double) -> [Int] {
        let order = boxes.indices.sorted { boxes[$0].1 > boxes[$1].1 }
        var keep: [Int] = []
        var removed = Set<Int>()
        for i in order where !removed.contains(i) {
            keep.append(i)
            for j in order where j != i && !removed.contains(j) {
                if iouOf(boxes[i].0, boxes[j].0) > threshold { removed.insert(j) }
            }
        }
        return keep
    }

    private func iouOf(_ a: (Double, Double, Double, Double), _ b: (Double, Double, Double, Double)) -> Double {
        let x1 = max(a.0, b.0), y1 = max(a.1, b.1), x2 = min(a.2, b.2), y2 = min(a.3, b.3)
        let inter = max(0, x2 - x1) * max(0, y2 - y1)
        let areaA = max(0, a.2 - a.0) * max(0, a.3 - a.1)
        let areaB = max(0, b.2 - b.0) * max(0, b.3 - b.1)
        let union = areaA + areaB - inter
        return union > 0 ? inter / union : 0
    }
}
