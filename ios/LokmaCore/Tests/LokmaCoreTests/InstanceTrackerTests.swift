// InstanceTrackerTests — per-instance spatial smoothing (Phase 6, not parity).
import XCTest
@testable import LokmaCore

final class InstanceTrackerTests: XCTestCase {
    private func frame(_ k: Int) -> Date { Date(timeIntervalSince1970: Double(k)) }

    func testTwoSameClassPlatesTrackedIndependently() {
        let tracker = InstanceTracker()
        let a = (0.0, 0.0, 0.30)        // plate A
        let b = (0.40, 0.0, 0.30)       // plate B, 40 cm away
        var lastA = 0.0, lastB = 0.0
        for k in 0..<5 {
            lastA = tracker.smooth(className: "baklava", grams: 100, position: a, now: frame(k))
            lastB = tracker.smooth(className: "baklava", grams: 200, position: b, now: frame(k))
        }
        XCTAssertEqual(lastA, 100, accuracy: 1e-9)   // not contaminated by B's 200
        XCTAssertEqual(lastB, 200, accuracy: 1e-9)
    }

    func testJitteringSinglePlateStaysOneTrackAndRejectsSpike() {
        let tracker = InstanceTracker()
        let grams = [100.0, 100.0, 500.0, 100.0, 100.0]   // a single-frame spike
        var out = 0.0
        for (k, g) in grams.enumerated() {
            let jitter = (Double(k) * 0.01, 0.0, 0.30)     // ≤ 4 cm drift, under the 6 cm gate
            out = tracker.smooth(className: "baklava", grams: g, position: jitter, now: frame(k))
        }
        XCTAssertEqual(out, 100, accuracy: 1e-9)            // median rejects the spike → one track
    }

    func testStaleTrackEvicted() {
        let tracker = InstanceTracker(staleAfter: 1.0)
        let p = (0.0, 0.0, 0.30)
        _ = tracker.smooth(className: "baklava", grams: 100, position: p, now: frame(0))
        let out = tracker.smooth(className: "baklava", grams: 250, position: p, now: frame(10))
        XCTAssertEqual(out, 250, accuracy: 1e-9)            // old track gone → fresh window
    }

    func testNilPositionClassBucketFallback() {
        let tracker = InstanceTracker()
        var out = 0.0
        for k in 0..<3 {
            out = tracker.smooth(className: "doner", grams: 150, position: nil, now: frame(k))
        }
        XCTAssertEqual(out, 150, accuracy: 1e-9)
    }

    func testCameraCentroidOfFlatBox() throws {
        let W = 24, H = 18, fx = 30.0, fy = 30.0, cx = 12.0, cy = 9.0, z = 280.0
        var depth = [Double](repeating: 0, count: W * H)   // 0 = invalid outside the food
        var mask = [Double](repeating: 0, count: W * H)
        for v in 6...11 { for u in 8...15 { depth[v * W + u] = z; mask[v * W + u] = 1 } }
        let sample = DepthSample(depthMm: depth, mask: mask, width: W, height: H,
                                 fx: fx, fy: fy, cx: cx, cy: cy)
        let c = try XCTUnwrap(sample.cameraCentroidMm())
        XCTAssertEqual(c.x, (11.5 - cx) / fx * z, accuracy: 1e-9)   // mean u over [8,15] = 11.5
        XCTAssertEqual(c.y, (8.5 - cy) / fy * z, accuracy: 1e-9)    // mean v over [6,11] = 8.5
        XCTAssertEqual(c.z, z, accuracy: 1e-9)
    }

    func testCameraCentroidNilWithoutFood() {
        let sample = DepthSample(depthMm: [0, 0, 0, 0], mask: [0, 0, 0, 0], width: 2, height: 2,
                                 fx: 30, fy: 30, cx: 1, cy: 1)
        XCTAssertNil(sample.cameraCentroidMm())
    }
}
