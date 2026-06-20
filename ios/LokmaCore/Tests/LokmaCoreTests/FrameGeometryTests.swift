// FrameGeometryTests — the pure capture coordinate map (iOS-specific, not parity).
import XCTest
@testable import LokmaCore

final class FrameGeometryTests: XCTestCase {
    // iPhone-ish: 1920×1440 capture, 256×192 LiDAR depth.
    private let geo = FrameGeometry(imageWidth: 1920, imageHeight: 1440,
                                    depthWidth: 256, depthHeight: 192,
                                    fx: 1500.0, fy: 1500.0, cx: 960.0, cy: 720.0)

    func testCropOffset() {
        XCTAssertEqual(geo.cropSide, 1440)
        XCTAssertEqual(geo.cropOriginX, 240.0, accuracy: 1e-9)   // (1920-1440)/2
        XCTAssertEqual(geo.cropOriginY, 0.0, accuracy: 1e-9)
    }

    func testDepthSpaceIntrinsicsScale() {
        let r = 256.0 / 1920.0
        XCTAssertEqual(geo.scaleX, r, accuracy: 1e-12)
        XCTAssertEqual(geo.scaleY, 192.0 / 1440.0, accuracy: 1e-12)
        let i = geo.depthIntrinsics
        XCTAssertEqual(i.fx, 1500.0 * r, accuracy: 1e-9)
        XCTAssertEqual(i.cx, 960.0 * r, accuracy: 1e-9)
        XCTAssertEqual(i.cy, 720.0 * (192.0 / 1440.0), accuracy: 1e-9)
    }

    func testDepthPixelInsideCropMapsBack() {
        let f = try? XCTUnwrap(geo.depthPixelToCropFrame(128, 96))
        XCTAssertNotNil(f)
        XCTAssertEqual(f!.x, 128.5 / geo.scaleX - 240.0, accuracy: 1e-9)
        XCTAssertEqual(f!.y, 96.5 / geo.scaleY, accuracy: 1e-9)
        XCTAssertTrue(f!.x >= 0 && f!.x < 1440 && f!.y >= 0 && f!.y < 1440)
    }

    func testDepthPixelOutsideCropIsNil() {
        // Far-left depth column sits in the cropped-out landscape margin.
        XCTAssertNil(geo.depthPixelToCropFrame(0, 96))
    }

    func testPortraitBufferCropOffset() {
        // A portrait-shaped buffer crops vertically instead; origin moves to Y.
        let p = FrameGeometry(imageWidth: 1440, imageHeight: 1920, depthWidth: 192, depthHeight: 256,
                              fx: 1500, fy: 1500, cx: 720, cy: 960)
        XCTAssertEqual(p.cropSide, 1440)
        XCTAssertEqual(p.cropOriginX, 0.0, accuracy: 1e-9)
        XCTAssertEqual(p.cropOriginY, 240.0, accuracy: 1e-9)
    }
}
