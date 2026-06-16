// swift-tools-version:5.9
import PackageDescription

// LokmaCore — the platform-agnostic Swift port of LOKMA's estimation math.
//
// It contains NO Apple-framework dependencies (no ARKit/CoreML/Vision), so it
// builds and `swift test`s on macOS and is consumed by the iOS app as a local
// package. The Python `lokma/` package remains the source of truth; the test
// target asserts parity against golden vectors emitted by
// `scripts/dump_golden_vectors.py`.
let package = Package(
    name: "LokmaCore",
    platforms: [.macOS(.v12), .iOS(.v16)],
    products: [
        .library(name: "LokmaCore", targets: ["LokmaCore"]),
        .executable(name: "LokmaParity", targets: ["LokmaParity"]),
    ],
    targets: [
        .target(name: "LokmaCore"),
        // XCTest parity gate for Xcode/CI (needs full Xcode for the XCTest SDK).
        .testTarget(
            name: "LokmaCoreTests",
            dependencies: ["LokmaCore"],
            resources: [.copy("Fixtures")]
        ),
        // Same checks as a plain executable so parity is verifiable with only the
        // Command Line Tools (no XCTest): `swift run LokmaParity`.
        .executableTarget(name: "LokmaParity", dependencies: ["LokmaCore"]),
    ]
)
