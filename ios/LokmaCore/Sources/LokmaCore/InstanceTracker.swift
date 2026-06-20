// InstanceTracker — per-instance temporal smoothing of estimated mass (Phase 6).
//
// Replaces the class-keyed `MassStabilizer`: each detection is matched to a
// persistent spatial track by its world centroid, so two distinct plates of the
// same dish smooth independently (no cross-contamination). Pure and deterministic
// (no ARKit/CoreVideo) so it's unit-tested here; the app supplies world positions
// from the LiDAR depth + camera pose. Presentation stability only — not parity-locked.
//
// A stationary plate keeps a stable ARKit world centroid that re-matches frame to
// frame; two plates have centroids far apart → separate tracks. `position == nil`
// (non-LiDAR / failed depth) degrades to a single class bucket (old behavior).
import Foundation

public final class InstanceTracker {
    private struct Track {
        let id: Int
        let className: String
        var position: (Double, Double, Double)?
        var window: [Double]
        var lastSeen: Date
    }

    private let windowSize: Int
    private let staleAfter: TimeInterval
    private let maxMatchDistanceM: Double
    private var tracks: [Track] = []
    private var nextId = 0

    public init(windowSize: Int = 5, staleAfter: TimeInterval = 1.5, maxMatchDistanceM: Double = 0.06) {
        self.windowSize = max(1, windowSize)
        self.staleAfter = staleAfter
        self.maxMatchDistanceM = maxMatchDistanceM
    }

    /// Smooth one detection's grams within its own spatial track and return the
    /// windowed median. `position` is the world centroid in metres (nil → class
    /// bucket). Pass **one `now` per frame** so two same-class plates can't share a track.
    public func smooth(className: String, grams: Double, position: (Double, Double, Double)?, now: Date = Date()) -> Double {
        evictStale(now: now)
        let idx = matchOrCreate(className: className, position: position, now: now)
        tracks[idx].window.append(grams)
        if tracks[idx].window.count > windowSize {
            tracks[idx].window.removeFirst(tracks[idx].window.count - windowSize)
        }
        if position != nil { tracks[idx].position = position }
        tracks[idx].lastSeen = now
        let sorted = tracks[idx].window.sorted()
        return sorted[sorted.count / 2]
    }

    public func reset() {
        tracks.removeAll()
        nextId = 0
    }

    // MARK: - private

    private func matchOrCreate(className: String, position: (Double, Double, Double)?, now: Date) -> Int {
        guard let p = position else {
            // Degenerate fallback: a single class bucket (old MassStabilizer behavior).
            if let i = tracks.firstIndex(where: { $0.className == className && $0.position == nil }) { return i }
            return appendTrack(className: className, position: nil, now: now)
        }
        // Spatial match: nearest same-class track not yet matched this frame, within range.
        var bestIdx: Int?
        var bestDist = Double.greatestFiniteMagnitude
        for (i, t) in tracks.enumerated() {
            guard t.className == className, t.lastSeen != now, let q = t.position else { continue }
            let d = distance(p, q)
            if d <= maxMatchDistanceM, d < bestDist { bestDist = d; bestIdx = i }
        }
        return bestIdx ?? appendTrack(className: className, position: p, now: now)
    }

    private func appendTrack(className: String, position: (Double, Double, Double)?, now: Date) -> Int {
        tracks.append(Track(id: nextId, className: className, position: position, window: [], lastSeen: now))
        nextId += 1
        return tracks.count - 1
    }

    private func distance(_ a: (Double, Double, Double), _ b: (Double, Double, Double)) -> Double {
        let dx = a.0 - b.0, dy = a.1 - b.1, dz = a.2 - b.2
        return (dx * dx + dy * dy + dz * dz).squareRoot()
    }

    private func evictStale(now: Date) {
        tracks.removeAll { now.timeIntervalSince($0.lastSeen) > staleAfter }
    }
}
