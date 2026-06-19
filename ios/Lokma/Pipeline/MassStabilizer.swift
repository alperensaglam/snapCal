// MassStabilizer — temporal smoothing of the per-frame mass estimate.
//
// The volumetric estimate is noisy under hand motion: depth and camera pose
// jitter frame-to-frame, so the displayed grams flicker even when the food is
// still. We keep a short rolling window of recent grams *per food class* and
// report its median, which rejects single-frame spikes without the lag of a long
// EMA. This is presentation stability only — the LokmaCore math is untouched and
// the raw (unsmoothed) estimate still drives calibration/source reporting.
//
// Class-keyed matching is intentionally simple: with several instances of the
// same class it smooths them together (acceptable for a stabilizer). Proper
// per-instance tracking is a Tier-2 concern once depth-integrated volume lands.
import Foundation

public final class MassStabilizer {
    private let windowSize: Int
    private let staleAfter: TimeInterval
    private var history: [String: [Double]] = [:]
    private var lastSeen: [String: Date] = [:]

    public init(windowSize: Int = 5, staleAfter: TimeInterval = 1.5) {
        self.windowSize = max(1, windowSize)
        self.staleAfter = staleAfter
    }

    /// Append `grams` for `className` and return the windowed median. Classes not
    /// seen for `staleAfter` are dropped so a removed plate doesn't anchor the value.
    public func smooth(className: String, grams: Double, now: Date = Date()) -> Double {
        evictStale(now: now)
        var window = history[className] ?? []
        window.append(grams)
        if window.count > windowSize { window.removeFirst(window.count - windowSize) }
        history[className] = window
        lastSeen[className] = now

        let sorted = window.sorted()
        return sorted[sorted.count / 2]
    }

    public func reset() {
        history.removeAll()
        lastSeen.removeAll()
    }

    private func evictStale(now: Date) {
        for (key, seen) in lastSeen where now.timeIntervalSince(seen) > staleAfter {
            history[key] = nil
            lastSeen[key] = nil
        }
    }
}
