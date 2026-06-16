// DensityService — mirror of `lokma/density/density_service.py`.
//
// Resolves ρ (g/cm³) through a 4-level fallback: DB density → (USDA cup, offline,
// skipped at runtime) → category average → water.
import Foundation

public enum DensitySource: String, Sendable {
    case nutrition5k = "nutrition5k_depth"
    case database = "database"
    case usdaPortion = "usda_portion"
    case category = "category"
    case water = "water"
}

public struct DensityService: Sendable {
    /// Volume of one US legal cup in cm³ (Level-2 conversion).
    public static let usdaCupVolumeCm3: Double = 236.588

    public let waterDensity: Double

    public init(waterDensity: Double = Categories.waterDensity) {
        self.waterDensity = waterDensity
    }

    public func resolve(_ food: FoodRecord) -> (density: Double, source: DensitySource) {
        // Level 1: authoritative density stored in the DB.
        if let d = food.density, d > 0 {
            return (d, .database)
        }
        // Level 2 (USDA cup→gram) is offline-only; no cup volume on the runtime record.
        // Level 3: categorical average.
        if let rho = Categories.densityFor(food.className), rho > 0 {
            return (rho, .category)
        }
        // Level 4: water.
        return (waterDensity, .water)
    }

    /// Level-2 conversion: density (g/cm³) from a USDA cup portion weight.
    public static func densityFromUsdaCup(grams: Double, cups: Double = 1.0, cupVolumeCm3: Double = usdaCupVolumeCm3) -> Double {
        let volumeCm3 = cups * cupVolumeCm3
        precondition(volumeCm3 > 0, "cup volume must be positive")
        return grams / volumeCm3
    }
}
