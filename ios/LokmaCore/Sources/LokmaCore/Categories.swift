// Categories — mirror of `lokma/density/categories.py`.
//
// Shared category→density and class→shape/height maps (the Level-3 density data
// and the per-class height priors the volume engine uses). Kept verbatim so the
// build-time and run-time defaults never diverge from Python.
import Foundation

public enum Categories {
    /// Level-4 terminal fallback density.
    public static let waterDensity: Double = 1.0
    public static let defaultHeightCm: Double = 2.5

    /// Food-101 class -> coarse density category.
    public static let classCategory: [String: String] = [
        "apple_pie": "pastry",
        "bread_pudding": "pastry",
        "baklava": "syrup_pastry",
        "baby_back_ribs": "meat",
        "beef_carpaccio": "meat",
        "beef_tartare": "meat",
        "beet_salad": "salad",
        "beignets": "fried_dough",
        "bibimbap": "rice_bowl",
        "breakfast_burrito": "wrap",
    ]

    /// Category -> representative density (g/cm³).
    public static let categoryDensity: [String: Double] = [
        "pastry": 0.60,
        "syrup_pastry": 1.20,
        "meat": 1.05,
        "salad": 0.50,
        "fried_dough": 0.35,
        "rice_bowl": 0.85,
        "wrap": 0.90,
        // Turkish + everyday-essentials categories
        "flatbread": 0.55,
        "doner_meat": 1.05,
        "porous_dough": 0.35,
        "stew": 1.00,
        "soup": 1.00,
        "rice": 0.85,
        "egg_dish": 0.90,
        "poultry": 1.05,
        "fruit": 0.94,
        "dairy": 1.03,
        "grain": 1.00,
        "bread": 0.30,
    ]

    /// Class -> geometric shape prior used by the volume engine.
    public static let classShape: [String: String] = [
        "apple_pie": "prism",
        "bread_pudding": "prism",
        "baklava": "prism",
        "baby_back_ribs": "prism",
        "beef_carpaccio": "flat",
        "beef_tartare": "paraboloid",
        "beet_salad": "paraboloid",
        "beignets": "prism",
        "bibimbap": "cylinder",
        "breakfast_burrito": "cylinder",
    ]

    /// Category -> representative plated height (cm). Height dominates volume
    /// error once the footprint is metric, so it is a per-class prior.
    public static let categoryHeightCm: [String: Double] = [
        "pastry": 3.5,
        "syrup_pastry": 4.0,
        "meat": 3.0,
        "salad": 4.0,
        "fried_dough": 4.0,
        "rice_bowl": 5.0,
        "wrap": 5.0,
        // Turkish + everyday-essentials categories
        "flatbread": 0.8,
        "doner_meat": 6.0,
        "porous_dough": 3.0,
        "stew": 4.0,
        "soup": 4.0,
        "rice": 4.0,
        "egg_dish": 2.5,
        "poultry": 2.5,
        "fruit": 3.5,
        "dairy": 3.0,
        "grain": 2.5,
        "bread": 4.0,
    ]

    private static func norm(_ className: String) -> String {
        className.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Category-average density for a class, or `nil` if unknown.
    public static func densityFor(_ className: String) -> Double? {
        guard let category = classCategory[norm(className)] else { return nil }
        return categoryDensity[category]
    }

    /// Geometric-shape prior for a class, or `nil` if unknown.
    public static func shapeFor(_ className: String) -> String? {
        classShape[norm(className)]
    }

    /// Category-average plated height (cm); falls back to a global default.
    public static func heightFor(_ className: String) -> Double {
        if let category = classCategory[norm(className)], let h = categoryHeightCm[category] {
            return h
        }
        return defaultHeightCm
    }
}
