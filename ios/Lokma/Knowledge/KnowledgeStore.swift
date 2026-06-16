// KnowledgeStore — the iOS read side of the LOKMA knowledge base.
//
// Mirrors `lokma/knowledge/database_manager.py:DatabaseManager.load()`: opens the
// bundled, read-only `lokma_local.db` and loads every food (through the same
// `nutrition` compatibility view) into a `[className: FoodRecord]` cache, so
// per-frame lookups are O(1) and never touch SQL. The `nutrition` view already
// resolves `ref_area` against `kb_meta.active_model_version`, so swapping the
// bundled DB (or its meta value) is what flips v1 <-> v2.
import Foundation
import GRDB
import LokmaCore

public final class KnowledgeStore {
    /// Same projection as `schema.py:select_sql()` (the FoodRecord columns).
    private static let selectSQL = """
        SELECT class_name, source, usda_desc, calories, protein, fat, carbs,
               portion_g, ref_area, density, geometric_shape
        FROM nutrition
        """

    private let cache: [String: FoodRecord]
    public let activeModelVersion: String?

    public var count: Int { cache.count }

    public init(databaseURL: URL) throws {
        var config = Configuration()
        config.readonly = true
        let dbQueue = try DatabaseQueue(path: databaseURL.path, configuration: config)

        (cache, activeModelVersion) = try dbQueue.read { db in
            var map: [String: FoodRecord] = [:]
            for row in try Row.fetchAll(db, sql: KnowledgeStore.selectSQL) {
                let record = KnowledgeStore.foodRecord(from: row)
                map[record.className.lowercased().trimmingCharacters(in: .whitespaces)] = record
            }
            let version = try String.fetchOne(
                db, sql: "SELECT value FROM kb_meta WHERE key = 'active_model_version'"
            )
            return (map, version)
        }
    }

    /// Convenience initializer for the app bundle (`lokma_local.db`).
    public convenience init(bundle: Bundle = .main, resource: String = "lokma_local", ext: String = "db") throws {
        guard let url = bundle.url(forResource: resource, withExtension: ext) else {
            throw KnowledgeError.missingDatabase("\(resource).\(ext) not in bundle")
        }
        try self.init(databaseURL: url)
    }

    /// O(1), case-insensitive — mirrors `get_food_record`.
    public func foodRecord(for className: String) -> FoodRecord? {
        cache[className.lowercased().trimmingCharacters(in: .whitespaces)]
    }

    public func allRecords() -> [FoodRecord] { Array(cache.values) }

    // Mirrors `FoodRecord.from_row` (_as_float -> 0.0, _as_optional_float -> nil).
    private static func foodRecord(from row: Row) -> FoodRecord {
        FoodRecord(
            className: row["class_name"] ?? "",
            source: row["source"],
            usdaDesc: row["usda_desc"],
            caloriesPer100g: row["calories"] ?? 0.0,
            proteinPer100g: row["protein"] ?? 0.0,
            fatPer100g: row["fat"] ?? 0.0,
            carbsPer100g: row["carbs"] ?? 0.0,
            portionG: row["portion_g"] ?? 0.0,
            refArea: row["ref_area"],            // nil when NULL
            density: row["density"],             // nil when NULL
            geometricShape: row["geometric_shape"]
        )
    }
}

public enum KnowledgeError: Error {
    case missingDatabase(String)
}
