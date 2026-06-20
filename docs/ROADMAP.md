# LOKMA Roadmap

LOKMA = optical nutrition scale: `mass = V × ρ × (1 − P)` from a camera frame
(YOLOv11-seg masks → volume V; density ρ; porosity correction P → grams → macros).
Architecture rule: the Python `lokma/` package is the source of truth; the Swift
`ios/LokmaCore` port is pinned to it by **golden-vector parity** (`scripts/dump_golden_vectors.py`
→ `ParityTests.swift`). Regenerate + `swift test` + `pytest` after any math change.

## Shipped

| Phase | What | State |
|---|---|---|
| 1–3 | `lokma/` services, frictionless calibration, multilingual knowledge | done |
| 3b | Golden List taxonomy + training scaffolding (data-gated) | code done |
| 4 | iOS app (ARKit + CoreML) on parity-tested `LokmaCore` | on-device |
| Tier 1 | Tilt-robust scalar engine (1/cos foreshortening, gates, depth-confidence, smoothing) | `cc13d6a` |
| Tier 2 | **LiDAR depth-integrated volume** above a fitted support plane | `5caede0`/`e884d69`/`d419e2c`, on-device |
| 5 | **Porosity correction** `mass = V·ρ·(1−P)`, hybrid (ML-ready) | this phase |

## Post-Tier-2 backlog — "The Big 5" (Phases 5→9)

Sequenced by architectural dependency, not ambition:

| Phase | Track | Scope | Depends on |
|---|---|---|---|
| **5** | Porosity & Structural Correction | `mass = V·ρ·(1−P)`; per-class `CATEGORY_POROSITY` fallback + optional ML-predicted P | — (**done**) |
| **6** | Per-Instance Multi-Object Tracking | upgrade `MassStabilizer` from class-keyed to spatial-ID (ARKit anchors) so same-dish plates smooth independently | — |
| **7** | Golden List Dataset & Retrain | 14-class Altın Liste YOLOv11-seg + density/seg metadata; add a **context-regression head** (height + porosity) | data collection |
| **8** | Non-LiDAR ML Height Regressor | dynamic per-instance height for LiDAR-less iPhones; also upgrades Phase 5 P to model-predicted | Phase 7 |
| **9** | Apple-Grade UI/UX | remove debug strings; neon segmentation overlay; Health-style animated macro cards on mass-lock | engine final |

**Keystone:** Phases 8 and the model-predicted upgrade of Phase 5 both need the model to
emit per-instance context scalars — so Phase 7's **shared context-regression head** is the
single investment that unblocks both. This is the "physical depth + ML context" merge.

**Phase 5 hybrid architecture (shipped):** `Detection.predicted_porosity` (optional float,
both languages) → resolver: `predicted ?? porosity_for(class) ?? 0.0` → `grams = V·ρ·(1−P)`,
applied in the AutoStrategy depth branch and the scalar `VolumetricStrategy`. The ML field +
canonical formula + fallback routing all exist now; Phase 7 only starts *populating* the
predicted value — no Swift/Python refactor needed.

**Parallel now:** Phase 7 image collection into `data/raw/lokma_v2/<slug>/` (the slow,
user-gated long pole) per `docs/phase3b_runbook.md`.
