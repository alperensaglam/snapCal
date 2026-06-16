# Phase 4 Runbook — iOS app (CoreML + ARKit) on the parity-tested core

Phase 4 puts the LOKMA pipeline on-device. The estimation **math** is not
rewritten — it is ported to Swift (`ios/LokmaCore`) and pinned to the Python
reference by golden-vector parity, so the two can never silently drift. The app
shell (`ios/Lokma`) supplies only the Apple-framework edges: ARKit capture,
CoreML segmentation, mask decode, the read-only KB, and the SwiftUI overlay.

Python `lokma/` stays the **source of truth**. Change the math there, re-dump the
vectors, and the Swift parity gate tells you what to update.

## Architecture map (Python → Swift)

| Concern | Python (source of truth) | Swift |
|---|---|---|
| Pure math (intrinsics, volume, density, strategies, calibration) | `lokma/core`, `lokma/geometry`, `lokma/density` | `LokmaCore/Sources/LokmaCore/*` |
| Per-frame device context | `FrameContext.from_device` | `Capture/ARCaptureController.swift` |
| Segmentation + decode | YOLOv11-seg / `InferencePipeline._build_detection` | `Vision/FoodSegModel.swift`, `YOLOSegDecoder.swift`, `DetectionBuilder.swift` |
| Knowledge base read | `DatabaseManager.load` (via `nutrition` view) | `Knowledge/KnowledgeStore.swift` (GRDB) |
| Post-seg estimation | `InferencePipeline.process_frame` | `Pipeline/EstimationEngine.swift` |
| Parity contract | `scripts/dump_golden_vectors.py` | `Tests/LokmaCoreTests` + `LokmaParity` |

`LokmaCore` has **no** ARKit/CoreML/Vision imports, so it builds and tests on a
plain macOS toolchain — the parity gate runs without a device or full Xcode.

## One-time setup

```bash
brew install xcodegen          # generates Lokma.xcodeproj from ios/project.yml
```

## Build steps

### 1. Export the CoreML model (ML env)
```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_coreml.py            # current v1 weights
# or prove the pipeline end-to-end with a placeholder:
KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_coreml.py --pretrained
```
Writes `ios/Lokma/Resources/FoodSeg.mlpackage` + `FoodSeg.metadata.json`
(model_version, imgsz, class names). Both are gitignored build artifacts.
`imgsz` is pinned to `config.mask_resolution` (640) so on-device mask areas land
on the same grid the stored `ref_area` assumes — **do not** override it.

While validating the decode, export fp32 + raw NMS so shapes are inspectable:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_coreml.py --pretrained --no-half --no-nms
```

### 2. Refresh the bundled KB snapshot (only after rebuilding it)
```bash
cp data/processed/lokma_local.db ios/Lokma/Resources/lokma_local.db
```
The bundled DB carries both `foodyolo_v1` and `foodyolo_v2` class_maps and a
`kb_meta.active_model_version`; the `nutrition` view resolves `ref_area` against
that key, so swapping the DB (or the meta value) is the whole v1→v2 flip.

### 3. Generate the project and build
```bash
cd ios && xcodegen generate    # writes Lokma.xcodeproj (gitignored)
open Lokma.xcodeproj            # build & run on an ARKit device
```
Requires a physical device — ARKit/CoreML do not run in the Simulator.

## Verification ladder

The code comments reference these by number; run them in order on the first real
export.

1. **Confirm the CoreML output layout.** Open `FoodSeg.mlpackage` in Xcode (or
   print `model.modelDescription.outputDescriptionsByName`). The raw (`--no-nms`)
   ultralytics layout is `output0 [1, 4+nc+32, N]` (box+scores+coeffs) and
   `output1 [1, 32, mh, mw]` (proto masks). If the feature **names** differ, set
   `FoodSegModel.outputKey0/outputKey1`; the decoder also auto-resolves by rank
   (3-D → dets, 4-D → proto) as a fallback.
2. **Parity gate (the contract).** Math port == Python reference:
   ```bash
   python scripts/dump_golden_vectors.py          # refresh fixtures from Python
   cd ios/LokmaCore && swift run LokmaParity       # CLI gate — no Xcode needed
   # with full Xcode: `swift test` runs the same checks under XCTest
   ```
   Expect `✅ all 113 checks passed`. Any failure prints the case + both values.
3. **Single-image decode check.** Feed one known photo through `FoodSegModel`;
   confirm instance count, class names, and mask coverage look sane.
4. **Calibration ladder.** With LiDAR you should see `calib: depth_intrinsics`
   (conf 0.90); on a non-LiDAR device it degrades to plane → anchor → static. The
   overlay prints the live source + confidence.
5. **Device↔Python cross-check.** Run the same frame through `run_live.py` and the
   app; grams should agree within mask-resampling tolerance. `DetectionBuilder`
   uses an area-preserving coverage fraction rather than a second `cv2.resize`, so
   small differences here are expected and bounded.

## Known seams (fill on first real export)

- **`FoodSegModel.resized()`** is a pass-through. If the model's input layer does
  not auto-resize the camera buffer to `imgsz`, replace it with a real CIContext
  square render (or a `VNImageRequestHandler` crop-and-scale).
- **Decode is export-shape-dependent.** Validate shapes per step 1 before trusting
  masses; the decoder assumes Float32 multiarrays (export `--no-half` while
  validating).

## Swapping v1 → v2 on device

1. Train `foodyolo_v2` per `docs/phase3b_runbook.md`.
2. `python scripts/export_coreml.py --weights runs/train/foodyolo_v2/weights/best.pt --model-version foodyolo_v2`
3. Set `active_model_version = foodyolo_v2` in the bundled DB (rebuild the KB,
   then copy it per step 2), and rebuild the app.

No Swift changes: the model sidecar carries the class names, and the `nutrition`
view keys on `food.slug`, so v2's slug class-names resolve unchanged.
