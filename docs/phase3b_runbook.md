# Phase 3b Runbook — Train `foodyolo_v2` on the Altın Liste

The knowledge layer already knows the Altın Liste; this teaches the **visual**
model to see it. The pipeline is taxonomy-driven, so you only touch data + run
four commands. `class_map(model_version)` makes the v1→v2 swap a clean flip.

## The Altın Liste (14 classes, order = class_id)

Defined once in `lokma/knowledge/taxonomy.py` → `GOLDEN_LIST`:

```
baklava lahmacun doner pide kuru_fasulye mercimek_corbasi pilav kofte   # Turkish
egg chicken_breast cooked_rice banana                                    # essentials
apple_pie breakfast_burrito                                              # keepers
```

To change the set, edit `GOLDEN_LIST` (one tuple) — everything else (data.yaml,
class order, v2 class_map) derives from it.

## Steps

### 1. Acquire images (~200–500 per class)
- **TurkishFoods-15** dataset covers several Turkish dishes.
- Image search using the **en + tr aliases already in the taxonomy** as queries
  (e.g. `lahmacun`, `döner kebap`, `mercimek çorbası`, `tavuk göğsü`).
- Your own photos — prefer varied, near-top-down plated shots (matches the
  capture UX so calibration behaves).

### 2. Organize into per-slug folders
```
data/raw/lokma_v2/
├── baklava/        *.jpg
├── lahmacun/       *.jpg
├── doner/          *.jpg
└── ...             (one folder per GOLDEN_LIST slug)
```

### 3. Auto-label + split
```
python scripts/prepare_v2_dataset.py
```
Produces `data/processed/yolo_v2/{images,labels}/{train,val}` + `data.yaml`.
Labels are **pseudo-masks** from the generic seg model — fast bootstrap, but
spot-check the hero classes; a manual cleanup pass on a subset materially
improves accuracy. The script reports per-class counts and flags empty classes.

### 4. Train
```
python scripts/train.py
```
Writes `runs/train/foodyolo_v2/weights/best.pt` (device auto-detected:
MPS on Apple Silicon; ~hours depending on size/epochs — `config.train_epochs`,
default 80).

### 5. Activate v2
```
export LOKMA_MODEL_PATH="$(pwd)/runs/train/foodyolo_v2/weights/best.pt"
export LOKMA_STRATEGY=auto              # (default)
# set model_version=foodyolo_v2 (env or config) so the KB's active class_map flips
python scripts/build_knowledge_base.py  # ~1s rebuild; v2 class_map ref_areas now live
```

### 6. Verify
```
python scripts/run_live.py              # or the headless smoke
```
Point at a real lahmacun/döner photo → it segments, resolves Turkish macros, and
(with calibration confidence) reports mass via `m = V·ρ` using the per-food
density — end-to-end, through the **unchanged** Phase 1/2 pipeline.

## Notes
- `class_map` keeps **both** `foodyolo_v1` and `foodyolo_v2`; switching is just
  `model_version`. Roll back instantly by flipping it back.
- The compat `nutrition` view keys on `food.slug`, so the v2 model's slug
  class-names resolve with zero pipeline changes.
