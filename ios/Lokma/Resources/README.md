# Bundled resources

| File | Source | Tracked? |
|---|---|---|
| `lokma_local.db` | copy of `data/processed/lokma_local.db` (built by `scripts/build_knowledge_base.py`) | yes — small read-only KB snapshot |
| `FoodSeg.mlpackage` | `python scripts/export_coreml.py` (ML env) | no — generated build artifact (gitignored) |
| `FoodSeg.metadata.json` | written alongside the package by the export script | no — generated |

Before the first device build, generate the model:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_coreml.py            # current v1 weights
# or, to prove the pipeline with a placeholder:
KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_coreml.py --pretrained
```

Refreshing the KB snapshot is **automatic**: `scripts/build_knowledge_base.py`
copies the freshly built DB here at the end of the build (pass `--no-deploy` to
skip). The same auto-deploy runs when the data pipeline's nutrition lane builds
the KB. No manual `cp` needed.
