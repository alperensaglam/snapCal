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

To refresh the KB snapshot after rebuilding it:

```bash
cp data/processed/lokma_local.db ios/Lokma/Resources/lokma_local.db
```
