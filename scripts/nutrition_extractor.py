#!/usr/bin/env python3
"""DEPRECATED — superseded by ``scripts/build_knowledge_base.py``.

The USDA -> SQLite ETL moved into ``lokma/knowledge/builder.py`` (with the
11-vs-9 INSERT mismatch fixed and the embedding corpus computed once). This file
now only redirects.
"""

from __future__ import annotations

import runpy
from pathlib import Path

print("[DEPRECATED] scripts/nutrition_extractor.py -> use `python scripts/build_knowledge_base.py`. Redirecting...")
runpy.run_path(str(Path(__file__).resolve().with_name("build_knowledge_base.py")), run_name="__main__")
