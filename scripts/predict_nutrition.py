#!/usr/bin/env python3
"""DEPRECATED — superseded by ``scripts/run_live.py``.

The live-inference logic moved into ``lokma/pipeline/inference_pipeline.py`` and
its thin UI adapter ``scripts/run_live.py``. This file now only redirects, so any
existing muscle memory / shortcuts keep working.
"""

from __future__ import annotations

import runpy
from pathlib import Path

print("[DEPRECATED] scripts/predict_nutrition.py -> use `python scripts/run_live.py`. Redirecting...")
runpy.run_path(str(Path(__file__).resolve().with_name("run_live.py")), run_name="__main__")
