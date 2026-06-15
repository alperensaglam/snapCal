"""Pytest bootstrap: ensure the repo root is importable so ``import lokma`` works."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
