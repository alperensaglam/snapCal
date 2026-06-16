#!/usr/bin/env python3
"""Prepare the v2 (Altın Liste) YOLO-seg dataset.

Reads ``data/raw/lokma_v2/<slug>/*.jpg`` (one folder per taxonomy slug),
auto-labels + splits into ``data/processed/yolo_v2/``, and writes ``data.yaml``.

    python scripts/prepare_v2_dataset.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402
from lokma.training.dataset_builder import build_v2_dataset  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AppConfig()
    counts = build_v2_dataset(config)
    total = sum(counts.values())
    print(f"\nv2 dataset: {total} images across {len(counts)} classes -> {config.yolo_v2_dataset_dir}")
    for slug, n in counts.items():
        note = "" if n else f"   <-- add images to {config.raw_v2_dir / slug}/"
        print(f"  {slug:18} {n}{note}")
    if total == 0:
        print("\nNo images found. See docs/phase3b_runbook.md for how to acquire them.")


if __name__ == "__main__":
    main()
