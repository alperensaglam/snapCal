#!/usr/bin/env python3
"""Build (or rebuild) the LOKMA knowledge base ``lokma_local.db`` from USDA data.

Run from the repository root::

    python scripts/build_knowledge_base.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/build_knowledge_base.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402
from lokma.knowledge.builder import KnowledgeBaseBuilder  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AppConfig.from_env()
    count = KnowledgeBaseBuilder(config).build()
    print(f"Knowledge base built: {count} rows -> {config.db_path}")


if __name__ == "__main__":
    main()
