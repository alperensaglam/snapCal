#!/usr/bin/env python3
"""Build (or rebuild) the LOKMA knowledge base ``lokma_local.db`` from USDA data.

Run from the repository root::

    python scripts/build_knowledge_base.py            # build + auto-deploy to iOS
    python scripts/build_knowledge_base.py --no-deploy  # build only

By default the freshly built DB is copied into ``ios/Lokma/Resources/`` so the
Xcode bundle always tracks the latest compiled data (no manual ``cp``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/build_knowledge_base.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402
from lokma.knowledge.builder import KnowledgeBaseBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-deploy", action="store_true",
                        help="build only; do not copy into ios/Lokma/Resources/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AppConfig.from_env()
    count = KnowledgeBaseBuilder(config).build(deploy=not args.no_deploy)
    print(f"Knowledge base built: {count} rows -> {config.db_path}")
    if not args.no_deploy:
        print(f"Deployed -> {config.ios_resources_dir / 'lokma_local.db'}")


if __name__ == "__main__":
    main()
