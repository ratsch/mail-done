#!/usr/bin/env python3
"""Rebuild spam-prototype centroids from seed emails.

Intended to run via cron (daily) OR manually when a new spam wave starts.

Usage:
    poetry run python scripts/rebuild_spam_prototypes.py [--force]

Behavior:
    - Reads config/spam_prototypes.yaml (resolved via CONFIG_DIR)
    - For each prototype: if not yet built, stale (> TTL), or config-hash
      changed, rebuild the centroid and UPSERT into
      spam_prototype_centroids
    - --force overrides all TTL/hash checks and rebuilds everything
    - Exits 0 on success even if no prototypes were rebuilt (no-op case)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make backend importable when run from repo root or scripts/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("rebuild_spam_prototypes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild spam prototype centroids")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if TTL/hash check says it's fresh")
    parser.add_argument("--config", default=None,
                        help="Path to spam_prototypes.yaml (default: CONFIG_DIR/spam_prototypes.yaml)")
    args = parser.parse_args()

    from backend.core.database import init_db, get_db
    from backend.core.ai.prototype_classifier import rebuild_all

    init_db()
    db = next(get_db())
    try:
        results = rebuild_all(db, config_path=args.config, force=args.force)
    finally:
        db.close()

    if not results:
        logger.warning("No prototype configs loaded — nothing to do")
        return 0

    rebuilt = sum(1 for r in results if r["rebuilt"])
    skipped = len(results) - rebuilt
    logger.info(f"Rebuild summary: {rebuilt} rebuilt, {skipped} skipped")
    for r in results:
        mark = "✓ rebuilt" if r["rebuilt"] else "· skipped"
        logger.info(f"  {mark:<12} {r['name']:<30} seeds={r['seed_count']:<4} reason={r['reason']!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
