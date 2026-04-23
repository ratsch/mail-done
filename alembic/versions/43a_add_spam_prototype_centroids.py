"""Add spam_prototype_centroids table for embedding-based spam filtering

Revision ID: 43a1b2c3d4e5
Revises: 42a1b2c3d4e5
Create Date: 2026-04-23

Stores one row per spam-campaign prototype (BCBS insurance, AAA road kit,
future additions). Each row carries:
  - name:              unique identifier referenced by config + sweep code
  - centroid:          normalized mean of seed-email embeddings (vector(EMBEDDING_DIM))
  - threshold:         cosine-distance cutoff (e.g., 0.20)
  - action_folder:     destination folder for matched emails (e.g., 'MD/Spam')
  - action_category:   ai_category to record when matched (e.g., 'spam')
  - seed_count:        how many seed emails the centroid was built from
  - built_at:          timestamp of last rebuild — used for TTL (7 days)
  - config_hash:       sha256 of seed_filter + threshold + embedding_model,
                       so a config change triggers rebuild
  - embedding_model:   which embedding model the centroid targets
                       (centroids are model-specific; a model change
                       invalidates the centroid entirely)

Classification is a single SQL JOIN against ``email_embeddings`` using the
existing diskann index — no Python-side vector math at sweep time.
"""
import os
from typing import Sequence, Union

from alembic import op

revision: str = '43a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = '42a1b2c3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _require_embedding_dim() -> int:
    """Read EMBEDDING_DIM from env. Invoked inside upgrade() so read-only
    alembic commands don't require the env var."""
    raw = os.environ.get("EMBEDDING_DIM")
    if not raw:
        raise RuntimeError(
            "EMBEDDING_DIM environment variable is required for alembic "
            "migrations. Set it (matching migration 001's dim) before "
            "running `alembic upgrade`."
        )
    return int(raw)


def upgrade() -> None:
    embedding_dim = _require_embedding_dim()
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS spam_prototype_centroids (
            id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name             varchar(100) NOT NULL UNIQUE,
            description      text,
            centroid         vector({embedding_dim}) NOT NULL,
            threshold        double precision NOT NULL,
            action_folder    varchar(200) NOT NULL,
            action_category  varchar(50) NOT NULL,
            seed_count       integer NOT NULL,
            built_at         timestamp NOT NULL DEFAULT now(),
            config_hash      varchar(64) NOT NULL,
            embedding_model  varchar(50) NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_spam_prototype_centroids_built_at
        ON spam_prototype_centroids (built_at)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_spam_prototype_centroids_built_at")
    op.execute("DROP TABLE IF EXISTS spam_prototype_centroids")
