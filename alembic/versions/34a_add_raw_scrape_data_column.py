"""Add raw_scrape_data column for chrome-bridge extraction storage

Revision ID: 34a0b2c1e5f8
Revises: 33ac1c8dc94b
Create Date: 2026-03-24

Stores the full chrome-bridge extraction JSON (~5KB per listing) as JSONB.
Allows re-parsing if the enrichment parser improves without re-scraping.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = '34a0b2c1e5f8'
down_revision: Union[str, Sequence[str], None] = '33ac1c8dc94b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE property_listings
        ADD COLUMN IF NOT EXISTS raw_scrape_data jsonb
    """)


def downgrade() -> None:
    op.drop_column('property_listings', 'raw_scrape_data')
