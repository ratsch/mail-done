"""Add enrichment status tracking columns

Revision ID: 33ac1c8dc94b
Revises: 5348f1936c09
Create Date: 2026-03-23

Adds enrichment_status, enrichment_started_at, enrichment_error
to property_listings for frontend progress tracking.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '33ac1c8dc94b'
down_revision: Union[str, Sequence[str], None] = '5348f1936c09'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('property_listings', sa.Column('enrichment_status', sa.String(), nullable=True))
    op.add_column('property_listings', sa.Column('enrichment_started_at', sa.DateTime(), nullable=True))
    op.add_column('property_listings', sa.Column('enrichment_error', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('property_listings', 'enrichment_error')
    op.drop_column('property_listings', 'enrichment_started_at')
    op.drop_column('property_listings', 'enrichment_status')
