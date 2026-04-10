"""Add sun exposure, garage price, financing summary columns

Revision ID: 36a2b3c4d5e6
Revises: 35a1d4e7f9b2
Create Date: 2026-03-26

Adds columns for sun assessment, garage pricing, and financing summary text.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '36a2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = '35a1d4e7f9b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE property_listings ADD COLUMN IF NOT EXISTS garage_price float")
    op.execute("ALTER TABLE property_listings ADD COLUMN IF NOT EXISTS terrace_orientation varchar(50)")
    op.execute("ALTER TABLE property_listings ADD COLUMN IF NOT EXISTS sun_exposure_notes text")
    op.execute("ALTER TABLE property_listings ADD COLUMN IF NOT EXISTS sun_score integer")
    op.execute("ALTER TABLE property_listings ADD COLUMN IF NOT EXISTS financing_summary text")
    op.execute("ALTER TABLE property_listings ADD COLUMN IF NOT EXISTS total_cash_needed float")


def downgrade() -> None:
    op.drop_column('property_listings', 'total_cash_needed')
    op.drop_column('property_listings', 'financing_summary')
    op.drop_column('property_listings', 'sun_score')
    op.drop_column('property_listings', 'sun_exposure_notes')
    op.drop_column('property_listings', 'terrace_orientation')
    op.drop_column('property_listings', 'garage_price')
