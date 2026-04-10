"""Add financing calculator columns to property_listings

Revision ID: 35a1d4e7f9b2
Revises: 34a0b2c1e5f8
Create Date: 2026-03-25

Adds monthly_cost, monthly_amortization, monthly_total, and financing_details
columns for the property financing calculator.
"""
from typing import Sequence, Union

from alembic import op

revision: str = '35a1d4e7f9b2'
down_revision: Union[str, Sequence[str], None] = '34a0b2c1e5f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE property_listings
        ADD COLUMN IF NOT EXISTS monthly_cost float
    """)
    op.execute("""
        ALTER TABLE property_listings
        ADD COLUMN IF NOT EXISTS monthly_amortization float
    """)
    op.execute("""
        ALTER TABLE property_listings
        ADD COLUMN IF NOT EXISTS monthly_total float
    """)
    op.execute("""
        ALTER TABLE property_listings
        ADD COLUMN IF NOT EXISTS financing_details jsonb
    """)


def downgrade() -> None:
    op.drop_column('property_listings', 'financing_details')
    op.drop_column('property_listings', 'monthly_total')
    op.drop_column('property_listings', 'monthly_amortization')
    op.drop_column('property_listings', 'monthly_cost')
