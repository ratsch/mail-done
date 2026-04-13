"""Add MFH-specific fields to property_listings

Revision ID: 39a1b2c3d4e5
Revises: 38a1b2c3d4e5
Create Date: 2026-04-13

Fields for Scenario C (multi-family house) evaluation: gross yield,
rental income, zoning, development potential, ÖREB risk flags, STWE
conversion feasibility, ideal party count.
"""
from typing import Sequence, Union
from alembic import op

revision: str = '39a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = '38a1b2c3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE property_listings
            ADD COLUMN IF NOT EXISTS bruttorendite_pct FLOAT,
            ADD COLUMN IF NOT EXISTS annual_rental_income_chf INTEGER,
            ADD COLUMN IF NOT EXISTS zoning_zone VARCHAR(10),
            ADD COLUMN IF NOT EXISTS ausnutzungsreserve_sqm INTEGER,
            ADD COLUMN IF NOT EXISTS development_options JSONB,
            ADD COLUMN IF NOT EXISTS denkmalschutz BOOLEAN,
            ADD COLUMN IF NOT EXISTS altlasten BOOLEAN,
            ADD COLUMN IF NOT EXISTS kernzone BOOLEAN,
            ADD COLUMN IF NOT EXISTS stwe_conversion_feasibility VARCHAR(20),
            ADD COLUMN IF NOT EXISTS ideal_party_count INTEGER
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE property_listings
            DROP COLUMN IF EXISTS bruttorendite_pct,
            DROP COLUMN IF EXISTS annual_rental_income_chf,
            DROP COLUMN IF EXISTS zoning_zone,
            DROP COLUMN IF EXISTS ausnutzungsreserve_sqm,
            DROP COLUMN IF EXISTS development_options,
            DROP COLUMN IF EXISTS denkmalschutz,
            DROP COLUMN IF EXISTS altlasten,
            DROP COLUMN IF EXISTS kernzone,
            DROP COLUMN IF EXISTS stwe_conversion_feasibility,
            DROP COLUMN IF EXISTS ideal_party_count
    """)
