"""Add validated column to property_emails

Revision ID: 38a1b2c3d4e5
Revises: 37a1b2c3d4e5
Create Date: 2026-04-12

NULL = not checked, True = LLM confirmed relevant, False = irrelevant (hidden from UI).
"""
from typing import Sequence, Union
from alembic import op

revision: str = '38a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = '37a1b2c3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE property_emails ADD COLUMN IF NOT EXISTS validated BOOLEAN")


def downgrade() -> None:
    op.execute("ALTER TABLE property_emails DROP COLUMN IF EXISTS validated")
