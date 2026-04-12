"""Add property_emails table for linking correspondence to listings

Revision ID: 37a1b2c3d4e5
Revises: 36a2b3c4d5e6
Create Date: 2026-04-12

Links emails (agent replies, inquiries, bank correspondence) to
property listings for display in the portal and LLM scoring context.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '37a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = '36a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS property_emails (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            listing_id UUID NOT NULL REFERENCES property_listings(id) ON DELETE CASCADE,
            email_id UUID NOT NULL REFERENCES emails(id),
            email_type VARCHAR NOT NULL,
            relevance_score FLOAT,
            linked_by VARCHAR DEFAULT 'auto',
            linked_at TIMESTAMP DEFAULT now(),
            CONSTRAINT uq_property_email UNIQUE (listing_id, email_id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_property_emails_listing ON property_emails (listing_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_property_emails_email ON property_emails (email_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS property_emails")
