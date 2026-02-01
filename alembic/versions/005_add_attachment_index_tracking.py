"""Add attachment index tracking fields to emails

Revision ID: 005_add_attachment_index_tracking
Revises: 004_add_ocr_detection_fields
Create Date: 2026-02-01

Add fields to track attachment indexing status for backfill operations:
- attachment_index_status: None/pending/success/partial/failed
- attachment_index_attempts: retry counter
- attachment_index_last_attempt: timestamp for backoff calculation
- attachment_index_error: last error message
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005_add_attachment_index_tracking'
down_revision: Union[str, Sequence[str], None] = '004_add_ocr_detection_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add attachment indexing tracking fields to emails table."""
    # Add columns
    op.add_column('emails', sa.Column('attachment_index_status', sa.String(length=20), nullable=True))
    op.add_column('emails', sa.Column('attachment_index_attempts', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('emails', sa.Column('attachment_index_last_attempt', sa.DateTime(), nullable=True))
    op.add_column('emails', sa.Column('attachment_index_error', sa.Text(), nullable=True))

    # Add composite index for backfill queries
    op.create_index(
        'ix_emails_attachment_backfill',
        'emails',
        ['has_attachments', 'attachment_index_status', 'date']
    )


def downgrade() -> None:
    """Remove attachment indexing tracking fields."""
    # Drop index
    op.drop_index('ix_emails_attachment_backfill', table_name='emails')

    # Drop columns
    op.drop_column('emails', 'attachment_index_error')
    op.drop_column('emails', 'attachment_index_last_attempt')
    op.drop_column('emails', 'attachment_index_attempts')
    op.drop_column('emails', 'attachment_index_status')
