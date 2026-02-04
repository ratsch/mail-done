"""Add application_share_tokens table

Revision ID: 006_application_share_tokens
Revises: 005_add_attachment_index_tracking
Create Date: 2026-02-04

Adds secure, time-limited share tokens for sharing application details
with external parties without requiring portal authentication.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '006_application_share_tokens'
down_revision: Union[str, Sequence[str], None] = '005_add_attachment_index_tracking'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create application_share_tokens table."""
    op.create_table(
        'application_share_tokens',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('email_id', UUID(as_uuid=True), sa.ForeignKey('emails.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_by', UUID(as_uuid=True), sa.ForeignKey('lab_members.id', ondelete='SET NULL'), nullable=True),

        # Token security - store hash, not plain token
        sa.Column('token_hash', sa.String(64), nullable=False, unique=True),

        # Permissions - what can be viewed via this share link
        sa.Column('permissions', sa.JSON(), nullable=False, server_default='{}'),

        # Expiration and usage limits
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('max_uses', sa.Integer(), nullable=True),  # NULL = unlimited
        sa.Column('uses_count', sa.Integer(), nullable=False, server_default='0'),

        # Revocation
        sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('revoked_by', UUID(as_uuid=True), sa.ForeignKey('lab_members.id', ondelete='SET NULL'), nullable=True),

        # Audit fields
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_ip', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    # Create indexes
    op.create_index('ix_share_tokens_email_id', 'application_share_tokens', ['email_id'])
    op.create_index('ix_share_tokens_created_by', 'application_share_tokens', ['created_by'])
    op.create_index('ix_share_tokens_token_hash', 'application_share_tokens', ['token_hash'], unique=True)
    op.create_index('ix_share_tokens_expires_at', 'application_share_tokens', ['expires_at'])
    op.create_index('ix_share_tokens_active', 'application_share_tokens', ['email_id', 'is_revoked', 'expires_at'])


def downgrade() -> None:
    """Drop application_share_tokens table."""
    # Drop indexes first
    op.drop_index('ix_share_tokens_active', table_name='application_share_tokens')
    op.drop_index('ix_share_tokens_expires_at', table_name='application_share_tokens')
    op.drop_index('ix_share_tokens_token_hash', table_name='application_share_tokens')
    op.drop_index('ix_share_tokens_created_by', table_name='application_share_tokens')
    op.drop_index('ix_share_tokens_email_id', table_name='application_share_tokens')

    # Drop table
    op.drop_table('application_share_tokens')
