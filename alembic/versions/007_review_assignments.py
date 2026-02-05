"""Add review assignment tables

Revision ID: 007_review_assignments
Revises: 006_application_share_tokens
Create Date: 2026-02-05

Adds assignment_batches, assignment_batch_shares, and
application_review_assignments tables for the review assignments feature.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '007_review_assignments'
down_revision: Union[str, None] = '006_application_share_tokens'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create assignment_batches table
    op.create_table(
        'assignment_batches',
        sa.Column('id', UUID(as_uuid=True), nullable=False),
        sa.Column('created_by', UUID(as_uuid=True), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('deadline', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['created_by'], ['lab_members.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_batches_creator', 'assignment_batches', ['created_by'])

    # 2. Create assignment_batch_shares table
    op.create_table(
        'assignment_batch_shares',
        sa.Column('id', UUID(as_uuid=True), nullable=False),
        sa.Column('batch_id', UUID(as_uuid=True), nullable=False),
        sa.Column('shared_with', UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['batch_id'], ['assignment_batches.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shared_with'], ['lab_members.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('batch_id', 'shared_with', name='uq_batch_share')
    )
    op.create_index('idx_batch_shares_batch', 'assignment_batch_shares', ['batch_id'])
    op.create_index('idx_batch_shares_user', 'assignment_batch_shares', ['shared_with'])

    # 3. Create application_review_assignments table
    op.create_table(
        'application_review_assignments',
        sa.Column('id', UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', UUID(as_uuid=True), nullable=False),
        sa.Column('assigned_to', UUID(as_uuid=True), nullable=False),
        sa.Column('batch_id', UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('declined_reason', sa.Text(), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('declined_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['assigned_to'], ['lab_members.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['batch_id'], ['assignment_batches.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email_id', 'assigned_to', name='uq_assignment_email_user')
    )
    op.create_index('idx_assignments_assignee', 'application_review_assignments', ['assigned_to'])
    op.create_index('idx_assignments_batch', 'application_review_assignments', ['batch_id'])
    op.create_index('idx_assignments_email', 'application_review_assignments', ['email_id'])
    op.create_index('idx_assignments_status', 'application_review_assignments', ['status'])
    op.create_index('idx_assignments_batch_status', 'application_review_assignments', ['batch_id', 'status'])


def downgrade() -> None:
    op.drop_table('application_review_assignments')
    op.drop_table('assignment_batch_shares')
    op.drop_table('assignment_batches')
