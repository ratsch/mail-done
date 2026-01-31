"""Add document lifecycle fields.

Revision ID: 003_documents_lifecycle
Revises: 002_documents
Create Date: 2026-01-31

Adds fields for:
- Structured extraction (pages/sheets/sections for per-chunk embeddings)
- Orphan tracking (documents with no remaining origins)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003_documents_lifecycle'
down_revision: Union[str, Sequence[str]] = '002_documents'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add lifecycle columns to documents table."""

    # Structured extraction data (pages/sheets/sections for embedding generation)
    op.add_column('documents', sa.Column(
        'extraction_structure',
        postgresql.JSONB(),
        nullable=True,
        comment='Structured extraction: {"pages": [...]} or {"sheets": [...]} or {"sections": [...]}'
    ))

    # Orphan tracking (document has no remaining origins)
    op.add_column('documents', sa.Column(
        'is_orphaned',
        sa.Boolean(),
        nullable=False,
        server_default='false',
        comment='True when document has no remaining origins'
    ))

    op.add_column('documents', sa.Column(
        'orphaned_at',
        sa.DateTime(timezone=True),
        nullable=True,
        comment='When document became orphaned (for grace period cleanup)'
    ))

    # Partial index for orphan queries (cleanup job)
    op.execute("""
        CREATE INDEX idx_documents_orphaned
        ON documents (orphaned_at)
        WHERE is_orphaned = true AND is_deleted = false
    """)


def downgrade() -> None:
    """Remove lifecycle columns from documents table."""
    op.execute("DROP INDEX IF EXISTS idx_documents_orphaned")
    op.drop_column('documents', 'orphaned_at')
    op.drop_column('documents', 'is_orphaned')
    op.drop_column('documents', 'extraction_structure')
