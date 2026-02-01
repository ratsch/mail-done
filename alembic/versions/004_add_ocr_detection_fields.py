"""Add OCR detection fields to documents

Revision ID: 004_add_ocr_detection_fields
Revises: 003_documents_lifecycle
Create Date: 2026-02-01

Adds fields for detecting image content and tracking OCR status:
- has_images: File contains image content
- has_native_text: File has extractable text layer
- is_image_only: File is all images, no native text
- ocr_applied: OCR was run on this document
- ocr_pipeline_version: Version of OCR pipeline used
- text_source: Where extracted_text came from ('native', 'ocr', 'mixed', 'none')
- ocr_recommended: Flag for documents that need OCR (initialized by formula, can be overwritten)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '004_add_ocr_detection_fields'
down_revision: Union[str, Sequence[str], None] = '003_documents_lifecycle'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add OCR detection fields to documents table."""
    # Content analysis fields (detected from source file)
    op.add_column('documents', sa.Column('has_images', sa.Boolean(), nullable=True))
    op.add_column('documents', sa.Column('has_native_text', sa.Boolean(), nullable=True))
    op.add_column('documents', sa.Column('is_image_only', sa.Boolean(), nullable=True))
    op.add_column('documents', sa.Column('is_scanned_with_ocr', sa.Boolean(), nullable=True))

    # OCR state fields (our processing in DB)
    op.add_column('documents', sa.Column('ocr_applied', sa.Boolean(), server_default='false', nullable=True))
    op.add_column('documents', sa.Column('ocr_pipeline_version', sa.String(50), nullable=True))
    op.add_column('documents', sa.Column('text_source', sa.String(20), nullable=True))

    # OCR recommendation flag
    op.add_column('documents', sa.Column('ocr_recommended', sa.Boolean(), nullable=True))

    # Indexes for efficient OCR candidate queries
    op.create_index(
        'idx_documents_ocr_recommended',
        'documents',
        ['id'],
        unique=False,
        postgresql_where=sa.text('ocr_recommended = true')
    )
    op.create_index('idx_documents_ocr_applied', 'documents', ['ocr_applied'], unique=False)
    op.create_index('idx_documents_text_source', 'documents', ['text_source'], unique=False)


def downgrade() -> None:
    """Remove OCR detection fields from documents table."""
    op.drop_index('idx_documents_text_source', table_name='documents')
    op.drop_index('idx_documents_ocr_applied', table_name='documents')
    op.drop_index('idx_documents_ocr_recommended', table_name='documents')

    op.drop_column('documents', 'ocr_recommended')
    op.drop_column('documents', 'text_source')
    op.drop_column('documents', 'ocr_pipeline_version')
    op.drop_column('documents', 'ocr_applied')
    op.drop_column('documents', 'is_scanned_with_ocr')
    op.drop_column('documents', 'is_image_only')
    op.drop_column('documents', 'has_native_text')
    op.drop_column('documents', 'has_images')
