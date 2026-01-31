"""Add document indexing tables.

Revision ID: 002_documents
Revises: 001_initial
Create Date: 2026-01-31

Creates tables for document indexing feature:
- documents: Core document storage with checksum deduplication
- document_origins: Track where documents are found (folder, email attachment, etc.)
- document_embeddings: Vector embeddings for semantic search
- document_processing_queue: Async processing queue

Architecture follows email storage model:
- Reference-only storage (files stay at origin)
- Encrypted text storage
- Plaintext metadata for keyword search
- Vector embeddings (diskann index) for semantic search
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_documents'
down_revision: Union[str, Sequence[str]] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create document indexing tables."""

    # ==========================================================================
    # documents - Core document storage with checksum deduplication
    # ==========================================================================
    op.create_table('documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),

        # Identity (deduplication key)
        sa.Column('checksum', sa.String(64), nullable=False),
        sa.Column('checksum_algorithm', sa.String(20), server_default='sha256'),

        # File metadata
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('mime_type', sa.String(127), nullable=True),
        sa.Column('original_filename', sa.String(500), nullable=True),
        sa.Column('page_count', sa.Integer(), nullable=True),

        # Extracted text (ENCRYPTED - consistent with email body treatment)
        sa.Column('extracted_text', sa.Text(), nullable=True),  # Encrypted via EncryptedText type

        # Extraction state machine
        sa.Column('extraction_status', sa.String(20), nullable=False, server_default='pending'),

        # Extraction metadata
        sa.Column('extraction_version', sa.String(20), nullable=True),
        sa.Column('extraction_method', sa.String(50), nullable=True),
        sa.Column('extraction_model', sa.String(100), nullable=True),
        sa.Column('extraction_quality', sa.Float(), nullable=True),
        sa.Column('extraction_cost', sa.Float(), nullable=True),
        sa.Column('extracted_at', sa.DateTime(timezone=True), nullable=True),

        # Plaintext metadata (keyword searchable - like email subject)
        sa.Column('title', sa.String(500), nullable=True),
        sa.Column('summary', sa.String(1000), nullable=True),
        sa.Column('document_date', sa.Date(), nullable=True),
        sa.Column('document_type', sa.String(100), nullable=True),
        sa.Column('language', sa.String(10), nullable=True),
        sa.Column('ai_category', sa.String(100), nullable=True),
        sa.Column('ai_tags', postgresql.ARRAY(sa.String()), nullable=True),

        # Lifecycle
        sa.Column('first_seen_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('is_deleted', sa.Boolean(), server_default='false'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),

        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('checksum')
    )

    # Document indexes
    op.create_index('idx_documents_checksum', 'documents', ['checksum'])
    op.create_index('idx_documents_mime_type', 'documents', ['mime_type'])
    op.create_index('idx_documents_document_type', 'documents', ['document_type'])
    op.create_index('idx_documents_document_date', 'documents', ['document_date'])
    op.create_index('idx_documents_extraction_quality', 'documents', ['extraction_quality'])
    op.create_index('idx_documents_ai_category', 'documents', ['ai_category'])
    op.create_index('idx_documents_extraction_status', 'documents', ['extraction_status'])

    # GIN index for ai_tags array search
    op.execute("CREATE INDEX idx_documents_ai_tags ON documents USING GIN(ai_tags)")

    # Trigram indexes for keyword search on title/summary (like email subject search)
    op.execute("CREATE INDEX idx_documents_title_trgm ON documents USING GIN(title gin_trgm_ops)")
    op.execute("CREATE INDEX idx_documents_summary_trgm ON documents USING GIN(summary gin_trgm_ops)")

    # Partial index for documents needing extraction
    op.execute("""
        CREATE INDEX idx_documents_needs_extraction ON documents(id)
        WHERE extraction_status IN ('pending', 'failed')
    """)

    # ==========================================================================
    # document_origins - Track where documents are found
    # ==========================================================================
    op.create_table('document_origins',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),

        # Origin identification
        sa.Column('origin_type', sa.String(50), nullable=False),  # 'folder', 'email_attachment', 'google_drive'
        sa.Column('origin_host', sa.String(255), nullable=True),
        sa.Column('origin_path', sa.Text(), nullable=True),
        sa.Column('origin_filename', sa.String(500), nullable=True),

        # For email attachments
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('attachment_index', sa.Integer(), nullable=True),

        # Discovery metadata
        sa.Column('file_modified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('discovered_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_primary', sa.Boolean(), server_default='false'),

        # Status
        sa.Column('is_deleted', sa.Boolean(), server_default='false'),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),

        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # Document origins indexes
    op.create_index('idx_document_origins_document', 'document_origins', ['document_id'])
    op.create_index('idx_document_origins_email', 'document_origins', ['email_id'])
    op.create_index('idx_document_origins_host', 'document_origins', ['origin_host'])
    op.create_index('idx_document_origins_type', 'document_origins', ['origin_type'])

    # Unique constraint: prevent duplicate origins for same document
    op.execute("""
        CREATE UNIQUE INDEX uq_document_origins_location
        ON document_origins (document_id, origin_type, origin_host, origin_path)
        WHERE origin_path IS NOT NULL
    """)

    # Unique constraint for email attachment origins
    op.execute("""
        CREATE UNIQUE INDEX uq_document_origins_email_attachment
        ON document_origins (document_id, email_id, attachment_index)
        WHERE email_id IS NOT NULL
    """)

    # ==========================================================================
    # document_embeddings - Vector embeddings for semantic search
    # ==========================================================================
    op.create_table('document_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),

        # Page/chunk identification
        sa.Column('page_number', sa.Integer(), nullable=True),  # NULL for non-paginated
        sa.Column('chunk_index', sa.Integer(), server_default='0'),
        sa.Column('chunk_start', sa.Integer(), nullable=True),
        sa.Column('chunk_end', sa.Integer(), nullable=True),
        sa.Column('chunk_text', sa.Text(), nullable=True),  # Encrypted

        # Model info (must match email embedding model)
        sa.Column('model', sa.String(100), nullable=False, server_default='text-embedding-3-large'),
        sa.Column('model_version', sa.String(50), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),

        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Add vector column using raw SQL (pgvector) - 3072 dimensions for text-embedding-3-large
    op.execute("ALTER TABLE document_embeddings ADD COLUMN embedding vector(3072) NOT NULL")

    # Document embeddings indexes
    op.create_index('idx_document_embeddings_document', 'document_embeddings', ['document_id'])

    # Unique constraint: one embedding per document + page + chunk
    op.execute("""
        CREATE UNIQUE INDEX uq_document_embeddings_page_chunk
        ON document_embeddings (document_id, COALESCE(page_number, 0), chunk_index)
    """)

    # Vector index using DiskANN (same as email_embeddings)
    op.execute("""
        CREATE INDEX idx_document_embeddings_vector
        ON document_embeddings
        USING diskann (embedding)
    """)

    # ==========================================================================
    # document_processing_queue - Async processing queue
    # ==========================================================================
    op.create_table('document_processing_queue',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=True),

        # Task type
        sa.Column('task_type', sa.String(50), nullable=False),  # 'extract_text', 'generate_embedding', 'classify'
        sa.Column('priority', sa.Integer(), server_default='5'),

        # Status
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('attempts', sa.Integer(), server_default='0'),
        sa.Column('max_attempts', sa.Integer(), server_default='3'),
        sa.Column('last_error', sa.Text(), nullable=True),

        # Scheduling
        sa.Column('scheduled_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('worker_id', sa.String(100), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),

        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Processing queue indexes
    op.create_index('idx_processing_queue_document', 'document_processing_queue', ['document_id'])
    op.create_index('idx_processing_queue_status', 'document_processing_queue', ['status'])
    op.create_index('idx_processing_queue_task_type', 'document_processing_queue', ['task_type'])

    # Partial index for pending tasks by priority
    op.execute("""
        CREATE INDEX idx_processing_queue_pending
        ON document_processing_queue (priority, scheduled_at)
        WHERE status = 'pending'
    """)


def downgrade() -> None:
    """Drop document indexing tables."""
    op.drop_table('document_processing_queue')
    op.drop_table('document_embeddings')
    op.drop_table('document_origins')
    op.drop_table('documents')
