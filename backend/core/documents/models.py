"""
SQLAlchemy Models for Document Indexing

Provides document storage with:
- Checksum-based deduplication (SHA-256)
- Multi-origin tracking (folder, email attachment, Google Drive)
- Encrypted text storage (consistent with email body encryption)
- Page-level embeddings (3072 dimensions, text-embedding-3-large)
- Processing queue for async extraction/embedding

Architecture follows email storage model:
- Source of truth stays at origin (reference-only, no binary storage)
- Extracted text encrypted at rest
- Plaintext metadata for keyword search (title, summary)
- Vector embeddings for semantic search
"""

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    BigInteger,
    ForeignKey,
    Index,
    Enum,
    Date,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.core.database.models import Base
from backend.core.database.encryption import EncryptedText

# pgvector for native vector support
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False


class ExtractionStatus(str, enum.Enum):
    """
    Document text extraction lifecycle states.

    pending -> processing -> completed | no_content | failed
    """
    PENDING = "pending"           # Queued for extraction
    PROCESSING = "processing"     # Worker is extracting
    COMPLETED = "completed"       # Has text, quality scored
    NO_CONTENT = "no_content"     # Processed, no extractable text (image, empty file)
    FAILED = "failed"             # Extraction failed (will retry)


class Document(Base):
    """
    Core document record - one row per unique document by checksum.

    Follows email storage pattern:
    - Source of truth stays at origin (reference-only, no binary storage)
    - Extracted text encrypted at rest
    - Plaintext metadata for keyword search (title, summary)
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Identity (deduplication key)
    checksum = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256
    checksum_algorithm = Column(String(20), default="sha256")

    # File metadata
    file_size = Column(BigInteger, nullable=False)
    mime_type = Column(String(127))
    original_filename = Column(String(500))
    page_count = Column(Integer)

    # Extracted text (ENCRYPTED - consistent with email body treatment)
    extracted_text = Column(EncryptedText)

    # Extraction state machine
    extraction_status = Column(
        String(20),
        default=ExtractionStatus.PENDING.value,
        nullable=False
    )

    # Extraction metadata (not sensitive - plaintext OK)
    extraction_version = Column(String(20))
    extraction_method = Column(String(50))  # 'sandboxed', 'pdftotext', 'tesseract', 'claude'
    extraction_model = Column(String(100))  # LLM model if used
    extraction_quality = Column(Float)       # 0.0-1.0
    extraction_cost = Column(Float)          # API cost in USD
    extracted_at = Column(DateTime(timezone=True))

    # Content analysis (detected from source file)
    has_images = Column(Boolean)              # File contains image content (scanned pages, embedded images)
    has_native_text = Column(Boolean)         # File has extractable text layer built-in
    is_image_only = Column(Boolean)           # File is all images, no native text
    is_scanned_with_ocr = Column(Boolean)     # Scanned PDF with existing OCR text overlay

    # OCR state (our processing in DB)
    ocr_applied = Column(Boolean, default=False)    # OCR was run, result stored in extracted_text
    ocr_pipeline_version = Column(String(50))       # e.g., "tesseract-5.3", "claude-ocr-v1"
    text_source = Column(String(20))                # 'native', 'ocr', 'mixed', 'none'

    # OCR recommendation flag - initialized by formula, can be overwritten
    # Default: (is_image_only AND NOT ocr_applied) OR (has_images AND NOT has_native_text AND NOT ocr_applied)
    ocr_recommended = Column(Boolean)

    # Structured extraction data for per-page/per-sheet embeddings
    # JSON format: {"pages": [{"page": 1, "text": "..."}]} or {"sheets": [{"sheet": "Name", "text": "..."}]}
    extraction_structure = Column(JSON)

    # Plaintext metadata (like email subject - keyword searchable)
    title = Column(String(500))              # Extracted or filename-derived
    summary = Column(String(1000))           # One-line description ("subject" equivalent)
    document_date = Column(Date)
    document_type = Column(String(100))      # 'invoice', 'contract', 'letter'
    language = Column(String(10))
    ai_category = Column(String(100))
    ai_tags = Column(ARRAY(String))

    # NOTE: No binary storage columns - files stay at origin (like emails stay on IMAP)
    # Use document_origins.origin_path to retrieve original file

    # Lifecycle
    first_seen_at = Column(DateTime(timezone=True), default=func.now())
    last_seen_at = Column(DateTime(timezone=True), default=func.now())
    is_deleted = Column(Boolean, default=False)

    # Orphan tracking (document has no remaining origins)
    is_orphaned = Column(Boolean, default=False)
    orphaned_at = Column(DateTime(timezone=True))  # When it became orphaned
    # Grace period: orphaned documents are deleted after N days (configurable)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Relationships
    origins = relationship("DocumentOrigin", back_populates="document", cascade="all, delete-orphan")
    embeddings = relationship("DocumentEmbedding", back_populates="document", cascade="all, delete-orphan")
    processing_tasks = relationship("DocumentProcessingQueue", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_documents_checksum', 'checksum'),
        Index('idx_documents_mime_type', 'mime_type'),
        Index('idx_documents_document_type', 'document_type'),
        Index('idx_documents_document_date', 'document_date'),
        Index('idx_documents_extraction_quality', 'extraction_quality'),
        Index('idx_documents_ai_category', 'ai_category'),
        Index('idx_documents_extraction_status', 'extraction_status'),
        # Partial index for documents needing processing
        Index(
            'idx_documents_needs_extraction',
            'id',
            postgresql_where=(extraction_status.in_([ExtractionStatus.PENDING.value, ExtractionStatus.FAILED.value]))
        ),
        # Partial index for orphaned documents (for cleanup)
        Index(
            'idx_documents_orphaned',
            'orphaned_at',
            postgresql_where=(is_orphaned == True)
        ),
        # Partial index for documents needing OCR
        Index(
            'idx_documents_ocr_recommended',
            'id',
            postgresql_where=(ocr_recommended == True)
        ),
        # Index for OCR state queries
        Index('idx_documents_ocr_applied', 'ocr_applied'),
        Index('idx_documents_text_source', 'text_source'),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.original_filename}, status={self.extraction_status})>"


class DocumentOrigin(Base):
    """
    Track where each document was found (and how to retrieve it).
    This is the equivalent of IMAP folder/UID for emails.

    A document can have multiple origins (same file found in different places).
    """
    __tablename__ = "document_origins"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)

    # Origin identification (used to retrieve original file)
    origin_type = Column(String(50), nullable=False)  # 'folder', 'email_attachment', 'google_drive'
    origin_host = Column(String(255))                  # 'nas.local', 'laptop', 'nvme-pi'
    # NOTE: origin_path is the FULL file path including filename (e.g., "/path/to/file.pdf")
    # origin_filename is stored separately for display/search convenience only
    origin_path = Column(Text)                         # Full path to file INCLUDING filename
    origin_filename = Column(String(500))              # Filename (redundant, for display/search)

    # For email attachments specifically
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id', ondelete='SET NULL'))
    attachment_index = Column(Integer)

    # Discovery metadata
    file_modified_at = Column(DateTime(timezone=True))
    discovered_at = Column(DateTime(timezone=True), default=func.now())
    last_verified_at = Column(DateTime(timezone=True))
    is_primary = Column(Boolean, default=False)

    # Status
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    document = relationship("Document", back_populates="origins")
    email = relationship("Email", foreign_keys=[email_id])

    __table_args__ = (
        Index('idx_document_origins_document', 'document_id'),
        Index('idx_document_origins_email', 'email_id'),
        Index('idx_document_origins_host', 'origin_host'),
        Index('idx_document_origins_type', 'origin_type'),
        # Unique constraint: same origin location for same document
        Index(
            'uq_document_origins_location',
            'document_id', 'origin_type', 'origin_host', 'origin_path',
            unique=True
        ),
    )

    def __repr__(self) -> str:
        return f"<DocumentOrigin(id={self.id}, type={self.origin_type}, path={self.origin_path})>"


class DocumentEmbedding(Base):
    """
    Vector embeddings for semantic search.

    MUST match email embedding dimensions (3072 for text-embedding-3-large).
    Supports page-level and chunk-level embeddings for multi-page documents.
    """
    __tablename__ = "document_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)

    # Embedding (same dimension as email_embeddings!)
    # Using pgvector's Vector type for native PostgreSQL vector operations
    if HAS_PGVECTOR:
        embedding = Column(Vector(3072), nullable=False)  # text-embedding-3-large
    else:
        # Fallback for development if pgvector not installed
        from sqlalchemy import JSON
        embedding = Column(JSON, nullable=False)

    # Page-level granularity (especially useful for scanned docs)
    page_number = Column(Integer)           # NULL for non-paginated, 1-indexed for pages

    # Within-page chunking (for very long pages)
    chunk_index = Column(Integer, default=0)  # 0 = first/only chunk on page
    chunk_start = Column(Integer)              # Character offset within page
    chunk_end = Column(Integer)
    chunk_text = Column(EncryptedText)         # Encrypted chunk text

    # Model info (must match email embedding model)
    model = Column(String(100), nullable=False, default='text-embedding-3-large')
    model_version = Column(String(50))

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # Relationships
    document = relationship("Document", back_populates="embeddings")

    __table_args__ = (
        Index('idx_document_embeddings_document', 'document_id'),
        # Unique per document + page + chunk
        Index(
            'uq_document_embeddings_page_chunk',
            'document_id', 'page_number', 'chunk_index',
            unique=True
        ),
        # Vector index will be created via migration:
        # CREATE INDEX idx_document_embeddings_vector ON document_embeddings USING diskann (embedding);
    )

    def __repr__(self) -> str:
        return f"<DocumentEmbedding(id={self.id}, doc={self.document_id}, page={self.page_number}, chunk={self.chunk_index})>"


class DocumentProcessingQueue(Base):
    """
    Processing queue for async extraction/embedding.

    Enables:
    - Async processing of documents
    - Retry logic for failed extractions
    - Priority-based processing
    - Worker coordination
    """
    __tablename__ = "document_processing_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'))

    # Task type
    task_type = Column(String(50), nullable=False)  # 'extract_text', 'generate_embedding', 'classify'
    priority = Column(Integer, default=5)           # Lower = higher priority

    # Status
    status = Column(String(20), default='pending')  # 'pending', 'processing', 'completed', 'failed'
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    last_error = Column(Text)

    # Scheduling
    scheduled_at = Column(DateTime(timezone=True), default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    worker_id = Column(String(100))

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # Relationships
    document = relationship("Document", back_populates="processing_tasks")

    __table_args__ = (
        Index('idx_processing_queue_document', 'document_id'),
        Index('idx_processing_queue_status', 'status'),
        Index('idx_processing_queue_task_type', 'task_type'),
        # Index for finding pending tasks by priority
        Index(
            'idx_processing_queue_pending',
            'priority', 'scheduled_at',
            postgresql_where=(status == 'pending')
        ),
    )

    def __repr__(self) -> str:
        return f"<DocumentProcessingQueue(id={self.id}, task={self.task_type}, status={self.status})>"
