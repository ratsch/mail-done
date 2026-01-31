"""
Document Indexing Module

Provides document storage, extraction, and semantic search capabilities.
Documents are first-class objects with checksum-based deduplication
and origin tracking.

Phase 1: Documents as First-Class Objects
- SQLAlchemy models for documents, origins, embeddings, processing queue
- Repository for CRUD operations
- Document processor for registration and extraction
- Embedding service for vector generation
"""

from backend.core.documents.models import (
    Document,
    DocumentOrigin,
    DocumentEmbedding,
    DocumentProcessingQueue,
    ExtractionStatus,
)
from backend.core.documents.repository import DocumentRepository
from backend.core.documents.processor import DocumentProcessor, ExtractionResult

__all__ = [
    "Document",
    "DocumentOrigin",
    "DocumentEmbedding",
    "DocumentProcessingQueue",
    "ExtractionStatus",
    "DocumentRepository",
    "DocumentProcessor",
    "ExtractionResult",
]
