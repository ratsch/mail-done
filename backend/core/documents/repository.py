"""
Document Repository

Provides CRUD operations for documents, origins, embeddings, and processing queue.
Follows the same patterns as EmailRepository for consistency.
"""

import logging
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, update, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.core.documents.models import (
    Document,
    DocumentOrigin,
    DocumentEmbedding,
    DocumentProcessingQueue,
    ExtractionStatus,
)

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository for document CRUD operations.

    Handles:
    - Document creation with checksum deduplication
    - Origin management (multiple origins per document)
    - Extraction status updates
    - Processing queue operations
    - Embedding management
    """

    def __init__(self, db: Session):
        self.db = db

    # =========================================================================
    # Document CRUD
    # =========================================================================

    async def create_document(
        self,
        checksum: str,
        file_size: int,
        mime_type: Optional[str] = None,
        original_filename: Optional[str] = None,
        checksum_algorithm: str = "sha256",
    ) -> Document:
        """
        Create a new document.

        Args:
            checksum: SHA-256 checksum of file content
            file_size: Size of file in bytes
            mime_type: MIME type of file
            original_filename: Original filename
            checksum_algorithm: Algorithm used for checksum (default: sha256)

        Returns:
            Created Document instance

        Raises:
            IntegrityError: If document with same checksum already exists
        """
        document = Document(
            checksum=checksum,
            checksum_algorithm=checksum_algorithm,
            file_size=file_size,
            mime_type=mime_type,
            original_filename=original_filename,
            extraction_status=ExtractionStatus.PENDING.value,
        )
        self.db.add(document)
        self.db.flush()  # Get the ID without committing
        logger.info(f"Created document {document.id} with checksum {checksum[:16]}...")
        return document

    async def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Get document by ID."""
        return self.db.get(Document, document_id)

    async def get_by_checksum(self, checksum: str) -> Optional[Document]:
        """Get document by checksum (for deduplication)."""
        stmt = select(Document).where(Document.checksum == checksum)
        result = self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_or_create(
        self,
        checksum: str,
        file_size: int,
        mime_type: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> tuple[Document, bool]:
        """
        Get existing document or create new one.

        Handles race conditions where multiple processes try to create
        the same document simultaneously by catching IntegrityError
        and fetching the existing document.

        Args:
            checksum: SHA-256 checksum
            file_size: File size in bytes
            mime_type: MIME type
            original_filename: Original filename

        Returns:
            Tuple of (Document, is_new) where is_new is True if created
        """
        existing = await self.get_by_checksum(checksum)
        if existing:
            # Update last_seen_at
            existing.last_seen_at = datetime.utcnow()
            return existing, False

        try:
            document = await self.create_document(
                checksum=checksum,
                file_size=file_size,
                mime_type=mime_type,
                original_filename=original_filename,
            )
            return document, True
        except IntegrityError:
            # Race condition: another process created the document
            # Roll back the failed insert and fetch the existing document
            self.db.rollback()
            existing = await self.get_by_checksum(checksum)
            if existing:
                existing.last_seen_at = datetime.utcnow()
                return existing, False
            # This shouldn't happen, but re-raise if it does
            raise

    async def update_extraction(
        self,
        document_id: UUID,
        extracted_text: Optional[str],
        extraction_status: ExtractionStatus,
        extraction_method: Optional[str] = None,
        extraction_quality: Optional[float] = None,
        extraction_model: Optional[str] = None,
        extraction_cost: Optional[float] = None,
        page_count: Optional[int] = None,
        title: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> Optional[Document]:
        """
        Update document with extraction results.

        Args:
            document_id: Document UUID
            extracted_text: Extracted text content (will be encrypted)
            extraction_status: New extraction status
            extraction_method: Method used ('sandboxed', 'tesseract', 'claude', etc.)
            extraction_quality: Quality score 0.0-1.0
            extraction_model: Model used if applicable
            extraction_cost: API cost in USD if applicable
            page_count: Number of pages
            title: Extracted or derived title
            summary: One-line summary

        Returns:
            Updated Document or None if not found
        """
        document = await self.get_by_id(document_id)
        if not document:
            return None

        document.extracted_text = extracted_text
        document.extraction_status = extraction_status.value
        document.extraction_method = extraction_method
        document.extraction_quality = extraction_quality
        document.extraction_model = extraction_model
        document.extraction_cost = extraction_cost
        document.extracted_at = datetime.utcnow()

        if page_count is not None:
            document.page_count = page_count
        if title is not None:
            document.title = title
        if summary is not None:
            document.summary = summary

        logger.info(f"Updated extraction for document {document_id}: status={extraction_status.value}")
        return document

    async def set_extraction_status(
        self,
        document_id: UUID,
        status: ExtractionStatus,
    ) -> bool:
        """
        Set extraction status for a document.

        Returns:
            True if document was updated, False if not found
        """
        stmt = (
            update(Document)
            .where(Document.id == document_id)
            .values(extraction_status=status.value)
        )
        result = self.db.execute(stmt)
        return result.rowcount > 0

    async def get_documents_needing_extraction(
        self,
        limit: int = 100,
    ) -> List[Document]:
        """Get documents that need text extraction."""
        stmt = (
            select(Document)
            .where(Document.extraction_status.in_([
                ExtractionStatus.PENDING.value,
                ExtractionStatus.FAILED.value,
            ]))
            .order_by(Document.created_at)
            .limit(limit)
        )
        result = self.db.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Origin Management
    # =========================================================================

    async def add_origin(
        self,
        document_id: UUID,
        origin_type: str,
        origin_host: Optional[str] = None,
        origin_path: Optional[str] = None,
        origin_filename: Optional[str] = None,
        email_id: Optional[UUID] = None,
        attachment_index: Optional[int] = None,
        file_modified_at: Optional[datetime] = None,
        is_primary: bool = False,
    ) -> Optional[DocumentOrigin]:
        """
        Add an origin to a document.

        If the same origin already exists, returns None (no duplicate).

        Args:
            document_id: Document UUID
            origin_type: Type of origin ('folder', 'email_attachment', 'google_drive')
            origin_host: Host where file is located
            origin_path: Full path to file
            origin_filename: Filename at this location
            email_id: Email ID if origin is email attachment
            attachment_index: Attachment index if origin is email attachment
            file_modified_at: File modification time
            is_primary: Whether this is the primary origin

        Returns:
            Created DocumentOrigin or None if duplicate
        """
        # Check for existing origin with same location
        existing = await self.get_origin(
            document_id=document_id,
            origin_type=origin_type,
            origin_host=origin_host,
            origin_path=origin_path,
            email_id=email_id,
            attachment_index=attachment_index,
        )
        if existing:
            # Update last_verified_at if already exists
            existing.last_verified_at = datetime.utcnow()
            if file_modified_at:
                existing.file_modified_at = file_modified_at
            return None  # Not a new origin

        origin = DocumentOrigin(
            document_id=document_id,
            origin_type=origin_type,
            origin_host=origin_host,
            origin_path=origin_path,
            origin_filename=origin_filename,
            email_id=email_id,
            attachment_index=attachment_index,
            file_modified_at=file_modified_at,
            is_primary=is_primary,
        )
        self.db.add(origin)
        self.db.flush()
        logger.info(f"Added origin {origin_type}:{origin_host}:{origin_path} to document {document_id}")
        return origin

    async def get_origin(
        self,
        document_id: UUID,
        origin_type: str,
        origin_host: Optional[str] = None,
        origin_path: Optional[str] = None,
        email_id: Optional[UUID] = None,
        attachment_index: Optional[int] = None,
    ) -> Optional[DocumentOrigin]:
        """Get a specific origin by its identifying attributes."""
        conditions = [
            DocumentOrigin.document_id == document_id,
            DocumentOrigin.origin_type == origin_type,
        ]

        if origin_type == 'email_attachment' and email_id:
            conditions.append(DocumentOrigin.email_id == email_id)
            if attachment_index is not None:
                conditions.append(DocumentOrigin.attachment_index == attachment_index)
        else:
            if origin_host is not None:
                conditions.append(DocumentOrigin.origin_host == origin_host)
            if origin_path is not None:
                conditions.append(DocumentOrigin.origin_path == origin_path)

        stmt = select(DocumentOrigin).where(and_(*conditions))
        result = self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_origins(
        self,
        document_id: UUID,
        include_deleted: bool = False,
    ) -> List[DocumentOrigin]:
        """Get all origins for a document."""
        stmt = select(DocumentOrigin).where(DocumentOrigin.document_id == document_id)
        if not include_deleted:
            stmt = stmt.where(DocumentOrigin.is_deleted == False)
        stmt = stmt.order_by(DocumentOrigin.is_primary.desc(), DocumentOrigin.discovered_at)
        result = self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_origins_by_email(self, email_id: UUID) -> List[DocumentOrigin]:
        """Get all document origins for an email (its attachments)."""
        stmt = (
            select(DocumentOrigin)
            .where(DocumentOrigin.email_id == email_id)
            .where(DocumentOrigin.is_deleted == False)
            .order_by(DocumentOrigin.attachment_index)
        )
        result = self.db.execute(stmt)
        return list(result.scalars().all())

    async def mark_origin_deleted(
        self,
        origin_id: Optional[UUID] = None,
        origin_type: Optional[str] = None,
        origin_host: Optional[str] = None,
        origin_path: Optional[str] = None,
        origin_filename: Optional[str] = None,
    ) -> bool:
        """
        Mark an origin as deleted (soft delete).

        Can be called with origin_id, or with origin attributes.

        Args:
            origin_id: Origin UUID (if known)
            origin_type: Type of origin ('folder', 'email_attachment', etc.)
            origin_host: Host name
            origin_path: Path to file directory
            origin_filename: Filename

        Returns:
            True if origin was found and marked deleted
        """
        if origin_id:
            stmt = (
                update(DocumentOrigin)
                .where(DocumentOrigin.id == origin_id)
                .values(is_deleted=True, deleted_at=datetime.utcnow())
            )
        else:
            # Find by attributes
            conditions = []
            if origin_type:
                conditions.append(DocumentOrigin.origin_type == origin_type)
            if origin_host:
                conditions.append(DocumentOrigin.origin_host == origin_host)
            if origin_path:
                conditions.append(DocumentOrigin.origin_path == origin_path)
            if origin_filename:
                conditions.append(DocumentOrigin.origin_filename == origin_filename)

            if not conditions:
                return False

            stmt = (
                update(DocumentOrigin)
                .where(and_(*conditions))
                .where(DocumentOrigin.is_deleted == False)
                .values(is_deleted=True, deleted_at=datetime.utcnow())
            )

        result = self.db.execute(stmt)
        return result.rowcount > 0

    async def update_origin_verified(
        self,
        origin_id: UUID,
    ) -> bool:
        """Update last_verified_at timestamp for an origin."""
        stmt = (
            update(DocumentOrigin)
            .where(DocumentOrigin.id == origin_id)
            .values(last_verified_at=datetime.utcnow())
        )
        result = self.db.execute(stmt)
        return result.rowcount > 0

    # =========================================================================
    # Processing Queue
    # =========================================================================

    async def queue_task(
        self,
        document_id: UUID,
        task_type: str,
        priority: int = 5,
    ) -> DocumentProcessingQueue:
        """
        Add a task to the processing queue.

        Args:
            document_id: Document to process
            task_type: Type of task ('extract_text', 'generate_embedding', 'classify')
            priority: Priority (lower = higher priority, default 5)

        Returns:
            Created queue entry
        """
        task = DocumentProcessingQueue(
            document_id=document_id,
            task_type=task_type,
            priority=priority,
            status='pending',
        )
        self.db.add(task)
        self.db.flush()
        logger.debug(f"Queued {task_type} task for document {document_id}")
        return task

    async def get_pending_tasks(
        self,
        task_type: str,
        limit: int = 10,
    ) -> List[DocumentProcessingQueue]:
        """
        Get pending tasks of a specific type, ordered by priority.

        Args:
            task_type: Type of task to fetch
            limit: Maximum number of tasks to return

        Returns:
            List of pending tasks
        """
        stmt = (
            select(DocumentProcessingQueue)
            .where(DocumentProcessingQueue.task_type == task_type)
            .where(DocumentProcessingQueue.status == 'pending')
            .where(DocumentProcessingQueue.scheduled_at <= datetime.utcnow())
            .order_by(
                DocumentProcessingQueue.priority,
                DocumentProcessingQueue.scheduled_at,
            )
            .limit(limit)
        )
        result = self.db.execute(stmt)
        return list(result.scalars().all())

    async def mark_task_processing(
        self,
        task_id: UUID,
        worker_id: str,
    ) -> bool:
        """
        Mark a task as processing (claim it for a worker).

        Uses optimistic locking to prevent race conditions.

        Returns:
            True if successfully claimed, False if already claimed
        """
        stmt = (
            update(DocumentProcessingQueue)
            .where(DocumentProcessingQueue.id == task_id)
            .where(DocumentProcessingQueue.status == 'pending')
            .values(
                status='processing',
                started_at=datetime.utcnow(),
                worker_id=worker_id,
                attempts=DocumentProcessingQueue.attempts + 1,
            )
        )
        result = self.db.execute(stmt)
        return result.rowcount > 0

    async def mark_task_completed(
        self,
        task_id: UUID,
    ) -> bool:
        """Mark a task as completed."""
        stmt = (
            update(DocumentProcessingQueue)
            .where(DocumentProcessingQueue.id == task_id)
            .values(
                status='completed',
                completed_at=datetime.utcnow(),
            )
        )
        result = self.db.execute(stmt)
        return result.rowcount > 0

    async def mark_task_failed(
        self,
        task_id: UUID,
        error: str,
    ) -> bool:
        """
        Mark a task as failed.

        If max_attempts not reached, sets status back to 'pending' for retry.
        """
        # Get current task state
        task = self.db.get(DocumentProcessingQueue, task_id)
        if not task:
            return False

        if task.attempts >= task.max_attempts:
            # Max attempts reached, mark as failed
            task.status = 'failed'
            task.last_error = error
            task.completed_at = datetime.utcnow()
            logger.warning(f"Task {task_id} failed after {task.attempts} attempts: {error}")
        else:
            # Retry - set back to pending
            task.status = 'pending'
            task.last_error = error
            task.worker_id = None
            task.started_at = None
            logger.info(f"Task {task_id} will retry (attempt {task.attempts}/{task.max_attempts}): {error}")

        return True

    async def get_task(self, task_id: UUID) -> Optional[DocumentProcessingQueue]:
        """Get a task by ID."""
        return self.db.get(DocumentProcessingQueue, task_id)

    # =========================================================================
    # Embedding Management
    # =========================================================================

    async def add_embedding(
        self,
        document_id: UUID,
        embedding: List[float],
        page_number: Optional[int] = None,
        chunk_index: int = 0,
        chunk_start: Optional[int] = None,
        chunk_end: Optional[int] = None,
        chunk_text: Optional[str] = None,
        model: str = 'text-embedding-3-large',
        model_version: Optional[str] = None,
    ) -> DocumentEmbedding:
        """
        Add an embedding for a document.

        Args:
            document_id: Document UUID
            embedding: Vector embedding (3072 dimensions)
            page_number: Page number (None for whole document)
            chunk_index: Chunk index within page (0 for first/only)
            chunk_start: Character offset start
            chunk_end: Character offset end
            chunk_text: Text that was embedded (will be encrypted)
            model: Embedding model name
            model_version: Model version

        Returns:
            Created DocumentEmbedding
        """
        doc_embedding = DocumentEmbedding(
            document_id=document_id,
            embedding=embedding,
            page_number=page_number,
            chunk_index=chunk_index,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            chunk_text=chunk_text,
            model=model,
            model_version=model_version,
        )
        self.db.add(doc_embedding)
        self.db.flush()
        return doc_embedding

    async def get_embeddings(
        self,
        document_id: UUID,
    ) -> List[DocumentEmbedding]:
        """Get all embeddings for a document."""
        stmt = (
            select(DocumentEmbedding)
            .where(DocumentEmbedding.document_id == document_id)
            .order_by(
                DocumentEmbedding.page_number.nullsfirst(),
                DocumentEmbedding.chunk_index,
            )
        )
        result = self.db.execute(stmt)
        return list(result.scalars().all())

    async def delete_embeddings(
        self,
        document_id: UUID,
    ) -> int:
        """Delete all embeddings for a document (for re-embedding)."""
        stmt = (
            DocumentEmbedding.__table__.delete()
            .where(DocumentEmbedding.document_id == document_id)
        )
        result = self.db.execute(stmt)
        return result.rowcount

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict:
        """Get document indexing statistics."""
        from sqlalchemy import func

        # Total documents
        total = self.db.execute(
            select(func.count(Document.id))
        ).scalar() or 0

        # By extraction status
        status_counts = self.db.execute(
            select(Document.extraction_status, func.count(Document.id))
            .group_by(Document.extraction_status)
        ).all()
        by_status = {status: count for status, count in status_counts}

        # By document type
        type_counts = self.db.execute(
            select(Document.document_type, func.count(Document.id))
            .where(Document.document_type.isnot(None))
            .group_by(Document.document_type)
        ).all()
        by_type = {dtype: count for dtype, count in type_counts}

        # Total embeddings
        total_embeddings = self.db.execute(
            select(func.count(DocumentEmbedding.id))
        ).scalar() or 0

        # Pending tasks
        pending_tasks = self.db.execute(
            select(DocumentProcessingQueue.task_type, func.count(DocumentProcessingQueue.id))
            .where(DocumentProcessingQueue.status == 'pending')
            .group_by(DocumentProcessingQueue.task_type)
        ).all()
        pending_by_type = {task_type: count for task_type, count in pending_tasks}

        return {
            'total_documents': total,
            'by_extraction_status': by_status,
            'by_document_type': by_type,
            'total_embeddings': total_embeddings,
            'pending_tasks_by_type': pending_by_type,
        }
