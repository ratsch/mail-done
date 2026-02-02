"""
Attachment Indexer for indexing email attachments as searchable documents.

Integrates with the email processing pipeline to extract, deduplicate,
and index attachments for unified multi-source search.

Uses the existing DocumentProcessor and DocumentEmbeddingService for
extraction and embedding generation.
"""
import hashlib
import logging
import os
from typing import List, Optional, Tuple
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from backend.core.documents.models import Document, ExtractionStatus
from backend.core.documents.repository import DocumentRepository
from backend.core.email.models import AttachmentInfo

logger = logging.getLogger(__name__)

# Supported file types for text extraction
EXTRACTABLE_TYPES = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/rtf': '.rtf',
    'text/plain': '.txt',
    'text/csv': '.csv',
    'text/calendar': '.ics',
}

# File extensions that can be extracted
EXTRACTABLE_EXTENSIONS = {'.pdf', '.docx', '.xlsx', '.pptx', '.rtf', '.txt', '.csv', '.ics', '.ical'}

# Image types that may contain real content (screenshots, diagrams, photos)
# Only indexed if size > IMAGE_MIN_SIZE_BYTES to exclude small logos/icons
IMAGE_TYPES = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/tiff': '.tiff',
    'image/bmp': '.bmp',
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.tiff', '.tif', '.bmp'}

# Minimum size for images to be indexed (100KB) - smaller images are likely logos/icons
IMAGE_MIN_SIZE_BYTES = 100 * 1024  # 100KB


class AttachmentIndexer:
    """
    Index email attachments as searchable documents.

    Workflow:
    1. Receive attachment content from email processing
    2. Calculate content hash for deduplication
    3. Create/get Document record
    4. Add email attachment origin
    5. Extract text content (if not already extracted)
    6. Generate embedding for search
    """

    def __init__(self, db: Session):
        """
        Initialize attachment indexer.

        Args:
            db: Database session
        """
        self.db = db
        self.repo = DocumentRepository(db)

        # Lazy-load processor and embedding service
        self._processor = None
        self._embedding_service = None

    @property
    def processor(self):
        """Lazy load document processor."""
        if self._processor is None:
            from backend.core.documents.processor import DocumentProcessor
            self._processor = DocumentProcessor(self.repo)
        return self._processor

    @property
    def embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            from backend.core.documents.embeddings import DocumentEmbeddingService
            self._embedding_service = DocumentEmbeddingService(self.repo, single_embedding=True)
        return self._embedding_service

    def is_indexable(self, filename: str, content_type: Optional[str] = None, size: Optional[int] = None) -> bool:
        """
        Check if a file can be indexed (has extractable text or is a large image for OCR).

        Args:
            filename: Filename with extension
            content_type: MIME type
            size: File size in bytes (required for image size check)

        Returns:
            True if file can be indexed
        """
        # Check text-extractable types first
        if content_type and content_type in EXTRACTABLE_TYPES:
            return True

        if filename:
            ext = os.path.splitext(filename.lower())[1]
            if ext in EXTRACTABLE_EXTENSIONS:
                return True

        # Check for large images (may contain real content for OCR)
        is_image = False
        if content_type and content_type in IMAGE_TYPES:
            is_image = True
        elif filename:
            ext = os.path.splitext(filename.lower())[1]
            if ext in IMAGE_EXTENSIONS:
                is_image = True

        if is_image and size is not None and size >= IMAGE_MIN_SIZE_BYTES:
            return True

        return False

    def is_image_for_ocr(self, filename: str, content_type: Optional[str] = None) -> bool:
        """
        Check if a file is an image type that would need OCR.

        Args:
            filename: Filename with extension
            content_type: MIME type

        Returns:
            True if file is an image type
        """
        if content_type and content_type in IMAGE_TYPES:
            return True

        if filename:
            ext = os.path.splitext(filename.lower())[1]
            return ext in IMAGE_EXTENSIONS

        return False

    def _calculate_checksum(self, content: bytes) -> str:
        """Calculate SHA-256 checksum of content."""
        return hashlib.sha256(content).hexdigest()

    async def index_email_attachments(
        self,
        email,  # Email model from database
        attachment_infos: List[AttachmentInfo],
        attachment_contents: List[bytes],
        skip_if_exists: bool = True,
    ) -> List[Tuple[Document, bool]]:
        """
        Index all indexable attachments from an email.

        Args:
            email: Database Email model
            attachment_infos: List of AttachmentInfo from email processing
            attachment_contents: Raw attachment bytes (same order as infos)
            skip_if_exists: If True, skip text extraction for existing documents

        Returns:
            List of (Document, is_new) tuples for indexed documents
        """
        results = []

        if len(attachment_infos) != len(attachment_contents):
            logger.error(
                f"Mismatch: {len(attachment_infos)} infos vs {len(attachment_contents)} contents"
            )
            return results

        for idx, (info, content) in enumerate(zip(attachment_infos, attachment_contents)):
            if not content:
                logger.debug(f"Skipping empty attachment {info.filename}")
                continue

            content_size = len(content)
            if not self.is_indexable(info.filename, info.content_type, size=content_size):
                logger.debug(f"Skipping non-indexable attachment {info.filename} ({content_size} bytes)")
                continue

            try:
                # Check if this is an image (for OCR marking)
                is_image = self.is_image_for_ocr(info.filename, info.content_type)

                result = await self._index_single_attachment(
                    email=email,
                    info=info,
                    content=content,
                    attachment_index=idx,
                    skip_if_exists=skip_if_exists,
                    is_image=is_image,
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed to index attachment {info.filename}: {e}")
                continue

        return results

    async def _index_single_attachment(
        self,
        email,
        info: AttachmentInfo,
        content: bytes,
        attachment_index: int,
        skip_if_exists: bool = True,
        is_image: bool = False,
    ) -> Optional[Tuple[Document, bool]]:
        """
        Index a single attachment.

        Args:
            email: Parent email
            info: Attachment info
            content: Raw file bytes
            attachment_index: Index in email's attachment list
            skip_if_exists: Skip extraction if document exists

        Returns:
            (Document, is_new) tuple or None if failed
        """
        # Calculate checksum
        checksum = self._calculate_checksum(content)

        # Get or create document (handles deduplication)
        document, is_new = await self.repo.get_or_create(
            checksum=checksum,
            file_size=len(content),
            mime_type=info.content_type,
            original_filename=info.filename,
        )

        # Add origin linking to this email
        await self.repo.add_origin(
            document_id=document.id,
            origin_type='email_attachment',
            email_id=email.id,
            attachment_index=attachment_index,
            origin_filename=info.filename,
        )

        # Extract text if needed (or mark for OCR if image)
        needs_extraction = is_new or document.extraction_status in (None, 'pending', ExtractionStatus.PENDING.value)
        if needs_extraction or not skip_if_exists:
            if is_image:
                # Large images are indexed but need OCR later
                await self.repo.update_extraction(
                    document_id=document.id,
                    extracted_text=None,
                    extraction_status=ExtractionStatus.NEEDS_OCR,
                    extraction_method='pending_ocr',
                )
                logger.info(f"Indexed image {info.filename} ({len(content)} bytes) - marked for OCR")
            else:
                await self._extract_and_store_text(document, content, info)

        # Refresh document after extraction
        document = await self.repo.get_by_id(document.id)

        # Generate embedding if we have text
        if document and document.extracted_text and document.extraction_status == ExtractionStatus.COMPLETED.value:
            await self._generate_embedding(document)

        return document, is_new

    async def _extract_and_store_text(
        self,
        document: Document,
        content: bytes,
        info: AttachmentInfo,
    ):
        """
        Extract text from document content and store in database.

        Uses the same processing as folder_scanner via processor.process_document_content().
        """
        try:
            result = await self.processor.process_document_content(
                document=document,
                file_content=content,
                use_structure=False,  # Simple single embedding for attachments
            )

            if result and result.text:
                logger.info(f"Extracted {len(result.text)} chars from {info.filename}")
            else:
                logger.debug(f"No text content in {info.filename}")

        except Exception as e:
            logger.error(f"Extraction failed for {info.filename}: {e}")
            await self.repo.update_extraction(
                document_id=document.id,
                extracted_text=None,
                extraction_status=ExtractionStatus.FAILED,
                extraction_method='error',
                extraction_quality=0.0,
            )

    async def _generate_embedding(self, document: Document):
        """
        Generate and store vector embedding for document.

        Uses the DocumentEmbeddingService for embedding generation.
        """
        if not document.extracted_text:
            logger.debug(f"No text to embed for document {document.id}")
            return

        try:
            result = await self.embedding_service.generate_document_embedding(
                document,
                single_embedding=True,  # Simple single embedding for attachments
            )
            logger.info(f"Generated embedding for document {document.id}: {document.original_filename} ({result.embeddings_created} embeddings)")

        except Exception as e:
            logger.error(f"Embedding generation failed for {document.id}: {e}")

    def get_stats(self) -> dict:
        """Get indexing statistics."""
        # Use sync version if available
        try:
            return self.repo.get_stats()
        except Exception:
            return {}
