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
import tempfile
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

    def __init__(
        self,
        db: Session,
        embedding_model: str = "text-embedding-3-large",
    ):
        """
        Initialize attachment indexer.

        Args:
            db: Database session
            embedding_model: Model for generating embeddings (default: text-embedding-3-large)
        """
        self.db = db
        self.repo = DocumentRepository(db)
        self.embedding_model = embedding_model

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
            self._embedding_service = DocumentEmbeddingService(self.repo, model=self.embedding_model)
        return self._embedding_service

    def is_indexable(self, filename: str, content_type: Optional[str] = None) -> bool:
        """
        Check if a file can be indexed (has extractable text).

        Args:
            filename: Filename with extension
            content_type: MIME type

        Returns:
            True if file can be indexed
        """
        if content_type and content_type in EXTRACTABLE_TYPES:
            return True

        if filename:
            ext = os.path.splitext(filename.lower())[1]
            return ext in EXTRACTABLE_EXTENSIONS

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

            if not self.is_indexable(info.filename, info.content_type):
                logger.debug(f"Skipping non-indexable attachment {info.filename}")
                continue

            try:
                result = await self._index_single_attachment(
                    email=email,
                    info=info,
                    content=content,
                    attachment_index=idx,
                    skip_if_exists=skip_if_exists,
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

        # Extract text if needed
        needs_extraction = is_new or document.extraction_status in (None, 'pending', ExtractionStatus.PENDING.value)
        if needs_extraction or not skip_if_exists:
            await self._extract_and_store_text(document, content, info)

        # Refresh document after extraction
        document = await self.repo.get_by_id(document.id)

        # Generate embedding if we have text
        if document and document.extracted_text and document.extraction_status == ExtractionStatus.SUCCESS.value:
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

        Uses a temporary file approach since extractors typically work with files.
        """
        # Create a temporary file to write the content
        suffix = os.path.splitext(info.filename)[1] if info.filename else ''
        if not suffix and info.content_type:
            suffix = EXTRACTABLE_TYPES.get(info.content_type, '')

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name

            try:
                # Use the processor's extractor
                if self.processor.extractor:
                    extraction_result = await self.processor.extractor.extract_text(tmp_path)

                    if extraction_result and extraction_result.text:
                        await self.repo.update_extraction(
                            document_id=document.id,
                            extracted_text=extraction_result.text,
                            extraction_status=ExtractionStatus.SUCCESS,
                            extraction_method=extraction_result.method,
                            extraction_quality=extraction_result.quality_score,
                            page_count=extraction_result.page_count,
                        )
                        logger.info(f"Extracted {len(extraction_result.text)} chars from {info.filename}")
                    else:
                        await self.repo.update_extraction(
                            document_id=document.id,
                            extracted_text='',
                            extraction_status=ExtractionStatus.NO_CONTENT,
                            extraction_method='sandboxed',
                        )
                        logger.debug(f"No text content in {info.filename}")
                else:
                    # Fallback: try simple text extraction for plain text files
                    if info.content_type in ('text/plain', 'text/csv') or suffix in ('.txt', '.csv'):
                        text = content.decode('utf-8', errors='replace')
                        await self.repo.update_extraction(
                            document_id=document.id,
                            extracted_text=text,
                            extraction_status=ExtractionStatus.SUCCESS,
                            extraction_method='direct',
                        )
                        logger.info(f"Extracted {len(text)} chars from {info.filename} (direct)")
                    else:
                        await self.repo.update_extraction(
                            document_id=document.id,
                            extracted_text=None,
                            extraction_status=ExtractionStatus.FAILED,
                            extraction_method='none',
                        )
                        logger.warning(f"No extractor available for {info.filename}")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            error_msg = str(e)[:500]
            logger.error(f"Extraction failed for {info.filename}: {e}")
            await self.repo.update_extraction(
                document_id=document.id,
                extracted_text=None,
                extraction_status=ExtractionStatus.FAILED,
                extraction_method='error',
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
            logger.info(f"Generated embedding for document {document.id}: {document.filename} ({result.embeddings_generated} embeddings)")

        except Exception as e:
            logger.error(f"Embedding generation failed for {document.id}: {e}")

    def get_stats(self) -> dict:
        """Get indexing statistics."""
        # Use sync version if available
        try:
            return self.repo.get_stats()
        except Exception:
            return {}
