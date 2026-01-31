"""
Document Processor

Handles document registration, text extraction, and checksum calculation.
Reuses existing SandboxedExtractor for text extraction from supported formats.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from uuid import UUID

from backend.core.documents.models import Document, ExtractionStatus
from backend.core.documents.repository import DocumentRepository

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of text extraction from a document."""
    text: Optional[str]
    method: str  # 'sandboxed', 'pdftotext', 'tesseract', 'claude', 'none'
    quality_score: float  # 0.0-1.0
    page_count: Optional[int] = None
    extraction_model: Optional[str] = None
    extraction_cost: float = 0.0


class DocumentProcessor:
    """
    Processor for registering and extracting text from documents.

    Handles:
    - Checksum calculation for deduplication
    - Document registration with origin tracking
    - Text extraction using sandboxed extractor
    - Title/summary derivation from content
    """

    def __init__(self, repository: DocumentRepository):
        self.repository = repository
        self._extractor = None

    @property
    def extractor(self):
        """Lazy load the sandboxed extractor."""
        if self._extractor is None:
            try:
                from backend.core.email.sandboxed_extractor import SandboxedExtractor
                self._extractor = SandboxedExtractor()
            except ImportError:
                logger.warning("SandboxedExtractor not available")
                self._extractor = None
        return self._extractor

    def calculate_checksum(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex-encoded SHA-256 checksum

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in 64KB chunks for memory efficiency
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def calculate_checksum_from_bytes(self, content: bytes) -> str:
        """
        Calculate SHA-256 checksum from bytes.

        Args:
            content: File content as bytes

        Returns:
            Hex-encoded SHA-256 checksum
        """
        return hashlib.sha256(content).hexdigest()

    def get_mime_type(self, file_path: str) -> Optional[str]:
        """
        Get MIME type from file path.

        Uses file extension to determine type.
        """
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type

    async def register_document(
        self,
        file_path: str,
        origin_type: str,
        origin_host: Optional[str] = None,
        email_id: Optional[UUID] = None,
        attachment_index: Optional[int] = None,
        file_content: Optional[bytes] = None,
    ) -> Tuple[Document, bool]:
        """
        Register a document, creating new or adding origin to existing.

        Args:
            file_path: Path to the file (or virtual path for attachments)
            origin_type: Type of origin ('folder', 'email_attachment', 'google_drive')
            origin_host: Host where file is located
            email_id: Email ID if origin is email attachment
            attachment_index: Attachment index if origin is email attachment
            file_content: Optional file content (for attachments not on disk)

        Returns:
            Tuple of (Document, is_new) where is_new is True if newly created

        Raises:
            FileNotFoundError: If file doesn't exist and no content provided
        """
        # Get file info
        path = Path(file_path)
        filename = path.name

        if file_content is not None:
            # Content provided directly (e.g., email attachment)
            checksum = self.calculate_checksum_from_bytes(file_content)
            file_size = len(file_content)
            file_modified_at = None
        else:
            # Read from filesystem
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            checksum = self.calculate_checksum(file_path)
            file_size = path.stat().st_size
            file_modified_at = datetime.fromtimestamp(path.stat().st_mtime)

        mime_type = self.get_mime_type(file_path)

        # Get or create document
        document, is_new = await self.repository.get_or_create(
            checksum=checksum,
            file_size=file_size,
            mime_type=mime_type,
            original_filename=filename,
        )

        # Add origin
        await self.repository.add_origin(
            document_id=document.id,
            origin_type=origin_type,
            origin_host=origin_host,
            origin_path=file_path,
            origin_filename=filename,
            email_id=email_id,
            attachment_index=attachment_index,
            file_modified_at=file_modified_at,
            is_primary=is_new,  # First origin is primary
        )

        if is_new:
            # Queue for extraction
            await self.repository.queue_task(
                document_id=document.id,
                task_type="extract_text",
                priority=5,
            )
            logger.info(f"Registered new document: {filename} ({checksum[:16]}...)")
        else:
            logger.info(f"Added origin to existing document: {filename} ({checksum[:16]}...)")

        return document, is_new

    async def extract_text(
        self,
        document: Document,
        file_content: bytes,
    ) -> ExtractionResult:
        """
        Extract text from document content.

        Uses the existing SandboxedExtractor for supported formats.

        Args:
            document: Document to extract text from
            file_content: Binary content of the file

        Returns:
            ExtractionResult with text and metadata
        """
        if self.extractor is None:
            logger.error("SandboxedExtractor not available")
            return ExtractionResult(
                text=None,
                method="none",
                quality_score=0.0,
            )

        try:
            # Use existing sandboxed extractor
            text = await self.extractor.extract_text(
                content=file_content,
                content_type=document.mime_type or "application/octet-stream",
                filename=document.original_filename or "unknown",
            )

            if text and text.strip():
                quality_score = self._score_quality(text)
                return ExtractionResult(
                    text=text,
                    method="sandboxed",
                    quality_score=quality_score,
                    page_count=self._estimate_page_count(text),
                )
            else:
                return ExtractionResult(
                    text=None,
                    method="sandboxed",
                    quality_score=1.0,  # Extraction worked, just no content
                )
        except Exception as e:
            logger.error(f"Extraction failed for document {document.id}: {e}")
            return ExtractionResult(
                text=None,
                method="sandboxed",
                quality_score=0.0,
            )

    def _score_quality(self, text: str) -> float:
        """
        Score the quality of extracted text.

        Heuristics:
        - Garbled text patterns (random characters)
        - Word validity (proportion of dictionary-like words)
        - Character diversity

        Returns:
            Quality score 0.0-1.0
        """
        if not text or not text.strip():
            return 0.0

        # Check text length
        if len(text) < 10:
            return 0.3

        # Check for excessive special characters (sign of garbled OCR)
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / len(text)
        if special_ratio > 0.3:
            return 0.4

        # Check word patterns
        words = text.split()
        if len(words) < 5:
            return 0.5

        # Check for reasonable word lengths
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2 or avg_word_len > 20:
            return 0.5

        # Check for repeated garbage patterns
        unique_chars = len(set(text.lower()))
        if unique_chars < 10:
            return 0.3

        # Good quality indicators
        # - Has sentences (periods followed by capital)
        # - Has paragraphs (multiple newlines)
        # - Normal character distribution
        return min(1.0, 0.7 + (0.3 * (unique_chars / 50)))

    def _estimate_page_count(self, text: str) -> int:
        """
        Estimate page count from text length.

        Uses rough approximation of ~3000 characters per page.
        """
        if not text:
            return 0
        return max(1, len(text) // 3000)

    def derive_title(self, document: Document) -> str:
        """
        Derive a title for the document.

        Priority:
        1. First line of extracted text (if short and capitalized)
        2. Cleaned filename
        """
        # Try to get title from extracted text
        if document.extracted_text:
            first_line = document.extracted_text.split('\n')[0].strip()
            # Check if it looks like a title (short, capitalized)
            if len(first_line) < 100 and first_line and first_line[0].isupper():
                return first_line

        # Fall back to filename
        if document.original_filename:
            # Remove extension and clean up
            name = Path(document.original_filename).stem
            # Replace underscores/dashes with spaces
            name = name.replace('_', ' ').replace('-', ' ')
            # Title case
            return name.title()

        return "Untitled Document"

    def derive_summary(self, document: Document, max_length: int = 200) -> str:
        """
        Derive a one-line summary for the document.

        Uses the first paragraph of extracted text.
        """
        if not document.extracted_text:
            return f"Document: {document.original_filename or 'Unknown'}"

        # Get first paragraph
        text = document.extracted_text.strip()
        paragraphs = text.split('\n\n')
        first_para = paragraphs[0].replace('\n', ' ').strip()

        if len(first_para) <= max_length:
            return first_para

        # Truncate at word boundary
        truncated = first_para[:max_length].rsplit(' ', 1)[0]
        return truncated + "..."
