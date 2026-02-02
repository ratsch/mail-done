"""
Document Processor

Handles document registration, text extraction, and checksum calculation.
Reuses existing SandboxedExtractor for text extraction from supported formats.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
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
    # Structured data for per-page/per-sheet/per-section extraction
    pages: Optional[List[Dict[str, Any]]] = None  # [{"page": 1, "text": "..."}, ...]
    sheets: Optional[List[Dict[str, Any]]] = None  # [{"sheet": "Name", "sheet_index": 0, "text": "..."}, ...]
    sections: Optional[List[Dict[str, Any]]] = None  # [{"section": 0, "title": "...", "text": "..."}, ...]

    @property
    def has_structure(self) -> bool:
        """True if structured data (pages/sheets/sections) is available."""
        return bool(self.pages or self.sheets or self.sections)


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

    async def extract_with_structure(
        self,
        document: Document,
        file_content: bytes,
    ) -> ExtractionResult:
        """
        Extract text with page/sheet structure when available.

        For PDFs: returns per-page content
        For XLSX: returns per-sheet content
        For other formats: falls back to regular extraction

        Args:
            document: Document to extract text from
            file_content: Binary content of the file

        Returns:
            ExtractionResult with text and optional pages/sheets
        """
        if self.extractor is None:
            logger.error("SandboxedExtractor not available")
            return ExtractionResult(
                text=None,
                method="none",
                quality_score=0.0,
            )

        mime_type = document.mime_type or "application/octet-stream"
        filename = document.original_filename or "unknown"

        try:
            # Try structured extraction first
            structured_result = await self.extractor.extract_structured(
                content=file_content,
                content_type=mime_type,
                filename=filename,
            )

            if structured_result is not None and structured_result.success:
                # Determine if pages, sheets, or sections based on structure
                pages = None
                sheets = None
                sections = None

                if structured_result.items:
                    first_item = structured_result.items[0]
                    if "page" in first_item:
                        pages = structured_result.items
                    elif "sheet" in first_item:
                        sheets = structured_result.items
                    elif "section" in first_item:
                        sections = structured_result.items

                combined_text = structured_result.total_text
                quality_score = self._score_quality(combined_text) if combined_text else 0.0

                return ExtractionResult(
                    text=combined_text,
                    method="sandboxed_structured",
                    quality_score=quality_score,
                    page_count=structured_result.count,
                    pages=pages,
                    sheets=sheets,
                    sections=sections,
                )

            # Fall back to regular extraction for unsupported types
            return await self.extract_text(document, file_content)

        except Exception as e:
            logger.error(f"Structured extraction failed for document {document.id}: {e}")
            # Fall back to regular extraction
            return await self.extract_text(document, file_content)

    async def process_document_content(
        self,
        document: Document,
        file_content: bytes,
        use_structure: bool = False,
    ) -> ExtractionResult:
        """
        Process document content: analyze, extract text, and update database.

        This is the main entry point for processing document content. It:
        1. Analyzes content for OCR detection (has_images, ocr_recommended, etc.)
        2. Stores content analysis in database
        3. Extracts text (with structure if requested)
        4. Updates extraction status in database

        Use this method instead of extract_text() when you want the full
        processing workflow with content analysis and database updates.

        Args:
            document: Document to process
            file_content: Binary content of the file
            use_structure: If True, use structured extraction for multi-page docs

        Returns:
            ExtractionResult with text and metadata
        """
        from backend.core.documents.content_analyzer import analyze_content

        # Step 1: Analyze content for OCR detection
        content_analysis = analyze_content(
            file_content,
            document.mime_type,
            filename=document.original_filename,
        )

        # Step 2: Store content analysis
        await self.repository.update_content_analysis(
            document_id=document.id,
            has_images=content_analysis.has_images,
            has_native_text=content_analysis.has_native_text,
            is_image_only=content_analysis.is_image_only,
            is_scanned_with_ocr=content_analysis.is_scanned_with_ocr,
            ocr_recommended=content_analysis.ocr_recommended,
            text_source='native' if content_analysis.has_native_text else 'none',
        )

        # Step 3: Extract text
        if use_structure:
            extraction_result = await self.extract_with_structure(document, file_content)
        else:
            extraction_result = await self.extract_text(document, file_content)

        # Step 4: Update extraction status in database
        if extraction_result and extraction_result.text:
            await self.repository.update_extraction(
                document_id=document.id,
                extracted_text=extraction_result.text,
                extraction_status=ExtractionStatus.COMPLETED,
                extraction_method=extraction_result.method,
                extraction_quality=extraction_result.quality_score,
                page_count=extraction_result.page_count,
            )

            # Store structured data if available
            if extraction_result.has_structure:
                await self.repository.store_extraction_structure(
                    document_id=document.id,
                    pages=extraction_result.pages,
                    sheets=extraction_result.sheets,
                    sections=extraction_result.sections,
                )
        else:
            await self.repository.update_extraction(
                document_id=document.id,
                extracted_text=None,
                extraction_status=ExtractionStatus.NO_CONTENT,
                extraction_method=extraction_result.method if extraction_result else 'none',
                extraction_quality=0.0,
            )

        return extraction_result

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

    async def handle_file_change(
        self,
        origin_type: str,
        origin_host: str,
        file_path: str,
        new_checksum: str,
        file_size: int,
        mime_type: Optional[str] = None,
        file_modified_at: Optional[datetime] = None,
    ) -> Tuple[Optional[Document], bool, Optional[UUID]]:
        """
        Handle a file whose content has changed at a known path.

        Checks if we have an existing origin for this path:
        - If yes and checksum differs: update origin to point to new/existing document
        - If no: create new document and origin as usual

        Args:
            origin_type: Type of origin ('folder', etc.)
            origin_host: Host where file is located
            file_path: Full path to file
            new_checksum: New SHA-256 checksum
            file_size: New file size
            mime_type: MIME type
            file_modified_at: File modification time

        Returns:
            Tuple of (document, is_new_document, orphaned_document_id)
            - document: The document record (new or existing by checksum)
            - is_new_document: True if a new document was created
            - orphaned_document_id: ID of document that may have become orphaned
        """
        from pathlib import Path

        filename = Path(file_path).name
        orphaned_document_id = None

        # Check for existing origin at this path
        existing_origin = await self.repository.get_origin_by_path(
            origin_type=origin_type,
            origin_host=origin_host,
            origin_path=file_path,
        )

        if existing_origin:
            # Get the document this origin currently points to
            old_document = await self.repository.get_by_id(existing_origin.document_id)

            if old_document and old_document.checksum == new_checksum:
                # Checksum unchanged - just update verification time
                await self.repository.update_origin_verified(existing_origin.id)
                return old_document, False, None

            # Checksum changed - file content was modified
            logger.info(f"File content changed at {file_path}: {old_document.checksum[:16] if old_document else 'unknown'}... → {new_checksum[:16]}...")

            # Get or create document with new checksum
            new_document, is_new = await self.repository.get_or_create(
                checksum=new_checksum,
                file_size=file_size,
                mime_type=mime_type,
                original_filename=filename,
            )

            # Update origin to point to new document
            old_document_id = await self.repository.update_origin_document(
                origin_id=existing_origin.id,
                new_document_id=new_document.id,
            )

            # Check if old document became orphaned
            if old_document_id:
                was_orphaned = await self.repository.check_and_mark_orphaned(old_document_id)
                if was_orphaned:
                    orphaned_document_id = old_document_id

            # If document is new, queue for extraction
            if is_new:
                await self.repository.queue_task(
                    document_id=new_document.id,
                    task_type="extract_text",
                    priority=5,
                )

            # Un-orphan the new document if it was previously orphaned
            await self.repository.unorphan_document(new_document.id)

            return new_document, is_new, orphaned_document_id

        else:
            # No existing origin - standard registration
            document, is_new = await self.repository.get_or_create(
                checksum=new_checksum,
                file_size=file_size,
                mime_type=mime_type,
                original_filename=filename,
            )

            # Add new origin
            await self.repository.add_origin(
                document_id=document.id,
                origin_type=origin_type,
                origin_host=origin_host,
                origin_path=file_path,
                origin_filename=filename,
                file_modified_at=file_modified_at,
                is_primary=is_new,
            )

            if is_new:
                await self.repository.queue_task(
                    document_id=document.id,
                    task_type="extract_text",
                    priority=5,
                )

            # Un-orphan if this document was previously orphaned
            await self.repository.unorphan_document(document.id)

            return document, is_new, None

    async def provide_ocr_text(
        self,
        document_id: UUID,
        ocr_text: str,
        ocr_method: str = "ocr",
        ocr_quality: Optional[float] = None,
        ocr_structure: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
    ) -> Tuple[bool, bool]:
        """
        Provide OCR-extracted text for a document.

        Uses quality comparison to decide whether to update:
        - If existing extraction is empty/short, OCR wins
        - If force=True, OCR wins
        - Otherwise, compares quality scores

        Args:
            document_id: Document UUID
            ocr_text: OCR-extracted text
            ocr_method: OCR method used (e.g., 'ocr_tesseract', 'ocr_claude')
            ocr_quality: Quality score (0.0-1.0). If None, will be calculated.
            ocr_structure: Optional structure (pages from OCR)
            force: Force update regardless of quality

        Returns:
            Tuple of (was_updated, embeddings_queued)
        """
        # Calculate quality if not provided
        if ocr_quality is None:
            ocr_quality = self._score_quality(ocr_text)

        # Build structure dict if provided
        structure = None
        if ocr_structure:
            structure = {"pages": ocr_structure}

        return await self.repository.update_extraction_with_comparison(
            document_id=document_id,
            new_text=ocr_text,
            new_method=ocr_method,
            new_quality=ocr_quality,
            new_structure=structure,
            force=force,
        )
