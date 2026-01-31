"""
Tests for Document Processor.

Phase 1 tests for:
- Checksum calculation
- Document registration
- Text extraction
- Title/summary derivation
"""

import pytest
import uuid
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock, PropertyMock

from backend.core.documents.models import Document, ExtractionStatus
from backend.core.documents.processor import DocumentProcessor, ExtractionResult


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = MagicMock()
    repo.get_or_create = AsyncMock()
    repo.add_origin = AsyncMock()
    repo.queue_task = AsyncMock()
    repo.update_extraction = AsyncMock()
    return repo


@pytest.fixture
def processor(mock_repository):
    """Create a DocumentProcessor with mock repository."""
    return DocumentProcessor(mock_repository)


class TestChecksumCalculation:
    """Tests for checksum calculation."""

    def test_calculate_checksum(self, processor):
        """calculate_checksum should return correct SHA-256 hash."""
        # Create a temp file with known content
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b"Hello, World!")
            temp_path = f.name

        try:
            checksum = processor.calculate_checksum(temp_path)
            # Known SHA-256 of "Hello, World!"
            expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
            assert checksum == expected
        finally:
            os.unlink(temp_path)

    def test_calculate_checksum_from_bytes(self, processor):
        """calculate_checksum_from_bytes should return correct SHA-256 hash."""
        content = b"Hello, World!"
        checksum = processor.calculate_checksum_from_bytes(content)
        expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        assert checksum == expected

    def test_calculate_checksum_empty_file(self, processor):
        """calculate_checksum should handle empty files."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            # Empty file
            temp_path = f.name

        try:
            checksum = processor.calculate_checksum(temp_path)
            # SHA-256 of empty string
            expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            assert checksum == expected
        finally:
            os.unlink(temp_path)

    def test_calculate_checksum_file_not_found(self, processor):
        """calculate_checksum should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            processor.calculate_checksum("/nonexistent/path/file.pdf")

    def test_calculate_checksum_large_file(self, processor):
        """calculate_checksum should handle large files efficiently."""
        # Create a 1MB file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b"x" * (1024 * 1024))
            temp_path = f.name

        try:
            checksum = processor.calculate_checksum(temp_path)
            assert len(checksum) == 64  # SHA-256 hex is 64 chars
        finally:
            os.unlink(temp_path)


class TestDocumentRegistration:
    """Tests for document registration."""

    @pytest.mark.asyncio
    async def test_register_new_document(self, processor, mock_repository):
        """register_document should create new document for new file."""
        # Create a temp file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            f.write(b"PDF content here")
            temp_path = f.name

        try:
            doc = Document(
                checksum="abc123",
                file_size=16,
                mime_type="application/pdf",
            )
            doc.id = uuid.uuid4()
            mock_repository.get_or_create.return_value = (doc, True)
            mock_repository.add_origin.return_value = Mock()
            mock_repository.queue_task.return_value = Mock()

            result_doc, is_new = await processor.register_document(
                file_path=temp_path,
                origin_type="folder",
                origin_host="localhost",
            )

            assert is_new is True
            mock_repository.get_or_create.assert_called_once()
            mock_repository.add_origin.assert_called_once()
            mock_repository.queue_task.assert_called_once()  # Queued for extraction
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_register_duplicate_document(self, processor, mock_repository):
        """register_document should add origin for existing document."""
        # Create a temp file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            f.write(b"PDF content here")
            temp_path = f.name

        try:
            existing_doc = Document(
                checksum="abc123",
                file_size=16,
            )
            existing_doc.id = uuid.uuid4()
            mock_repository.get_or_create.return_value = (existing_doc, False)
            mock_repository.add_origin.return_value = Mock()

            result_doc, is_new = await processor.register_document(
                file_path=temp_path,
                origin_type="folder",
                origin_host="nas.local",
            )

            assert is_new is False
            mock_repository.queue_task.assert_not_called()  # Not queued again
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_register_document_from_bytes(self, processor, mock_repository):
        """register_document should accept file content directly."""
        doc = Document(checksum="abc123", file_size=100)
        doc.id = uuid.uuid4()
        mock_repository.get_or_create.return_value = (doc, True)
        mock_repository.add_origin.return_value = Mock()
        mock_repository.queue_task.return_value = Mock()

        content = b"File content here"
        result_doc, is_new = await processor.register_document(
            file_path="/virtual/path/attachment.pdf",
            origin_type="email_attachment",
            email_id=uuid.uuid4(),
            attachment_index=0,
            file_content=content,
        )

        assert is_new is True
        # Checksum should be calculated from content
        call_args = mock_repository.get_or_create.call_args
        assert call_args.kwargs['file_size'] == len(content)

    @pytest.mark.asyncio
    async def test_register_document_file_not_found(self, processor, mock_repository):
        """register_document should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            await processor.register_document(
                file_path="/nonexistent/file.pdf",
                origin_type="folder",
            )


class TestTextExtraction:
    """Tests for text extraction."""

    @pytest.mark.asyncio
    async def test_extract_text_from_pdf(self, processor):
        """extract_text should use sandboxed extractor for PDF."""
        # Mock the extractor
        mock_extractor = AsyncMock()
        mock_extractor.extract_text.return_value = "Extracted PDF content with enough words to pass quality check."
        processor._extractor = mock_extractor

        doc = Document(
            checksum="abc123",
            file_size=1000,
            mime_type="application/pdf",
            original_filename="test.pdf",
        )

        result = await processor.extract_text(doc, b"PDF binary content")

        assert result.text is not None
        assert result.method == "sandboxed"
        assert result.quality_score > 0
        mock_extractor.extract_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_text_empty_result(self, processor):
        """extract_text should handle empty extraction result."""
        mock_extractor = AsyncMock()
        mock_extractor.extract_text.return_value = ""
        processor._extractor = mock_extractor

        doc = Document(
            checksum="abc123",
            file_size=1000,
            mime_type="application/pdf",
        )

        result = await processor.extract_text(doc, b"PDF content")

        assert result.text is None
        assert result.quality_score == 1.0  # Extraction worked, just no content

    @pytest.mark.asyncio
    async def test_extract_text_extractor_not_available(self, processor):
        """extract_text should handle missing extractor gracefully."""
        # Set a sentinel value to prevent lazy loading
        processor._extractor = "UNAVAILABLE"

        # Override the property to return None
        with patch.object(DocumentProcessor, 'extractor', property(lambda self: None)):
            doc = Document(
                checksum="abc123",
                file_size=1000,
            )

            result = await processor.extract_text(doc, b"content")

            assert result.text is None
            assert result.method == "none"
            assert result.quality_score == 0.0

    @pytest.mark.asyncio
    async def test_handle_extraction_error(self, processor):
        """extract_text should handle extraction errors gracefully."""
        mock_extractor = AsyncMock()
        mock_extractor.extract_text.side_effect = Exception("Extraction failed")
        processor._extractor = mock_extractor

        doc = Document(
            checksum="abc123",
            file_size=1000,
            original_filename="test.pdf",
        )
        doc.id = uuid.uuid4()

        result = await processor.extract_text(doc, b"content")

        assert result.text is None
        assert result.quality_score == 0.0


class TestQualityScoring:
    """Tests for quality scoring."""

    def test_score_quality_good_text(self, processor):
        """Good text should score high."""
        good_text = """
        This is a well-formatted document with proper sentences.
        It contains multiple paragraphs and uses standard English.
        The text quality should be considered high.
        """
        score = processor._score_quality(good_text)
        assert score >= 0.7

    def test_score_quality_empty_text(self, processor):
        """Empty text should score 0."""
        score = processor._score_quality("")
        assert score == 0.0

        score = processor._score_quality("   ")
        assert score == 0.0

    def test_score_quality_short_text(self, processor):
        """Very short text should score low."""
        score = processor._score_quality("Hi")
        assert score <= 0.5

    def test_score_quality_garbled_text(self, processor):
        """Garbled text with many special characters should score low."""
        garbled = "!@#$%^&*(){}[]|\\:;<>,.?/~`"
        score = processor._score_quality(garbled)
        assert score <= 0.5

    def test_score_quality_repeated_chars(self, processor):
        """Text with few unique characters should score low."""
        repetitive = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        score = processor._score_quality(repetitive)
        assert score <= 0.5


class TestTitleDerivation:
    """Tests for title derivation."""

    def test_derive_title_from_text(self, processor):
        """derive_title should use first line if it looks like a title."""
        doc = Document(
            checksum="abc",
            file_size=100,
            original_filename="document.pdf",
        )
        doc.extracted_text = "Quarterly Financial Report\n\nThis document contains..."

        title = processor.derive_title(doc)
        assert title == "Quarterly Financial Report"

    def test_derive_title_from_filename(self, processor):
        """derive_title should fall back to filename."""
        doc = Document(
            checksum="abc",
            file_size=100,
            original_filename="quarterly_report_2024.pdf",
        )
        doc.extracted_text = None

        title = processor.derive_title(doc)
        assert title == "Quarterly Report 2024"

    def test_derive_title_with_dashes(self, processor):
        """derive_title should handle dashed filenames."""
        doc = Document(
            checksum="abc",
            file_size=100,
            original_filename="meeting-notes-january.docx",
        )
        doc.extracted_text = None

        title = processor.derive_title(doc)
        assert title == "Meeting Notes January"

    def test_derive_title_no_info(self, processor):
        """derive_title should return default when no info available."""
        doc = Document(
            checksum="abc",
            file_size=100,
        )
        doc.extracted_text = None

        title = processor.derive_title(doc)
        assert title == "Untitled Document"


class TestSummaryDerivation:
    """Tests for summary derivation."""

    def test_derive_summary_short_text(self, processor):
        """derive_summary should use first paragraph."""
        doc = Document(
            checksum="abc",
            file_size=100,
        )
        doc.extracted_text = "This is a short summary.\n\nMore content here."

        summary = processor.derive_summary(doc)
        assert summary == "This is a short summary."

    def test_derive_summary_long_text(self, processor):
        """derive_summary should truncate long text."""
        doc = Document(
            checksum="abc",
            file_size=100,
        )
        doc.extracted_text = "A" * 500 + "\n\nSecond paragraph."

        summary = processor.derive_summary(doc, max_length=100)
        assert len(summary) <= 103  # 100 + "..."
        assert summary.endswith("...")

    def test_derive_summary_no_text(self, processor):
        """derive_summary should use filename when no text."""
        doc = Document(
            checksum="abc",
            file_size=100,
            original_filename="report.pdf",
        )
        doc.extracted_text = None

        summary = processor.derive_summary(doc)
        assert "report.pdf" in summary


class TestMimeTypeDetection:
    """Tests for MIME type detection."""

    def test_get_mime_type_pdf(self, processor):
        """get_mime_type should detect PDF."""
        mime = processor.get_mime_type("/path/to/document.pdf")
        assert mime == "application/pdf"

    def test_get_mime_type_docx(self, processor):
        """get_mime_type should detect DOCX."""
        mime = processor.get_mime_type("/path/to/document.docx")
        assert mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def test_get_mime_type_unknown(self, processor):
        """get_mime_type should return None for unknown types."""
        mime = processor.get_mime_type("/path/to/file.xyz123")
        assert mime is None

    def test_get_mime_type_txt(self, processor):
        """get_mime_type should detect text files."""
        mime = processor.get_mime_type("/path/to/notes.txt")
        assert mime == "text/plain"
