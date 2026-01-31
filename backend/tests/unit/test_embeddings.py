"""
Tests for embedding generation, specifically attachment text inclusion.

Phase 0 of document indexing: Include attachment text in email embeddings
to enable semantic search over attachment content.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from backend.core.search.embeddings import (
    EmbeddingGenerator,
    MAX_ATTACHMENT_TEXT_CHARS,
    MAX_ATTACHMENTS_FOR_EMBEDDING,
)


@pytest.fixture
def mock_email():
    """Create a mock Email database object."""
    email = Mock()
    email.id = "test-email-id"
    email.subject = "Test Email Subject"
    email.from_name = "Test Sender"
    email.from_address = "sender@example.com"
    email.body_markdown = "This is the email body."
    email.body_text = "This is the email body."
    email.attachment_info = None
    return email


@pytest.fixture
def mock_metadata():
    """Create a mock EmailMetadata object."""
    metadata = Mock()
    metadata.ai_category = "work"
    metadata.ai_summary = "A test email about work."
    return metadata


class TestEmbeddingAttachmentText:
    """Tests for attachment text inclusion in embeddings."""

    def test_attachment_text_included_in_embedding(self, mock_email):
        """Attachment text should be included in embedding input."""
        mock_email.attachment_info = [
            {
                "filename": "report.pdf",
                "content_type": "application/pdf",
                "size": 1024,
                "extracted_text": "This is the PDF content about quarterly results.",
                "extraction_error": None,
            }
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, None)

        assert "quarterly results" in text.lower()
        assert "[Attachment: report.pdf]" in text

    def test_multiple_attachments_included(self, mock_email):
        """Multiple attachments should all be included."""
        mock_email.attachment_info = [
            {
                "filename": "doc1.pdf",
                "extracted_text": "First document content.",
            },
            {
                "filename": "doc2.docx",
                "extracted_text": "Second document content.",
            },
            {
                "filename": "data.xlsx",
                "extracted_text": "Spreadsheet data here.",
            },
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, None)

        assert "First document content" in text
        assert "Second document content" in text
        assert "Spreadsheet data here" in text
        assert "[Attachment: doc1.pdf]" in text
        assert "[Attachment: doc2.docx]" in text
        assert "[Attachment: data.xlsx]" in text

    def test_attachment_text_truncated(self, mock_email):
        """Long attachment text should be truncated."""
        long_text = "A" * (MAX_ATTACHMENT_TEXT_CHARS + 1000)
        mock_email.attachment_info = [
            {
                "filename": "large.pdf",
                "extracted_text": long_text,
            }
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, None)

        # Should be truncated with ellipsis
        assert "..." in text
        # Should not contain the full text
        assert long_text not in text

    def test_max_attachments_limit(self, mock_email):
        """Only first N attachments should be included."""
        mock_email.attachment_info = [
            {"filename": f"doc{i}.pdf", "extracted_text": f"Content {i}"}
            for i in range(MAX_ATTACHMENTS_FOR_EMBEDDING + 3)
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, None)

        # Should include first MAX_ATTACHMENTS_FOR_EMBEDDING
        for i in range(MAX_ATTACHMENTS_FOR_EMBEDDING):
            assert f"Content {i}" in text

        # Should NOT include attachments beyond the limit
        assert f"Content {MAX_ATTACHMENTS_FOR_EMBEDDING}" not in text
        assert f"Content {MAX_ATTACHMENTS_FOR_EMBEDDING + 1}" not in text

    def test_no_attachments_still_works(self, mock_email, mock_metadata):
        """Emails without attachments should work as before."""
        mock_email.attachment_info = None

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, mock_metadata)

        assert "Test Email Subject" in text
        assert "email body" in text
        assert "Attachment" not in text

    def test_empty_attachment_info_list(self, mock_email):
        """Empty attachment_info list should work."""
        mock_email.attachment_info = []

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, None)

        assert "Test Email Subject" in text
        assert "Attachment" not in text

    def test_attachment_without_extracted_text(self, mock_email):
        """Attachments without extracted text should be skipped."""
        mock_email.attachment_info = [
            {
                "filename": "image.png",
                "content_type": "image/png",
                "size": 2048,
                "extracted_text": None,
                "extraction_error": "Unsupported format",
            },
            {
                "filename": "doc.pdf",
                "extracted_text": "This PDF was extracted.",
            },
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, None)

        # Should skip the image but include the PDF
        assert "[Attachment: image.png]" not in text
        assert "[Attachment: doc.pdf]" in text
        assert "This PDF was extracted" in text

    def test_attachment_with_empty_text(self, mock_email):
        """Attachments with empty/whitespace text should be skipped."""
        mock_email.attachment_info = [
            {
                "filename": "empty.pdf",
                "extracted_text": "",
            },
            {
                "filename": "whitespace.pdf",
                "extracted_text": "   \n\t  ",
            },
            {
                "filename": "valid.pdf",
                "extracted_text": "Valid content here.",
            },
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, None)

        assert "[Attachment: empty.pdf]" not in text
        assert "[Attachment: whitespace.pdf]" not in text
        assert "[Attachment: valid.pdf]" in text
        assert "Valid content here" in text

    def test_attachment_info_not_list(self, mock_email):
        """Non-list attachment_info should be handled gracefully."""
        mock_email.attachment_info = "invalid"

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, None)

        # Should not crash, should just skip attachments
        assert "Test Email Subject" in text

    def test_attachment_info_with_non_dict_items(self, mock_email):
        """Non-dict items in attachment_info should be skipped."""
        mock_email.attachment_info = [
            "invalid",
            None,
            {"filename": "valid.pdf", "extracted_text": "Valid content."},
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, None)

        assert "[Attachment: valid.pdf]" in text
        assert "Valid content" in text

    def test_metadata_still_included_with_attachments(self, mock_email, mock_metadata):
        """AI metadata should still be included when attachments present."""
        mock_email.attachment_info = [
            {"filename": "doc.pdf", "extracted_text": "PDF content."}
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        text = generator._prepare_text(mock_email, mock_metadata)

        # All parts should be present
        assert "Test Email Subject" in text
        assert "Category: work" in text
        assert "Summary: A test email about work" in text
        assert "[Attachment: doc.pdf]" in text
        assert "PDF content" in text


class TestExtractAttachmentTexts:
    """Tests for the _extract_attachment_texts helper method."""

    def test_extract_from_valid_attachments(self, mock_email):
        """Should extract text from valid attachments."""
        mock_email.attachment_info = [
            {"filename": "a.pdf", "extracted_text": "Content A"},
            {"filename": "b.docx", "extracted_text": "Content B"},
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        texts = generator._extract_attachment_texts(mock_email)

        assert len(texts) == 2
        assert "[Attachment: a.pdf]" in texts[0]
        assert "Content A" in texts[0]
        assert "[Attachment: b.docx]" in texts[1]
        assert "Content B" in texts[1]

    def test_extract_returns_empty_for_no_attachments(self, mock_email):
        """Should return empty list when no attachments."""
        mock_email.attachment_info = None

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        texts = generator._extract_attachment_texts(mock_email)

        assert texts == []

    def test_extract_skips_failed_extractions(self, mock_email):
        """Should skip attachments that failed extraction."""
        mock_email.attachment_info = [
            {"filename": "failed.pdf", "extracted_text": None, "extraction_error": "Failed"},
            {"filename": "success.pdf", "extracted_text": "Success content"},
        ]

        generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
        texts = generator._extract_attachment_texts(mock_email)

        assert len(texts) == 1
        assert "Success content" in texts[0]
