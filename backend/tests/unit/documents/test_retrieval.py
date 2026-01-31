"""
Tests for Document Retrieval Service.

Phase 2 tests for:
- Local filesystem retrieval
- SSH/SCP retrieval
- IMAP attachment retrieval
- Origin accessibility checking
- Fallback to secondary origins
"""

import pytest
import tempfile
import os
from pathlib import Path
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock, patch
import subprocess

from backend.core.documents.retrieval import (
    DocumentRetrievalService,
    RetrievalError,
    OriginNotAccessibleError,
)
from backend.core.documents.models import Document, DocumentOrigin
from backend.core.documents.config import HostConfig


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = MagicMock()
    repo.get_origins = AsyncMock(return_value=[])
    repo.update_origin_verified = AsyncMock()
    return repo


@pytest.fixture
def sample_document():
    """Create a sample document."""
    doc = Document(
        checksum="abc123",
        file_size=1024,
        mime_type="application/pdf",
        original_filename="test.pdf",
    )
    doc.id = uuid4()
    return doc


@pytest.fixture
def local_origin(sample_document, temp_file):
    """Create a local filesystem origin."""
    origin = DocumentOrigin(
        document_id=sample_document.id,
        origin_type="folder",
        origin_host="localhost",
        origin_path=str(temp_file.parent),
        origin_filename=temp_file.name,
        is_primary=True,
        is_deleted=False,
    )
    origin.id = uuid4()
    return origin


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
        f.write(b"Test PDF content")
        path = Path(f.name)

    yield path

    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def retrieval_service(mock_repository):
    """Create a retrieval service."""
    return DocumentRetrievalService(mock_repository)


class TestLocalFilesystemRetrieval:
    """Tests for local filesystem retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_local_file(self, retrieval_service, sample_document, local_origin, mock_repository, temp_file):
        """Should retrieve content from local file."""
        mock_repository.get_origins.return_value = [local_origin]

        content = await retrieval_service.get_content(sample_document)

        assert content == b"Test PDF content"

    @pytest.mark.asyncio
    async def test_retrieve_file_not_found(self, retrieval_service, sample_document, mock_repository):
        """Should raise error when file not found."""
        origin = DocumentOrigin(
            document_id=sample_document.id,
            origin_type="folder",
            origin_host="localhost",
            origin_path="/nonexistent",
            origin_filename="missing.pdf",
            is_primary=True,
            is_deleted=False,
        )
        origin.id = uuid4()
        mock_repository.get_origins.return_value = [origin]

        with pytest.raises(RetrievalError):
            await retrieval_service.get_content(sample_document)

    @pytest.mark.asyncio
    async def test_retrieve_no_origins(self, retrieval_service, sample_document, mock_repository):
        """Should raise error when no origins available."""
        mock_repository.get_origins.return_value = []

        with pytest.raises(RetrievalError, match="No origins available"):
            await retrieval_service.get_content(sample_document)

    @pytest.mark.asyncio
    async def test_retrieve_with_origin_index(self, retrieval_service, sample_document, mock_repository):
        """Should try specified origin first."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            f.write(b"Secondary content")
            secondary_path = Path(f.name)

        try:
            primary = DocumentOrigin(
                document_id=sample_document.id,
                origin_type="folder",
                origin_host="localhost",
                origin_path="/nonexistent",
                origin_filename="missing.pdf",
                is_primary=True,
                is_deleted=False,
            )
            primary.id = uuid4()

            secondary = DocumentOrigin(
                document_id=sample_document.id,
                origin_type="folder",
                origin_host="localhost",
                origin_path=str(secondary_path.parent),
                origin_filename=secondary_path.name,
                is_primary=False,
                is_deleted=False,
            )
            secondary.id = uuid4()

            mock_repository.get_origins.return_value = [primary, secondary]

            # Request secondary origin
            content = await retrieval_service.get_content(sample_document, origin_index=1)

            assert content == b"Secondary content"
        finally:
            secondary_path.unlink()

    @pytest.mark.asyncio
    async def test_retrieve_fallback_to_secondary(self, retrieval_service, sample_document, mock_repository):
        """Should fall back to secondary origin if primary fails."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            f.write(b"Fallback content")
            fallback_path = Path(f.name)

        try:
            primary = DocumentOrigin(
                document_id=sample_document.id,
                origin_type="folder",
                origin_host="localhost",
                origin_path="/nonexistent",
                origin_filename="missing.pdf",
                is_primary=True,
                is_deleted=False,
            )
            primary.id = uuid4()

            secondary = DocumentOrigin(
                document_id=sample_document.id,
                origin_type="folder",
                origin_host="localhost",
                origin_path=str(fallback_path.parent),
                origin_filename=fallback_path.name,
                is_primary=False,
                is_deleted=False,
            )
            secondary.id = uuid4()

            mock_repository.get_origins.return_value = [primary, secondary]

            # Should fall back to secondary
            content = await retrieval_service.get_content(sample_document)

            assert content == b"Fallback content"
        finally:
            fallback_path.unlink()

    @pytest.mark.asyncio
    async def test_retrieve_no_fallback(self, retrieval_service, sample_document, mock_repository):
        """Should not fall back when fallback=False."""
        primary = DocumentOrigin(
            document_id=sample_document.id,
            origin_type="folder",
            origin_host="localhost",
            origin_path="/nonexistent",
            origin_filename="missing.pdf",
            is_primary=True,
            is_deleted=False,
        )
        primary.id = uuid4()

        mock_repository.get_origins.return_value = [primary]

        with pytest.raises(RetrievalError):
            await retrieval_service.get_content(sample_document, fallback=False)


class TestSSHRetrieval:
    """Tests for SSH/SCP retrieval."""

    @pytest.mark.asyncio
    async def test_ssh_retrieval_success(self, retrieval_service, sample_document, mock_repository):
        """Should retrieve via SCP."""
        origin = DocumentOrigin(
            document_id=sample_document.id,
            origin_type="folder",
            origin_host="ssh-server",
            origin_path="/remote/path",
            origin_filename="file.pdf",
            is_primary=True,
            is_deleted=False,
        )
        origin.id = uuid4()
        mock_repository.get_origins.return_value = [origin]

        # Mock host config
        with patch("backend.core.documents.retrieval.get_host_config") as mock_get_config:
            mock_get_config.return_value = HostConfig(
                name="ssh-server",
                type="ssh",
                ssh_host="server.example.com",
                ssh_user="testuser",
                ssh_port=22,
            )

            # Mock subprocess
            with patch("subprocess.run") as mock_run:
                # Create temp file to simulate SCP result
                with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                    f.write(b"SSH content")
                    temp_path = Path(f.name)

                try:
                    mock_run.return_value = MagicMock(returncode=0)

                    # Patch temp file read to return our content
                    with patch.object(Path, "read_bytes", return_value=b"SSH content"):
                        with patch.object(Path, "exists", return_value=True):
                            with patch.object(Path, "unlink"):
                                content = await retrieval_service.get_content(sample_document)

                    assert content == b"SSH content"
                finally:
                    if temp_path.exists():
                        temp_path.unlink()

    @pytest.mark.asyncio
    async def test_ssh_retrieval_failure(self, retrieval_service, sample_document, mock_repository):
        """Should handle SCP failure."""
        origin = DocumentOrigin(
            document_id=sample_document.id,
            origin_type="folder",
            origin_host="ssh-server",
            origin_path="/remote/path",
            origin_filename="file.pdf",
            is_primary=True,
            is_deleted=False,
        )
        origin.id = uuid4()
        mock_repository.get_origins.return_value = [origin]

        with patch("backend.core.documents.retrieval.get_host_config") as mock_get_config:
            mock_get_config.return_value = HostConfig(
                name="ssh-server",
                type="ssh",
                ssh_host="server.example.com",
                ssh_user="testuser",
            )

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1,
                    stderr=b"Connection refused"
                )

                with pytest.raises(RetrievalError):
                    await retrieval_service.get_content(sample_document)


class TestIMAPRetrieval:
    """Tests for IMAP attachment retrieval."""

    @pytest.mark.asyncio
    async def test_imap_retrieval_missing_email_id(self, retrieval_service, sample_document, mock_repository):
        """Should raise error when email_id not set."""
        origin = DocumentOrigin(
            document_id=sample_document.id,
            origin_type="email_attachment",
            origin_host=None,
            email_id=None,  # Missing
            attachment_index=0,
            is_primary=True,
            is_deleted=False,
        )
        origin.id = uuid4()
        mock_repository.get_origins.return_value = [origin]

        with pytest.raises(RetrievalError, match="Email ID not set"):
            await retrieval_service.get_content(sample_document)

    @pytest.mark.asyncio
    async def test_imap_retrieval_missing_index(self, retrieval_service, sample_document, mock_repository):
        """Should raise error when attachment_index not set."""
        origin = DocumentOrigin(
            document_id=sample_document.id,
            origin_type="email_attachment",
            origin_host=None,
            email_id=uuid4(),
            attachment_index=None,  # Missing
            is_primary=True,
            is_deleted=False,
        )
        origin.id = uuid4()
        mock_repository.get_origins.return_value = [origin]

        with pytest.raises(RetrievalError, match="Attachment index not set"):
            await retrieval_service.get_content(sample_document)


class TestOriginAccessibility:
    """Tests for origin accessibility checking."""

    @pytest.mark.asyncio
    async def test_check_local_accessible(self, retrieval_service, local_origin):
        """Should return True for accessible local file."""
        result = await retrieval_service.check_origin_accessible(local_origin)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_local_not_accessible(self, retrieval_service):
        """Should return False for missing file."""
        origin = DocumentOrigin(
            document_id=uuid4(),
            origin_type="folder",
            origin_host="localhost",
            origin_path="/nonexistent",
            origin_filename="missing.pdf",
            is_primary=True,
            is_deleted=False,
        )
        origin.id = uuid4()

        result = await retrieval_service.check_origin_accessible(origin)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_unknown_type_not_accessible(self, retrieval_service):
        """Should return False for unknown origin type."""
        origin = DocumentOrigin(
            document_id=uuid4(),
            origin_type="unknown_type",
            origin_host="localhost",
            is_primary=True,
            is_deleted=False,
        )
        origin.id = uuid4()

        result = await retrieval_service.check_origin_accessible(origin)
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_all_origins(self, retrieval_service, sample_document, mock_repository, temp_file):
        """Should verify all origins and return status."""
        accessible = DocumentOrigin(
            document_id=sample_document.id,
            origin_type="folder",
            origin_host="localhost",
            origin_path=str(temp_file.parent),
            origin_filename=temp_file.name,
            is_primary=True,
            is_deleted=False,
        )
        accessible.id = uuid4()

        inaccessible = DocumentOrigin(
            document_id=sample_document.id,
            origin_type="folder",
            origin_host="localhost",
            origin_path="/nonexistent",
            origin_filename="missing.pdf",
            is_primary=False,
            is_deleted=False,
        )
        inaccessible.id = uuid4()

        mock_repository.get_origins.return_value = [accessible, inaccessible]

        results = await retrieval_service.verify_all_origins(sample_document)

        assert str(accessible.id) in results
        assert str(inaccessible.id) in results
        assert results[str(accessible.id)]["accessible"] is True
        assert results[str(inaccessible.id)]["accessible"] is False


class TestContentType:
    """Tests for content type detection."""

    def test_get_content_type_from_mime(self, retrieval_service):
        """Should use document's mime_type."""
        doc = Document(
            checksum="abc",
            file_size=100,
            mime_type="application/pdf",
        )

        content_type = retrieval_service.get_content_type(doc)
        assert content_type == "application/pdf"

    def test_get_content_type_from_extension(self, retrieval_service):
        """Should infer from filename when no mime_type."""
        doc = Document(
            checksum="abc",
            file_size=100,
            original_filename="report.xlsx",
        )

        content_type = retrieval_service.get_content_type(doc)
        assert "spreadsheet" in content_type.lower() or "excel" in content_type.lower()

    def test_get_content_type_fallback(self, retrieval_service):
        """Should return octet-stream as fallback."""
        doc = Document(
            checksum="abc",
            file_size=100,
        )

        content_type = retrieval_service.get_content_type(doc)
        assert content_type == "application/octet-stream"

    def test_get_content_type_various_extensions(self, retrieval_service):
        """Should handle various file extensions."""
        test_cases = [
            ("doc.pdf", "application/pdf"),
            ("image.jpg", "image/jpeg"),
            ("image.jpeg", "image/jpeg"),
            ("photo.png", "image/png"),
            ("data.csv", "text/csv"),
            ("notes.txt", "text/plain"),
        ]

        for filename, expected_type in test_cases:
            doc = Document(
                checksum="abc",
                file_size=100,
                original_filename=filename,
            )
            content_type = retrieval_service.get_content_type(doc)
            assert content_type == expected_type, f"Failed for {filename}"
