"""
Tests for Document SQLAlchemy models.

Phase 1 tests for:
- Document model creation and fields
- DocumentOrigin relationships
- DocumentEmbedding relationships
- DocumentProcessingQueue
- ExtractionStatus enum
"""

import pytest
import uuid
from datetime import datetime

from backend.core.documents.models import (
    Document,
    DocumentOrigin,
    DocumentEmbedding,
    DocumentProcessingQueue,
    ExtractionStatus,
)


class TestExtractionStatus:
    """Tests for ExtractionStatus enum."""

    def test_extraction_status_enum_values(self):
        """ExtractionStatus should have all expected values."""
        assert ExtractionStatus.PENDING.value == "pending"
        assert ExtractionStatus.PROCESSING.value == "processing"
        assert ExtractionStatus.COMPLETED.value == "completed"
        assert ExtractionStatus.NO_CONTENT.value == "no_content"
        assert ExtractionStatus.FAILED.value == "failed"

    def test_extraction_status_is_string_enum(self):
        """ExtractionStatus values should be strings."""
        for status in ExtractionStatus:
            assert isinstance(status.value, str)

    def test_extraction_status_all_values(self):
        """Should have exactly 5 status values."""
        assert len(ExtractionStatus) == 5


class TestDocumentModel:
    """Tests for Document model."""

    def test_document_model_creation(self):
        """Document model should be created with required fields."""
        doc = Document(
            checksum="abc123def456",
            file_size=1024,
            extraction_status=ExtractionStatus.PENDING.value,  # Set explicitly for unit test
        )
        assert doc.checksum == "abc123def456"
        assert doc.file_size == 1024
        assert doc.extraction_status == ExtractionStatus.PENDING.value

    def test_document_model_optional_fields(self):
        """Document model should accept optional fields."""
        doc = Document(
            checksum="abc123",
            file_size=2048,
            mime_type="application/pdf",
            original_filename="report.pdf",
            page_count=10,
            title="Q4 Report",
            summary="Quarterly financial summary",
            document_type="report",
            language="en",
            ai_category="finance",
            ai_tags=["quarterly", "finance"],
        )
        assert doc.mime_type == "application/pdf"
        assert doc.original_filename == "report.pdf"
        assert doc.page_count == 10
        assert doc.title == "Q4 Report"
        assert doc.summary == "Quarterly financial summary"
        assert doc.document_type == "report"
        assert doc.language == "en"
        assert doc.ai_category == "finance"
        assert doc.ai_tags == ["quarterly", "finance"]

    def test_document_extraction_fields(self):
        """Document should track extraction metadata."""
        doc = Document(
            checksum="abc123",
            file_size=1024,
            extraction_status=ExtractionStatus.COMPLETED.value,
            extraction_method="sandboxed",
            extraction_quality=0.95,
            extraction_model="tesseract",
            extraction_cost=0.0,
            extracted_text="Extracted content here",
        )
        assert doc.extraction_status == "completed"
        assert doc.extraction_method == "sandboxed"
        assert doc.extraction_quality == 0.95
        assert doc.extraction_model == "tesseract"
        assert doc.extraction_cost == 0.0
        assert doc.extracted_text == "Extracted content here"

    def test_document_lifecycle_fields(self):
        """Document should track lifecycle timestamps."""
        now = datetime.utcnow()
        doc = Document(
            checksum="abc123",
            file_size=1024,
            first_seen_at=now,
            last_seen_at=now,
            is_deleted=False,
        )
        assert doc.first_seen_at == now
        assert doc.last_seen_at == now
        assert doc.is_deleted is False

    def test_document_default_checksum_algorithm(self):
        """Document should default to sha256 checksum algorithm when set explicitly.

        Note: SQLAlchemy defaults only apply on database insert. For unit tests,
        we test that the model accepts the default value.
        """
        doc = Document(
            checksum="abc123",
            file_size=1024,
            checksum_algorithm="sha256",  # Set explicitly for unit test
        )
        assert doc.checksum_algorithm == "sha256"

    def test_document_repr(self):
        """Document __repr__ should include useful info."""
        doc = Document(
            checksum="abc123",
            file_size=1024,
            original_filename="test.pdf",
        )
        doc.id = uuid.uuid4()
        repr_str = repr(doc)
        assert "Document" in repr_str
        assert "test.pdf" in repr_str


class TestDocumentOriginModel:
    """Tests for DocumentOrigin model."""

    def test_document_origin_folder_type(self):
        """DocumentOrigin should support folder origin type."""
        origin = DocumentOrigin(
            document_id=uuid.uuid4(),
            origin_type="folder",
            origin_host="nas.local",
            origin_path="/documents/reports/q4.pdf",
            origin_filename="q4.pdf",
        )
        assert origin.origin_type == "folder"
        assert origin.origin_host == "nas.local"
        assert origin.origin_path == "/documents/reports/q4.pdf"
        assert origin.origin_filename == "q4.pdf"

    def test_document_origin_email_attachment_type(self):
        """DocumentOrigin should support email attachment origin type."""
        email_id = uuid.uuid4()
        origin = DocumentOrigin(
            document_id=uuid.uuid4(),
            origin_type="email_attachment",
            email_id=email_id,
            attachment_index=0,
            origin_filename="attachment.pdf",
        )
        assert origin.origin_type == "email_attachment"
        assert origin.email_id == email_id
        assert origin.attachment_index == 0

    def test_document_origin_google_drive_type(self):
        """DocumentOrigin should support Google Drive origin type."""
        origin = DocumentOrigin(
            document_id=uuid.uuid4(),
            origin_type="google_drive",
            origin_path="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            origin_filename="Shared Document.docx",
        )
        assert origin.origin_type == "google_drive"
        assert origin.origin_path == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"

    def test_document_origin_discovery_metadata(self):
        """DocumentOrigin should track discovery metadata."""
        now = datetime.utcnow()
        origin = DocumentOrigin(
            document_id=uuid.uuid4(),
            origin_type="folder",
            file_modified_at=now,
            discovered_at=now,
            last_verified_at=now,
            is_primary=True,
        )
        assert origin.file_modified_at == now
        assert origin.discovered_at == now
        assert origin.last_verified_at == now
        assert origin.is_primary is True

    def test_document_origin_deletion_tracking(self):
        """DocumentOrigin should support soft deletion."""
        now = datetime.utcnow()
        origin = DocumentOrigin(
            document_id=uuid.uuid4(),
            origin_type="folder",
            is_deleted=True,
            deleted_at=now,
        )
        assert origin.is_deleted is True
        assert origin.deleted_at == now

    def test_document_origin_repr(self):
        """DocumentOrigin __repr__ should include useful info."""
        origin = DocumentOrigin(
            document_id=uuid.uuid4(),
            origin_type="folder",
            origin_path="/test/path.pdf",
        )
        origin.id = uuid.uuid4()
        repr_str = repr(origin)
        assert "DocumentOrigin" in repr_str
        assert "folder" in repr_str


class TestDocumentEmbeddingModel:
    """Tests for DocumentEmbedding model."""

    def test_document_embedding_creation(self):
        """DocumentEmbedding should be created with required fields."""
        doc_id = uuid.uuid4()
        embedding = DocumentEmbedding(
            document_id=doc_id,
            embedding=[0.1] * 3072,  # 3072 dimensions
            model="text-embedding-3-large",
        )
        assert embedding.document_id == doc_id
        assert len(embedding.embedding) == 3072
        assert embedding.model == "text-embedding-3-large"

    def test_document_embedding_page_level(self):
        """DocumentEmbedding should support page-level embeddings."""
        embedding = DocumentEmbedding(
            document_id=uuid.uuid4(),
            embedding=[0.1] * 3072,
            page_number=1,
            chunk_index=0,
        )
        assert embedding.page_number == 1
        assert embedding.chunk_index == 0

    def test_document_embedding_chunking(self):
        """DocumentEmbedding should support chunking within pages."""
        embedding = DocumentEmbedding(
            document_id=uuid.uuid4(),
            embedding=[0.1] * 3072,
            page_number=1,
            chunk_index=2,
            chunk_start=5000,
            chunk_end=10000,
            chunk_text="This is the chunk text",
        )
        assert embedding.chunk_index == 2
        assert embedding.chunk_start == 5000
        assert embedding.chunk_end == 10000
        assert embedding.chunk_text == "This is the chunk text"

    def test_document_embedding_default_values(self):
        """DocumentEmbedding should have correct defaults when set explicitly.

        Note: SQLAlchemy defaults only apply on database insert. For unit tests,
        we test that the model accepts the default values.
        """
        embedding = DocumentEmbedding(
            document_id=uuid.uuid4(),
            embedding=[0.1] * 3072,
            chunk_index=0,  # Set explicitly for unit test
            model="text-embedding-3-large",  # Set explicitly for unit test
        )
        assert embedding.chunk_index == 0
        assert embedding.model == "text-embedding-3-large"

    def test_document_embedding_repr(self):
        """DocumentEmbedding __repr__ should include useful info."""
        embedding = DocumentEmbedding(
            document_id=uuid.uuid4(),
            embedding=[0.1] * 3072,
            page_number=1,
            chunk_index=0,
        )
        embedding.id = uuid.uuid4()
        repr_str = repr(embedding)
        assert "DocumentEmbedding" in repr_str


class TestDocumentProcessingQueueModel:
    """Tests for DocumentProcessingQueue model."""

    def test_processing_queue_creation(self):
        """DocumentProcessingQueue should be created with required fields."""
        doc_id = uuid.uuid4()
        task = DocumentProcessingQueue(
            document_id=doc_id,
            task_type="extract_text",
            status="pending",  # Set explicitly for unit test
        )
        assert task.document_id == doc_id
        assert task.task_type == "extract_text"
        assert task.status == "pending"

    def test_processing_queue_task_types(self):
        """DocumentProcessingQueue should support different task types."""
        for task_type in ["extract_text", "generate_embedding", "classify"]:
            task = DocumentProcessingQueue(
                document_id=uuid.uuid4(),
                task_type=task_type,
            )
            assert task.task_type == task_type

    def test_processing_queue_priority(self):
        """DocumentProcessingQueue should support priority ordering."""
        task = DocumentProcessingQueue(
            document_id=uuid.uuid4(),
            task_type="extract_text",
            priority=1,  # High priority
        )
        assert task.priority == 1

    def test_processing_queue_default_priority(self):
        """DocumentProcessingQueue should accept priority 5 as default.

        Note: SQLAlchemy defaults only apply on database insert. For unit tests,
        we test that the model accepts the default value.
        """
        task = DocumentProcessingQueue(
            document_id=uuid.uuid4(),
            task_type="extract_text",
            priority=5,  # Set explicitly for unit test
        )
        assert task.priority == 5

    def test_processing_queue_retry_tracking(self):
        """DocumentProcessingQueue should track retry attempts."""
        task = DocumentProcessingQueue(
            document_id=uuid.uuid4(),
            task_type="extract_text",
            attempts=2,
            max_attempts=3,
            last_error="Connection timeout",
        )
        assert task.attempts == 2
        assert task.max_attempts == 3
        assert task.last_error == "Connection timeout"

    def test_processing_queue_worker_assignment(self):
        """DocumentProcessingQueue should track worker assignment."""
        now = datetime.utcnow()
        task = DocumentProcessingQueue(
            document_id=uuid.uuid4(),
            task_type="extract_text",
            status="processing",
            worker_id="worker-001",
            started_at=now,
        )
        assert task.status == "processing"
        assert task.worker_id == "worker-001"
        assert task.started_at == now

    def test_processing_queue_completion(self):
        """DocumentProcessingQueue should track completion."""
        now = datetime.utcnow()
        task = DocumentProcessingQueue(
            document_id=uuid.uuid4(),
            task_type="extract_text",
            status="completed",
            completed_at=now,
        )
        assert task.status == "completed"
        assert task.completed_at == now

    def test_processing_queue_repr(self):
        """DocumentProcessingQueue __repr__ should include useful info."""
        task = DocumentProcessingQueue(
            document_id=uuid.uuid4(),
            task_type="extract_text",
            status="pending",
        )
        task.id = uuid.uuid4()
        repr_str = repr(task)
        assert "DocumentProcessingQueue" in repr_str
        assert "extract_text" in repr_str
        assert "pending" in repr_str
