"""
Tests for Document Repository.

Phase 1 tests for:
- Document CRUD operations
- Origin management
- Processing queue operations
- Embedding management
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from backend.core.documents.models import (
    Document,
    DocumentOrigin,
    DocumentEmbedding,
    DocumentProcessingQueue,
    ExtractionStatus,
)
from backend.core.documents.repository import DocumentRepository


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    db.add = Mock()
    db.flush = Mock()
    db.get = Mock()
    db.execute = Mock()
    return db


@pytest.fixture
def repository(mock_db):
    """Create a DocumentRepository with mock db."""
    return DocumentRepository(mock_db)


class TestDocumentCRUD:
    """Tests for document CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_document(self, repository, mock_db):
        """create_document should create document with required fields."""
        doc = await repository.create_document(
            checksum="abc123def456",
            file_size=1024,
            mime_type="application/pdf",
            original_filename="test.pdf",
        )

        assert doc.checksum == "abc123def456"
        assert doc.file_size == 1024
        assert doc.mime_type == "application/pdf"
        assert doc.original_filename == "test.pdf"
        assert doc.extraction_status == ExtractionStatus.PENDING.value
        mock_db.add.assert_called_once()
        mock_db.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_document_default_algorithm(self, repository):
        """create_document should default to sha256 algorithm."""
        doc = await repository.create_document(
            checksum="abc123",
            file_size=1024,
        )
        assert doc.checksum_algorithm == "sha256"

    @pytest.mark.asyncio
    async def test_get_document_by_id(self, repository, mock_db):
        """get_by_id should retrieve document by UUID."""
        doc_id = uuid.uuid4()
        expected_doc = Document(checksum="abc", file_size=100)
        mock_db.get.return_value = expected_doc

        result = await repository.get_by_id(doc_id)

        assert result == expected_doc
        mock_db.get.assert_called_once_with(Document, doc_id)

    @pytest.mark.asyncio
    async def test_get_document_by_id_not_found(self, repository, mock_db):
        """get_by_id should return None if not found."""
        mock_db.get.return_value = None

        result = await repository.get_by_id(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_checksum(self, repository, mock_db):
        """get_by_checksum should find document by checksum."""
        expected_doc = Document(checksum="abc123", file_size=100)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_doc
        mock_db.execute.return_value = mock_result

        result = await repository.get_by_checksum("abc123")

        assert result == expected_doc
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_checksum_not_found(self, repository, mock_db):
        """get_by_checksum should return None if not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await repository.get_by_checksum("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self, repository, mock_db):
        """get_or_create should return existing document."""
        existing_doc = Document(checksum="abc123", file_size=100)
        existing_doc.last_seen_at = datetime(2024, 1, 1)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_doc
        mock_db.execute.return_value = mock_result

        doc, is_new = await repository.get_or_create(
            checksum="abc123",
            file_size=100,
        )

        assert doc == existing_doc
        assert is_new is False
        # last_seen_at should be updated
        assert doc.last_seen_at > datetime(2024, 1, 1)

    @pytest.mark.asyncio
    async def test_get_or_create_new(self, repository, mock_db):
        """get_or_create should create new document if not exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        doc, is_new = await repository.get_or_create(
            checksum="new123",
            file_size=2048,
            mime_type="application/pdf",
        )

        assert doc.checksum == "new123"
        assert is_new is True
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_extraction(self, repository, mock_db):
        """update_extraction should update extraction results."""
        doc_id = uuid.uuid4()
        existing_doc = Document(checksum="abc", file_size=100)
        mock_db.get.return_value = existing_doc

        result = await repository.update_extraction(
            document_id=doc_id,
            extracted_text="Extracted content",
            extraction_status=ExtractionStatus.COMPLETED,
            extraction_method="sandboxed",
            extraction_quality=0.95,
            page_count=5,
            title="Document Title",
            summary="Document summary",
        )

        assert result.extracted_text == "Extracted content"
        assert result.extraction_status == ExtractionStatus.COMPLETED.value
        assert result.extraction_method == "sandboxed"
        assert result.extraction_quality == 0.95
        assert result.page_count == 5
        assert result.title == "Document Title"
        assert result.summary == "Document summary"
        assert result.extracted_at is not None

    @pytest.mark.asyncio
    async def test_update_extraction_not_found(self, repository, mock_db):
        """update_extraction should return None if document not found."""
        mock_db.get.return_value = None

        result = await repository.update_extraction(
            document_id=uuid.uuid4(),
            extracted_text="content",
            extraction_status=ExtractionStatus.COMPLETED,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_set_extraction_status(self, repository, mock_db):
        """set_extraction_status should update status."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db.execute.return_value = mock_result

        result = await repository.set_extraction_status(
            document_id=uuid.uuid4(),
            status=ExtractionStatus.PROCESSING,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_set_extraction_status_not_found(self, repository, mock_db):
        """set_extraction_status should return False if not found."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db.execute.return_value = mock_result

        result = await repository.set_extraction_status(
            document_id=uuid.uuid4(),
            status=ExtractionStatus.PROCESSING,
        )

        assert result is False


class TestOriginManagement:
    """Tests for document origin management."""

    @pytest.mark.asyncio
    async def test_add_origin_folder(self, repository, mock_db):
        """add_origin should create folder origin."""
        doc_id = uuid.uuid4()
        # Mock get_origin to return None (no existing)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        origin = await repository.add_origin(
            document_id=doc_id,
            origin_type="folder",
            origin_host="nas.local",
            origin_path="/documents/test.pdf",
            origin_filename="test.pdf",
        )

        assert origin.document_id == doc_id
        assert origin.origin_type == "folder"
        assert origin.origin_host == "nas.local"
        assert origin.origin_path == "/documents/test.pdf"
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_origin_email_attachment(self, repository, mock_db):
        """add_origin should create email attachment origin."""
        doc_id = uuid.uuid4()
        email_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        origin = await repository.add_origin(
            document_id=doc_id,
            origin_type="email_attachment",
            email_id=email_id,
            attachment_index=0,
            origin_filename="attachment.pdf",
        )

        assert origin.origin_type == "email_attachment"
        assert origin.email_id == email_id
        assert origin.attachment_index == 0

    @pytest.mark.asyncio
    async def test_add_duplicate_origin_ignored(self, repository, mock_db):
        """add_origin should return None for duplicate origin."""
        existing_origin = DocumentOrigin(
            document_id=uuid.uuid4(),
            origin_type="folder",
            origin_path="/test.pdf",
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_origin
        mock_db.execute.return_value = mock_result

        result = await repository.add_origin(
            document_id=uuid.uuid4(),
            origin_type="folder",
            origin_path="/test.pdf",
        )

        assert result is None
        # last_verified_at should be updated on existing
        assert existing_origin.last_verified_at is not None
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_origins(self, repository, mock_db):
        """get_origins should return all origins for document."""
        origins = [
            DocumentOrigin(document_id=uuid.uuid4(), origin_type="folder"),
            DocumentOrigin(document_id=uuid.uuid4(), origin_type="email_attachment"),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = origins
        mock_db.execute.return_value = mock_result

        result = await repository.get_origins(uuid.uuid4())

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_origins_by_email(self, repository, mock_db):
        """get_origins_by_email should return origins for an email."""
        email_id = uuid.uuid4()
        origins = [
            DocumentOrigin(email_id=email_id, origin_type="email_attachment", attachment_index=0),
            DocumentOrigin(email_id=email_id, origin_type="email_attachment", attachment_index=1),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = origins
        mock_db.execute.return_value = mock_result

        result = await repository.get_origins_by_email(email_id)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_mark_origin_deleted(self, repository, mock_db):
        """mark_origin_deleted should soft delete origin."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db.execute.return_value = mock_result

        result = await repository.mark_origin_deleted(uuid.uuid4())

        assert result is True


class TestProcessingQueue:
    """Tests for processing queue operations."""

    @pytest.mark.asyncio
    async def test_queue_task(self, repository, mock_db):
        """queue_task should create queue entry."""
        doc_id = uuid.uuid4()

        task = await repository.queue_task(
            document_id=doc_id,
            task_type="extract_text",
            priority=3,
        )

        assert task.document_id == doc_id
        assert task.task_type == "extract_text"
        assert task.priority == 3
        assert task.status == "pending"
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_queue_task_default_priority(self, repository, mock_db):
        """queue_task should default to priority 5."""
        task = await repository.queue_task(
            document_id=uuid.uuid4(),
            task_type="generate_embedding",
        )

        assert task.priority == 5

    @pytest.mark.asyncio
    async def test_get_pending_tasks(self, repository, mock_db):
        """get_pending_tasks should return pending tasks by priority."""
        tasks = [
            DocumentProcessingQueue(task_type="extract_text", priority=1),
            DocumentProcessingQueue(task_type="extract_text", priority=5),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = tasks
        mock_db.execute.return_value = mock_result

        result = await repository.get_pending_tasks(
            task_type="extract_text",
            limit=10,
        )

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_mark_task_processing(self, repository, mock_db):
        """mark_task_processing should claim task for worker."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db.execute.return_value = mock_result

        result = await repository.mark_task_processing(
            task_id=uuid.uuid4(),
            worker_id="worker-001",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_mark_task_processing_already_claimed(self, repository, mock_db):
        """mark_task_processing should return False if already claimed."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db.execute.return_value = mock_result

        result = await repository.mark_task_processing(
            task_id=uuid.uuid4(),
            worker_id="worker-002",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_mark_task_completed(self, repository, mock_db):
        """mark_task_completed should update task status."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db.execute.return_value = mock_result

        result = await repository.mark_task_completed(uuid.uuid4())

        assert result is True

    @pytest.mark.asyncio
    async def test_mark_task_failed_with_retry(self, repository, mock_db):
        """mark_task_failed should set back to pending if retries remain."""
        task = DocumentProcessingQueue(
            task_type="extract_text",
            attempts=1,
            max_attempts=3,
        )
        mock_db.get.return_value = task

        result = await repository.mark_task_failed(
            task_id=uuid.uuid4(),
            error="Connection timeout",
        )

        assert result is True
        assert task.status == "pending"
        assert task.last_error == "Connection timeout"
        assert task.worker_id is None

    @pytest.mark.asyncio
    async def test_mark_task_failed_max_attempts(self, repository, mock_db):
        """mark_task_failed should fail permanently at max attempts."""
        task = DocumentProcessingQueue(
            task_type="extract_text",
            attempts=3,
            max_attempts=3,
        )
        mock_db.get.return_value = task

        result = await repository.mark_task_failed(
            task_id=uuid.uuid4(),
            error="Connection timeout",
        )

        assert result is True
        assert task.status == "failed"
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_get_task(self, repository, mock_db):
        """get_task should retrieve task by ID."""
        task_id = uuid.uuid4()
        expected_task = DocumentProcessingQueue(task_type="extract_text")
        mock_db.get.return_value = expected_task

        result = await repository.get_task(task_id)

        assert result == expected_task


class TestEmbeddingManagement:
    """Tests for embedding management."""

    @pytest.mark.asyncio
    async def test_add_embedding(self, repository, mock_db):
        """add_embedding should create embedding."""
        doc_id = uuid.uuid4()
        embedding_vector = [0.1] * 3072

        embedding = await repository.add_embedding(
            document_id=doc_id,
            embedding=embedding_vector,
            page_number=1,
            chunk_index=0,
        )

        assert embedding.document_id == doc_id
        assert embedding.embedding == embedding_vector
        assert embedding.page_number == 1
        assert embedding.chunk_index == 0
        assert embedding.model == "text-embedding-3-large"
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_embedding_with_chunk_text(self, repository, mock_db):
        """add_embedding should store encrypted chunk text."""
        embedding = await repository.add_embedding(
            document_id=uuid.uuid4(),
            embedding=[0.1] * 3072,
            page_number=1,
            chunk_index=0,
            chunk_start=0,
            chunk_end=1000,
            chunk_text="This is the chunk text",
        )

        assert embedding.chunk_text == "This is the chunk text"
        assert embedding.chunk_start == 0
        assert embedding.chunk_end == 1000

    @pytest.mark.asyncio
    async def test_get_embeddings(self, repository, mock_db):
        """get_embeddings should return all embeddings for document."""
        embeddings = [
            DocumentEmbedding(document_id=uuid.uuid4(), embedding=[0.1] * 3072, page_number=1),
            DocumentEmbedding(document_id=uuid.uuid4(), embedding=[0.1] * 3072, page_number=2),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = embeddings
        mock_db.execute.return_value = mock_result

        result = await repository.get_embeddings(uuid.uuid4())

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_delete_embeddings(self, repository, mock_db):
        """delete_embeddings should remove all embeddings for document."""
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_db.execute.return_value = mock_result

        result = await repository.delete_embeddings(uuid.uuid4())

        assert result == 5


class TestLifecycle:
    """Tests for document lifecycle operations (orphans, file changes, OCR updates)."""

    @pytest.mark.asyncio
    async def test_update_origin_document(self, repository, mock_db):
        """update_origin_document should move origin to new document."""
        old_doc_id = uuid.uuid4()
        new_doc_id = uuid.uuid4()
        origin_id = uuid.uuid4()

        origin = DocumentOrigin(
            document_id=old_doc_id,
            origin_type="folder",
            origin_path="/test.pdf",
        )
        origin.id = origin_id
        mock_db.get.return_value = origin

        result = await repository.update_origin_document(
            origin_id=origin_id,
            new_document_id=new_doc_id,
        )

        assert result == old_doc_id
        assert origin.document_id == new_doc_id
        assert origin.last_verified_at is not None

    @pytest.mark.asyncio
    async def test_update_origin_document_not_found(self, repository, mock_db):
        """update_origin_document should return None if origin not found."""
        mock_db.get.return_value = None

        result = await repository.update_origin_document(
            origin_id=uuid.uuid4(),
            new_document_id=uuid.uuid4(),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_origin_by_path(self, repository, mock_db):
        """get_origin_by_path should find origin by path."""
        expected_origin = DocumentOrigin(
            origin_type="folder",
            origin_host="localhost",
            origin_path="/path/to/file.pdf",
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_origin
        mock_db.execute.return_value = mock_result

        result = await repository.get_origin_by_path(
            origin_type="folder",
            origin_host="localhost",
            origin_path="/path/to/file.pdf",
        )

        assert result == expected_origin

    @pytest.mark.asyncio
    async def test_count_document_origins(self, repository, mock_db):
        """count_document_origins should return origin count."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_db.execute.return_value = mock_result

        result = await repository.count_document_origins(uuid.uuid4())

        assert result == 3

    @pytest.mark.asyncio
    async def test_check_and_mark_orphaned_when_no_origins(self, repository, mock_db):
        """check_and_mark_orphaned should mark document when origins count is 0."""
        doc_id = uuid.uuid4()
        document = Document(checksum="abc", file_size=100)
        document.id = doc_id
        document.is_orphaned = False

        # count_document_origins returns 0
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0
        mock_db.execute.return_value = mock_count_result
        mock_db.get.return_value = document

        result = await repository.check_and_mark_orphaned(doc_id)

        assert result is True
        assert document.is_orphaned is True
        assert document.orphaned_at is not None

    @pytest.mark.asyncio
    async def test_check_and_mark_orphaned_when_has_origins(self, repository, mock_db):
        """check_and_mark_orphaned should not mark when origins exist."""
        doc_id = uuid.uuid4()

        # count_document_origins returns > 0
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 2
        mock_db.execute.return_value = mock_count_result

        result = await repository.check_and_mark_orphaned(doc_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_orphaned_documents(self, repository, mock_db):
        """get_orphaned_documents should return orphaned documents."""
        orphans = [
            Document(checksum="abc1", file_size=100, is_orphaned=True),
            Document(checksum="abc2", file_size=200, is_orphaned=True),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = orphans
        mock_db.execute.return_value = mock_result

        result = await repository.get_orphaned_documents(older_than_days=30)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_delete_orphaned_documents(self, repository, mock_db):
        """delete_orphaned_documents should delete orphaned documents."""
        orphans = [
            Document(checksum="abc1", file_size=100, is_orphaned=True),
        ]
        orphans[0].id = uuid.uuid4()
        orphans[0].original_filename = "old.pdf"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = orphans
        mock_db.execute.return_value = mock_result

        result = await repository.delete_orphaned_documents(older_than_days=30)

        assert result == 1
        mock_db.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_unorphan_document(self, repository, mock_db):
        """unorphan_document should clear orphaned status."""
        doc_id = uuid.uuid4()
        document = Document(checksum="abc", file_size=100)
        document.is_orphaned = True
        document.orphaned_at = datetime.utcnow()
        mock_db.get.return_value = document

        result = await repository.unorphan_document(doc_id)

        assert result is True
        assert document.is_orphaned is False
        assert document.orphaned_at is None

    @pytest.mark.asyncio
    async def test_unorphan_document_not_orphaned(self, repository, mock_db):
        """unorphan_document should return False if not orphaned."""
        document = Document(checksum="abc", file_size=100)
        document.is_orphaned = False
        mock_db.get.return_value = document

        result = await repository.unorphan_document(uuid.uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_store_extraction_structure_pages(self, repository, mock_db):
        """store_extraction_structure should store page structure."""
        doc_id = uuid.uuid4()
        document = Document(checksum="abc", file_size=100)
        mock_db.get.return_value = document

        pages = [{"page": 1, "text": "Page 1"}, {"page": 2, "text": "Page 2"}]
        result = await repository.store_extraction_structure(
            document_id=doc_id,
            pages=pages,
        )

        assert result == document
        assert document.extraction_structure == {"pages": pages}

    @pytest.mark.asyncio
    async def test_store_extraction_structure_sheets(self, repository, mock_db):
        """store_extraction_structure should store sheet structure."""
        doc_id = uuid.uuid4()
        document = Document(checksum="abc", file_size=100)
        mock_db.get.return_value = document

        sheets = [{"sheet": "Sheet1", "sheet_index": 0, "text": "Data"}]
        result = await repository.store_extraction_structure(
            document_id=doc_id,
            sheets=sheets,
        )

        assert document.extraction_structure == {"sheets": sheets}

    @pytest.mark.asyncio
    async def test_get_extraction_structure(self, repository, mock_db):
        """get_extraction_structure should return stored structure."""
        doc_id = uuid.uuid4()
        document = Document(checksum="abc", file_size=100)
        document.extraction_structure = {"pages": [{"page": 1, "text": "Content"}]}
        mock_db.get.return_value = document

        result = await repository.get_extraction_structure(doc_id)

        assert result == {"pages": [{"page": 1, "text": "Content"}]}

    @pytest.mark.asyncio
    async def test_get_extraction_structure_none(self, repository, mock_db):
        """get_extraction_structure should return None if no structure."""
        document = Document(checksum="abc", file_size=100)
        document.extraction_structure = None
        mock_db.get.return_value = document

        result = await repository.get_extraction_structure(uuid.uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_update_extraction_with_comparison_empty_existing(self, repository, mock_db):
        """update_extraction_with_comparison should update when existing is empty."""
        doc_id = uuid.uuid4()
        document = Document(checksum="abc", file_size=100)
        document.extracted_text = ""  # Empty existing
        document.extraction_quality = 0.0
        mock_db.get.return_value = document

        # Mock queue_task
        mock_db.add = Mock()
        mock_db.flush = Mock()

        updated, queued = await repository.update_extraction_with_comparison(
            document_id=doc_id,
            new_text="OCR extracted text",
            new_method="ocr_tesseract",
            new_quality=0.85,
        )

        assert updated is True
        assert queued is True
        assert document.extracted_text == "OCR extracted text"
        assert document.extraction_method == "ocr_tesseract"

    @pytest.mark.asyncio
    async def test_update_extraction_with_comparison_better_quality(self, repository, mock_db):
        """update_extraction_with_comparison should update when quality is better."""
        doc_id = uuid.uuid4()
        document = Document(checksum="abc", file_size=100)
        document.extracted_text = "Existing text content here"
        document.extraction_quality = 0.5
        mock_db.get.return_value = document
        mock_db.add = Mock()
        mock_db.flush = Mock()

        updated, queued = await repository.update_extraction_with_comparison(
            document_id=doc_id,
            new_text="Better OCR text",
            new_method="ocr_claude",
            new_quality=0.95,
        )

        assert updated is True
        assert document.extraction_quality == 0.95

    @pytest.mark.asyncio
    async def test_update_extraction_with_comparison_worse_quality(self, repository, mock_db):
        """update_extraction_with_comparison should not update when quality is worse."""
        doc_id = uuid.uuid4()
        document = Document(checksum="abc", file_size=100)
        # Text must be >= 100 chars to trigger quality comparison (otherwise considered "too short")
        existing_text = "This is a substantial existing text extraction that has good quality and should not be replaced with worse OCR results."
        document.extracted_text = existing_text
        document.extraction_quality = 0.9
        mock_db.get.return_value = document

        updated, queued = await repository.update_extraction_with_comparison(
            document_id=doc_id,
            new_text="Worse OCR",
            new_method="ocr_tesseract",
            new_quality=0.6,
        )

        assert updated is False
        assert queued is False
        assert document.extracted_text == existing_text

    @pytest.mark.asyncio
    async def test_update_extraction_with_comparison_force(self, repository, mock_db):
        """update_extraction_with_comparison should update when force=True."""
        doc_id = uuid.uuid4()
        document = Document(checksum="abc", file_size=100)
        document.extracted_text = "Good existing text content here"
        document.extraction_quality = 0.9
        mock_db.get.return_value = document
        mock_db.add = Mock()
        mock_db.flush = Mock()

        updated, queued = await repository.update_extraction_with_comparison(
            document_id=doc_id,
            new_text="Forced update",
            new_method="manual",
            new_quality=0.5,
            force=True,
        )

        assert updated is True
        assert document.extracted_text == "Forced update"


class TestStatistics:
    """Tests for statistics methods."""

    @pytest.mark.asyncio
    async def test_get_stats(self, repository, mock_db):
        """get_stats should return document statistics."""
        # Mock the various count queries
        mock_db.execute.side_effect = [
            MagicMock(scalar=MagicMock(return_value=100)),  # total
            MagicMock(all=MagicMock(return_value=[("completed", 80), ("pending", 20)])),  # by status
            MagicMock(all=MagicMock(return_value=[("pdf", 60), ("docx", 40)])),  # by type
            MagicMock(scalar=MagicMock(return_value=500)),  # embeddings
            MagicMock(all=MagicMock(return_value=[("extract_text", 10)])),  # pending tasks
        ]

        result = await repository.get_stats()

        assert "total_documents" in result
        assert "by_extraction_status" in result
        assert "by_document_type" in result
        assert "total_embeddings" in result
        assert "pending_tasks_by_type" in result
