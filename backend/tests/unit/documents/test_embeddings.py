"""
Tests for Document Embedding Service.

Phase 1 tests for:
- Embedding text preparation
- Single document embeddings
- Page-level embeddings
- Text chunking
- Batch embedding processing
"""

import pytest
import uuid
from unittest.mock import Mock, MagicMock, AsyncMock, patch

from backend.core.documents.models import Document, DocumentOrigin, ExtractionStatus
from backend.core.documents.embeddings import (
    DocumentEmbeddingService,
    PageContent,
    EmbeddingResult,
    MAX_CHARS_PER_CHUNK,
    EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
)


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = MagicMock()
    repo.delete_embeddings = AsyncMock()
    repo.add_embedding = AsyncMock()
    repo.get_pending_tasks = AsyncMock(return_value=[])
    repo.mark_task_processing = AsyncMock(return_value=True)
    repo.mark_task_completed = AsyncMock()
    repo.mark_task_failed = AsyncMock()
    repo.get_by_id = AsyncMock()
    return repo


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = MagicMock()
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS)]
    client.embeddings.create.return_value = response
    return client


@pytest.fixture
def embedding_service(mock_repository, mock_openai_client):
    """Create an embedding service with mocks."""
    service = DocumentEmbeddingService(
        repository=mock_repository,
        embedding_client=mock_openai_client,
    )
    return service


class TestEmbeddingServiceInit:
    """Tests for service initialization."""

    def test_init_with_client(self, mock_repository, mock_openai_client):
        """Service should accept a provided client."""
        service = DocumentEmbeddingService(
            repository=mock_repository,
            embedding_client=mock_openai_client,
        )
        assert service._client == mock_openai_client
        assert service.model == DEFAULT_EMBEDDING_MODEL

    def test_lazy_load_client(self, mock_repository):
        """Service should lazy-load OpenAI client when not provided."""
        service = DocumentEmbeddingService(repository=mock_repository)
        assert service._client is None

        # Accessing client property would trigger lazy load
        # We don't actually test this to avoid requiring OpenAI key


class TestPrepareDocumentForEmbedding:
    """Tests for prepare_document_for_embedding."""

    def test_prepare_with_all_fields(self, embedding_service):
        """Should include all available document fields."""
        doc = Document(
            checksum="abc123",
            file_size=1000,
            original_filename="report.pdf",
            title="Quarterly Report",
            document_type="report",
            ai_tags=["finance", "quarterly"],
            ai_category="financial",
            summary="Summary of Q4 results.",
            extracted_text="Full text content here.",
        )
        doc.id = uuid.uuid4()
        doc.origins = []

        text = embedding_service.prepare_document_for_embedding(doc)

        assert "Filename: report.pdf" in text
        assert "Title: Quarterly Report" in text
        assert "Type: report" in text
        assert "Tags: finance, quarterly" in text
        assert "Category: financial" in text
        assert "Summary: Summary of Q4 results." in text
        assert "Full text content here." in text

    def test_prepare_with_minimal_fields(self, embedding_service):
        """Should handle documents with minimal data."""
        doc = Document(
            checksum="abc123",
            file_size=1000,
            original_filename="file.pdf",
        )
        doc.id = uuid.uuid4()
        doc.origins = []

        text = embedding_service.prepare_document_for_embedding(doc)

        assert "Filename: file.pdf" in text
        assert "Title:" not in text  # No title
        assert "Summary:" not in text  # No summary

    def test_prepare_with_origin_path(self, embedding_service):
        """Should include origin path for context."""
        doc = Document(
            checksum="abc123",
            file_size=1000,
            original_filename="file.pdf",
        )
        doc.id = uuid.uuid4()

        origin = MagicMock()
        origin.is_primary = True
        origin.origin_path = "/documents/finance/2024/"
        doc.origins = [origin]

        text = embedding_service.prepare_document_for_embedding(doc)

        assert "Location: /documents/finance/2024/" in text

    def test_prepare_truncates_long_text(self, embedding_service):
        """Should truncate very long extracted text."""
        long_text = "A" * (MAX_CHARS_PER_CHUNK + 1000)
        doc = Document(
            checksum="abc123",
            file_size=1000,
            extracted_text=long_text,
        )
        doc.id = uuid.uuid4()
        doc.origins = []

        text = embedding_service.prepare_document_for_embedding(doc)

        assert len(text) <= MAX_CHARS_PER_CHUNK + 500  # Some buffer for metadata
        assert "..." in text

    def test_prepare_empty_document(self, embedding_service):
        """Should handle document with no useful fields."""
        doc = Document(
            checksum="abc123",
            file_size=0,
        )
        doc.id = uuid.uuid4()
        doc.origins = []

        text = embedding_service.prepare_document_for_embedding(doc)

        # Should return empty or minimal text
        assert text.strip() == "" or len(text) < 50


class TestGenerateEmbedding:
    """Tests for generate_embedding."""

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, embedding_service, mock_openai_client):
        """Should generate embedding for text."""
        embedding = await embedding_service.generate_embedding("Test text")

        assert len(embedding) == EMBEDDING_DIMENSIONS
        mock_openai_client.embeddings.create.assert_called_once_with(
            model=DEFAULT_EMBEDDING_MODEL,
            input="Test text",
        )

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self, embedding_service):
        """Should raise error for empty text."""
        with pytest.raises(ValueError, match="empty text"):
            await embedding_service.generate_embedding("")

        with pytest.raises(ValueError, match="empty text"):
            await embedding_service.generate_embedding("   ")


class TestGenerateDocumentEmbedding:
    """Tests for generate_document_embedding."""

    @pytest.mark.asyncio
    async def test_generate_single_embedding(self, embedding_service, mock_repository):
        """Should create single embedding for short document."""
        doc = Document(
            checksum="abc123",
            file_size=1000,
            original_filename="file.pdf",
            extracted_text="Short document content.",
        )
        doc.id = uuid.uuid4()
        doc.origins = []

        result = await embedding_service.generate_document_embedding(doc)

        assert isinstance(result, EmbeddingResult)
        assert result.embeddings_created == 1
        assert result.chunks_created == 1
        assert result.model == DEFAULT_EMBEDDING_MODEL
        mock_repository.delete_embeddings.assert_called_once_with(doc.id)
        mock_repository.add_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_chunked_embeddings(self, embedding_service, mock_repository):
        """Should create multiple embeddings for long document."""
        # Create text longer than MAX_CHARS_PER_CHUNK
        long_text = "Word " * (MAX_CHARS_PER_CHUNK // 5 + 1000)
        doc = Document(
            checksum="abc123",
            file_size=1000,
            extracted_text=long_text,
        )
        doc.id = uuid.uuid4()
        doc.origins = []

        result = await embedding_service.generate_document_embedding(doc)

        assert result.embeddings_created > 1
        assert result.chunks_created > 1
        assert mock_repository.add_embedding.call_count > 1

    @pytest.mark.asyncio
    async def test_generate_no_content(self, embedding_service, mock_repository):
        """Should handle document with no embeddable content."""
        doc = Document(
            checksum="abc123",
            file_size=0,
        )
        doc.id = uuid.uuid4()
        doc.origins = []

        result = await embedding_service.generate_document_embedding(doc)

        assert result.embeddings_created == 0
        mock_repository.add_embedding.assert_not_called()


class TestGeneratePageEmbeddings:
    """Tests for generate_page_embeddings."""

    @pytest.mark.asyncio
    async def test_generate_page_embeddings(self, embedding_service, mock_repository):
        """Should create embeddings for each page."""
        doc = Document(
            checksum="abc123",
            file_size=1000,
            title="Multi-page Document",
        )
        doc.id = uuid.uuid4()

        pages = [
            PageContent(page_number=1, text="Content of page 1."),
            PageContent(page_number=2, text="Content of page 2."),
            PageContent(page_number=3, text="Content of page 3."),
        ]

        result = await embedding_service.generate_page_embeddings(doc, pages)

        assert result.embeddings_created == 3
        assert result.pages_processed == 3
        assert mock_repository.add_embedding.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_page_embeddings_with_empty_pages(self, embedding_service, mock_repository):
        """Should skip empty pages."""
        doc = Document(
            checksum="abc123",
            file_size=1000,
        )
        doc.id = uuid.uuid4()

        pages = [
            PageContent(page_number=1, text="Content of page 1."),
            PageContent(page_number=2, text=""),  # Empty
            PageContent(page_number=3, text="   "),  # Whitespace only
            PageContent(page_number=4, text="Content of page 4."),
        ]

        result = await embedding_service.generate_page_embeddings(doc, pages)

        assert result.embeddings_created == 2  # Only pages 1 and 4
        assert result.pages_processed == 4  # All pages counted

    @pytest.mark.asyncio
    async def test_generate_page_embeddings_long_page(self, embedding_service, mock_repository):
        """Should chunk long pages."""
        doc = Document(
            checksum="abc123",
            file_size=1000,
        )
        doc.id = uuid.uuid4()

        long_page_text = "Word " * (MAX_CHARS_PER_CHUNK // 5 + 1000)
        pages = [
            PageContent(page_number=1, text=long_page_text),
        ]

        result = await embedding_service.generate_page_embeddings(doc, pages)

        assert result.embeddings_created > 1  # Chunked
        assert result.pages_processed == 1


class TestChunkText:
    """Tests for _chunk_text."""

    def test_chunk_short_text(self, embedding_service):
        """Short text should return single chunk."""
        text = "Short text"
        chunks = embedding_service._chunk_text(text, 1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_at_paragraph_boundary(self, embedding_service):
        """Should prefer breaking at paragraphs."""
        text = "First paragraph.\n\nSecond paragraph that is longer."
        chunks = embedding_service._chunk_text(text, 25)

        # Should break at paragraph boundary
        assert len(chunks) >= 2
        assert chunks[0].strip().endswith("paragraph.")

    def test_chunk_at_sentence_boundary(self, embedding_service):
        """Should break at sentence boundary if no paragraph break."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = embedding_service._chunk_text(text, 25)

        assert len(chunks) >= 2
        # Each chunk should end at a sentence boundary (mostly)
        for chunk in chunks[:-1]:  # Except maybe the last
            stripped = chunk.strip()
            if stripped:
                assert stripped[-1] in '.!?'

    def test_chunk_at_word_boundary(self, embedding_service):
        """Should break at word boundary as fallback."""
        text = "abcdefghijklmnopqrstuvwxyz " * 10
        chunks = embedding_service._chunk_text(text, 30)

        assert len(chunks) > 1
        # No chunks should have broken words (mostly)
        for chunk in chunks[:-1]:
            assert chunk.endswith(" ") or chunk.endswith("z")


class TestPreparePageText:
    """Tests for _prepare_page_text."""

    def test_prepare_page_with_title(self, embedding_service):
        """Should include document title in page context."""
        doc = Document(
            checksum="abc123",
            file_size=1000,
            title="Important Document",
        )
        page = PageContent(page_number=5, text="Page content here.")

        text = embedding_service._prepare_page_text(doc, page)

        assert "Document: Important Document" in text
        assert "Page 5" in text
        assert "Page content here." in text

    def test_prepare_page_with_filename(self, embedding_service):
        """Should use filename if no title."""
        doc = Document(
            checksum="abc123",
            file_size=1000,
            original_filename="report.pdf",
        )
        page = PageContent(page_number=1, text="Page content.")

        text = embedding_service._prepare_page_text(doc, page)

        assert "Document: report.pdf" in text


class TestEmbedPendingDocuments:
    """Tests for embed_pending_documents."""

    @pytest.mark.asyncio
    async def test_embed_pending_no_tasks(self, embedding_service, mock_repository):
        """Should handle no pending tasks."""
        mock_repository.get_pending_tasks.return_value = []

        processed = await embedding_service.embed_pending_documents(limit=10)

        assert processed == 0
        mock_repository.get_pending_tasks.assert_called_once_with(
            task_type="generate_embedding",
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_embed_pending_processes_tasks(self, embedding_service, mock_repository):
        """Should process pending tasks."""
        task = MagicMock()
        task.id = uuid.uuid4()
        task.document_id = uuid.uuid4()

        doc = Document(
            checksum="abc123",
            file_size=1000,
            original_filename="file.pdf",
            extracted_text="Content to embed.",
        )
        doc.id = task.document_id
        doc.origins = []

        mock_repository.get_pending_tasks.return_value = [task]
        mock_repository.get_by_id.return_value = doc

        processed = await embedding_service.embed_pending_documents(limit=10)

        assert processed == 1
        mock_repository.mark_task_processing.assert_called_once()
        mock_repository.mark_task_completed.assert_called_once_with(task.id)

    @pytest.mark.asyncio
    async def test_embed_pending_handles_missing_document(self, embedding_service, mock_repository):
        """Should mark task failed if document not found."""
        task = MagicMock()
        task.id = uuid.uuid4()
        task.document_id = uuid.uuid4()

        mock_repository.get_pending_tasks.return_value = [task]
        mock_repository.get_by_id.return_value = None  # Document not found

        processed = await embedding_service.embed_pending_documents(limit=10)

        assert processed == 0
        mock_repository.mark_task_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_pending_handles_claim_failure(self, embedding_service, mock_repository):
        """Should skip task if claim fails (another worker claimed it)."""
        task = MagicMock()
        task.id = uuid.uuid4()
        task.document_id = uuid.uuid4()

        mock_repository.get_pending_tasks.return_value = [task]
        mock_repository.mark_task_processing.return_value = False  # Claim failed

        processed = await embedding_service.embed_pending_documents(limit=10)

        assert processed == 0
        mock_repository.get_by_id.assert_not_called()  # Didn't try to get doc

    @pytest.mark.asyncio
    async def test_embed_pending_handles_embedding_error(self, embedding_service, mock_repository, mock_openai_client):
        """Should mark task failed on embedding error."""
        task = MagicMock()
        task.id = uuid.uuid4()
        task.document_id = uuid.uuid4()

        doc = Document(
            checksum="abc123",
            file_size=1000,
            extracted_text="Content.",
        )
        doc.id = task.document_id
        doc.origins = []

        mock_repository.get_pending_tasks.return_value = [task]
        mock_repository.get_by_id.return_value = doc
        mock_openai_client.embeddings.create.side_effect = Exception("API error")

        processed = await embedding_service.embed_pending_documents(limit=10)

        assert processed == 0
        mock_repository.mark_task_failed.assert_called_once()


class TestConstants:
    """Tests for module constants."""

    def test_embedding_dimensions(self):
        """Should use correct embedding dimensions."""
        assert EMBEDDING_DIMENSIONS == 3072

    def test_default_model(self):
        """Should use text-embedding-3-large."""
        assert DEFAULT_EMBEDDING_MODEL == "text-embedding-3-large"

    def test_max_chars_reasonable(self):
        """Max chars should be reasonable for token limit."""
        # 8000 tokens * 4 chars/token = 32000 chars
        assert MAX_CHARS_PER_CHUNK == 32000
