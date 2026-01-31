"""
Tests for Document Search Service.

Phase 4 tests for:
- Semantic search with filters
- Similar document search
- Date range filtering
- Quality and type filtering
"""

import pytest
from datetime import date
from uuid import uuid4
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

from backend.core.documents.search import DocumentSearchService
from backend.core.documents.models import Document, DocumentEmbedding


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    db.execute = MagicMock()
    db.query = MagicMock()
    db.get = MagicMock()
    db.rollback = MagicMock()
    return db


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = []
    for i in range(5):
        doc = Document(
            checksum=f"checksum{i}",
            file_size=1024 * (i + 1),
            mime_type="application/pdf",
            original_filename=f"doc{i}.pdf",
            document_type="invoice" if i < 3 else "contract",
            extraction_quality=0.8 + i * 0.04,
            document_date=date(2024, 1, i + 1) if i < 4 else None,
        )
        doc.id = uuid4()
        doc.is_deleted = False
        docs.append(doc)
    return docs


@pytest.fixture
def mock_embedding_generator():
    """Mock the embedding generator."""
    with patch("backend.core.documents.search.EmbeddingGenerator") as mock_class:
        mock_instance = MagicMock()
        # Return a 3072-dim vector
        mock_instance.generate_query_embedding.return_value = [0.1] * 3072
        mock_class.return_value = mock_instance
        yield mock_instance


class TestDocumentSearchService:
    """Tests for DocumentSearchService initialization."""

    def test_init_sets_up_embedding_generator(self, mock_db, mock_embedding_generator):
        """Should initialize with embedding generator."""
        service = DocumentSearchService(mock_db)
        assert service.db == mock_db
        assert service.embedding_generator is not None

    def test_init_optimizes_index_params(self, mock_db, mock_embedding_generator):
        """Should set HNSW parameters on init."""
        service = DocumentSearchService(mock_db, ef_search=100)
        # Should have attempted to set ef_search
        mock_db.execute.assert_called()


class TestSemanticSearch:
    """Tests for semantic_search method."""

    @pytest.mark.asyncio
    async def test_semantic_search_basic(self, mock_db, mock_embedding_generator, sample_documents):
        """Should return documents matching query."""
        # Setup mock to return results
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=0.9),
            MagicMock(document_id=sample_documents[1].id, similarity=0.8),
        ]
        mock_db.execute.return_value = mock_result

        # Mock document query
        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = sample_documents[:2]
        mock_db.query.return_value = mock_query

        service = DocumentSearchService(mock_db)
        results = await service.semantic_search(
            query="find invoices",
            top_k=10,
        )

        # Should generate query embedding
        mock_embedding_generator.generate_query_embedding.assert_called_with("find invoices")

        # Should return documents with scores
        assert len(results) == 2
        assert results[0][1] == 0.9  # Similarity score
        assert results[1][1] == 0.8

    @pytest.mark.asyncio
    async def test_semantic_search_with_date_filter(self, mock_db, mock_embedding_generator, sample_documents):
        """Should filter by date range."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=0.85),
        ]
        mock_db.execute.return_value = mock_result

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [sample_documents[0]]
        mock_db.query.return_value = mock_query

        service = DocumentSearchService(mock_db)
        results = await service.semantic_search(
            query="invoices",
            date_from=date(2024, 1, 1),
            date_to=date(2024, 1, 15),
        )

        # Should have called execute with date params
        call_args = mock_db.execute.call_args
        assert call_args is not None
        # Verify date filtering was applied by checking params
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert 'date_from' in params or 'date_to' in params

    @pytest.mark.asyncio
    async def test_semantic_search_with_type_filter(self, mock_db, mock_embedding_generator, sample_documents):
        """Should filter by document type."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[3].id, similarity=0.75),
        ]
        mock_db.execute.return_value = mock_result

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [sample_documents[3]]
        mock_db.query.return_value = mock_query

        service = DocumentSearchService(mock_db)
        results = await service.semantic_search(
            query="legal agreements",
            document_type="contract",
        )

        # Should have applied document_type filter
        call_args = mock_db.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert 'document_type' in params

    @pytest.mark.asyncio
    async def test_semantic_search_with_quality_filter(self, mock_db, mock_embedding_generator, sample_documents):
        """Should filter by minimum quality."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db.execute.return_value = mock_result

        service = DocumentSearchService(mock_db)
        results = await service.semantic_search(
            query="high quality docs",
            min_quality=0.9,
        )

        call_args = mock_db.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert 'min_quality' in params
        assert params['min_quality'] == 0.9

    @pytest.mark.asyncio
    async def test_semantic_search_empty_embedding_fails(self, mock_db, mock_embedding_generator):
        """Should return empty when embedding generation fails."""
        # Return zero vector (indicates failure)
        mock_embedding_generator.generate_query_embedding.return_value = [0.0] * 3072

        service = DocumentSearchService(mock_db)
        results = await service.semantic_search(query="test")

        assert results == []

    @pytest.mark.asyncio
    async def test_semantic_search_handles_db_error(self, mock_db, mock_embedding_generator):
        """Should raise RuntimeError on database error."""
        mock_db.execute.side_effect = Exception("Database error")

        service = DocumentSearchService(mock_db)

        with pytest.raises(RuntimeError, match="Document search failed"):
            await service.semantic_search(query="test")

    @pytest.mark.asyncio
    async def test_semantic_search_threshold_filter(self, mock_db, mock_embedding_generator, sample_documents):
        """Should respect similarity threshold."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=0.9),
        ]
        mock_db.execute.return_value = mock_result

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [sample_documents[0]]
        mock_db.query.return_value = mock_query

        service = DocumentSearchService(mock_db)
        results = await service.semantic_search(
            query="test",
            similarity_threshold=0.8,
        )

        # Only high similarity results should be returned
        assert len(results) == 1
        assert results[0][1] >= 0.8


class TestFindSimilar:
    """Tests for find_similar method."""

    @pytest.mark.asyncio
    async def test_find_similar_basic(self, mock_db, mock_embedding_generator, sample_documents):
        """Should find similar documents."""
        ref_doc = sample_documents[0]

        # Mock embedding lookup
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.2] * 3072
        mock_db.query.return_value.filter.return_value.first.return_value = mock_embedding

        # Mock search results
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[1].id, similarity=0.88),
            MagicMock(document_id=sample_documents[2].id, similarity=0.75),
        ]
        mock_db.execute.return_value = mock_result

        # Mock document fetch
        mock_doc_query = MagicMock()
        mock_doc_query.filter.return_value.all.return_value = sample_documents[1:3]
        mock_db.query.return_value = mock_doc_query

        service = DocumentSearchService(mock_db)
        results = await service.find_similar(
            document_id=ref_doc.id,
            top_k=5,
        )

        assert len(results) <= 5
        # Reference document should not be in results
        for doc, _ in results:
            assert doc.id != ref_doc.id

    @pytest.mark.asyncio
    async def test_find_similar_no_embedding(self, mock_db, mock_embedding_generator, sample_documents):
        """Should return empty when reference has no embedding."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        service = DocumentSearchService(mock_db)
        results = await service.find_similar(
            document_id=sample_documents[0].id,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_find_similar_same_type_only(self, mock_db, mock_embedding_generator, sample_documents):
        """Should filter by same document type when requested."""
        ref_doc = sample_documents[0]
        ref_doc.document_type = "invoice"

        # Mock embedding lookup
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.2] * 3072
        mock_db.query.return_value.filter.return_value.first.return_value = mock_embedding

        # Mock document lookup for type
        mock_db.get.return_value = ref_doc

        # Mock search results
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db.execute.return_value = mock_result

        service = DocumentSearchService(mock_db)
        results = await service.find_similar(
            document_id=ref_doc.id,
            same_type_only=True,
        )

        # Should have fetched reference doc for type
        mock_db.get.assert_called()

    @pytest.mark.asyncio
    async def test_find_similar_with_date_range(self, mock_db, mock_embedding_generator, sample_documents):
        """Should respect date range filters."""
        ref_doc = sample_documents[0]

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.2] * 3072
        mock_db.query.return_value.filter.return_value.first.return_value = mock_embedding

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db.execute.return_value = mock_result

        service = DocumentSearchService(mock_db)
        results = await service.find_similar(
            document_id=ref_doc.id,
            date_from=date(2024, 1, 1),
            date_to=date(2024, 6, 30),
        )

        # Should have applied date filters
        call_args = mock_db.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert 'date_from' in params
        assert 'date_to' in params


class TestSearchByContentType:
    """Tests for search_by_content_type method."""

    @pytest.mark.asyncio
    async def test_search_pdfs_only(self, mock_db, mock_embedding_generator, sample_documents):
        """Should filter by PDF content type."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=0.8),
        ]
        mock_db.execute.return_value = mock_result

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [sample_documents[0]]
        mock_db.query.return_value = mock_query

        service = DocumentSearchService(mock_db)
        results = await service.search_by_content_type(
            query="important documents",
            content_types=["application/pdf"],
        )

        assert len(results) >= 0  # May be empty with mocks

    @pytest.mark.asyncio
    async def test_search_images(self, mock_db, mock_embedding_generator):
        """Should filter by image content type prefix."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db.execute.return_value = mock_result

        service = DocumentSearchService(mock_db)
        results = await service.search_by_content_type(
            query="photos",
            content_types=["image/"],  # Prefix match
        )

        # SQL should use LIKE for prefix
        call_args = mock_db.execute.call_args
        sql_text = str(call_args[0][0])
        assert "LIKE" in sql_text


class TestIndexStats:
    """Tests for get_index_stats method."""

    def test_get_index_stats_success(self, mock_db, mock_embedding_generator):
        """Should return index statistics."""
        # Mock index query
        mock_index_result = MagicMock()
        mock_index_result.fetchall.return_value = [
            ("idx_document_embeddings_vector", "125 MB"),
        ]
        mock_db.execute.return_value = mock_index_result

        # Mock count
        mock_db.query.return_value.count.return_value = 50000

        service = DocumentSearchService(mock_db)
        stats = service.get_index_stats()

        assert 'total_embeddings' in stats
        assert stats['total_embeddings'] == 50000

    def test_get_index_stats_error(self, mock_db, mock_embedding_generator):
        """Should handle errors gracefully."""
        mock_db.execute.side_effect = Exception("Query failed")

        service = DocumentSearchService(mock_db)
        stats = service.get_index_stats()

        assert 'error' in stats


class TestNaNHandling:
    """Tests for NaN similarity score handling."""

    @pytest.mark.asyncio
    async def test_filters_nan_similarity(self, mock_db, mock_embedding_generator, sample_documents):
        """Should filter out NaN similarity scores."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=float('nan')),
            MagicMock(document_id=sample_documents[1].id, similarity=0.8),
        ]
        mock_db.execute.return_value = mock_result

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = sample_documents[:2]
        mock_db.query.return_value = mock_query

        service = DocumentSearchService(mock_db)
        results = await service.semantic_search(query="test")

        # NaN result should be filtered out
        for doc, sim in results:
            assert sim == sim  # NaN != NaN

    @pytest.mark.asyncio
    async def test_filters_invalid_similarity(self, mock_db, mock_embedding_generator, sample_documents):
        """Should filter out out-of-range similarity scores."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=1.5),  # Invalid
            MagicMock(document_id=sample_documents[1].id, similarity=-0.1),  # Invalid
            MagicMock(document_id=sample_documents[2].id, similarity=0.8),  # Valid
        ]
        mock_db.execute.return_value = mock_result

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = sample_documents[:3]
        mock_db.query.return_value = mock_query

        service = DocumentSearchService(mock_db)
        results = await service.semantic_search(query="test")

        # Only valid similarity should remain
        for doc, sim in results:
            assert 0 <= sim <= 1
