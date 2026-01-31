"""
Tests for Unified Search Service.

Phase 5 tests for:
- Cross-corpus semantic search (emails + documents)
- Type filtering
- Date range filtering
- Find related items
"""

import pytest
from datetime import datetime, date
from uuid import uuid4
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List

from backend.core.search.unified_search import (
    UnifiedSearchService,
    UnifiedSearchResult,
    ResultType,
)
from backend.core.database.models import Email
from backend.core.documents.models import Document, DocumentEmbedding


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    db.execute = MagicMock()
    db.query = MagicMock()
    db.rollback = MagicMock()
    return db


@pytest.fixture
def sample_emails():
    """Create sample emails for testing."""
    emails = []
    for i in range(3):
        email = MagicMock(spec=Email)
        email.id = uuid4()
        email.subject = f"Test Email {i}"
        email.from_address = f"sender{i}@example.com"
        email.date = datetime(2024, 1, i + 1)
        emails.append(email)
    return emails


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = []
    for i in range(3):
        doc = MagicMock(spec=Document)
        doc.id = uuid4()
        doc.original_filename = f"doc{i}.pdf"
        doc.document_type = "invoice" if i < 2 else "contract"
        doc.document_date = date(2024, 1, i + 10)
        doc.is_deleted = False
        docs.append(doc)
    return docs


@pytest.fixture
def mock_embedding_generator():
    """Mock the embedding generator."""
    with patch("backend.core.search.unified_search.EmbeddingGenerator") as mock_class:
        mock_instance = MagicMock()
        mock_instance.generate_query_embedding.return_value = [0.1] * 3072
        mock_class.return_value = mock_instance
        yield mock_instance


class TestUnifiedSearchService:
    """Tests for UnifiedSearchService initialization."""

    def test_init_creates_embedding_generator(self, mock_db, mock_embedding_generator):
        """Should initialize with embedding generator."""
        service = UnifiedSearchService(mock_db)
        assert service.db == mock_db
        assert service.embedding_generator is not None

    def test_init_optimizes_hnsw_params(self, mock_db, mock_embedding_generator):
        """Should set HNSW parameters on init."""
        service = UnifiedSearchService(mock_db, ef_search=100)
        mock_db.execute.assert_called()


class TestUnifiedSearch:
    """Tests for unified search method."""

    @pytest.mark.asyncio
    async def test_search_all_types(self, mock_db, mock_embedding_generator, sample_emails, sample_documents):
        """Should search both emails and documents."""
        # Mock email search results
        email_result = MagicMock()
        email_result.fetchall.return_value = [
            MagicMock(email_id=sample_emails[0].id, similarity=0.9),
        ]

        # Mock document search results
        doc_result = MagicMock()
        doc_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=0.85),
        ]

        # Execute returns different results based on call (1 for hnsw setup, 2 for searches)
        mock_db.execute.side_effect = [None, email_result, doc_result]

        # Mock queries for both email and document fetches
        def query_side_effect(model):
            query_mock = MagicMock()
            if model == Email:
                query_mock.filter.return_value.all.return_value = [sample_emails[0]]
            else:  # Document
                query_mock.filter.return_value.all.return_value = [sample_documents[0]]
            return query_mock

        mock_db.query.side_effect = query_side_effect

        service = UnifiedSearchService(mock_db)
        results = await service.search(query="test", types="all", top_k=10)

        # Should return results from both types
        assert len(results) == 2
        types_found = {r.result_type for r in results}
        assert ResultType.EMAIL in types_found
        assert ResultType.DOCUMENT in types_found

    @pytest.mark.asyncio
    async def test_search_emails_only(self, mock_db, mock_embedding_generator, sample_emails):
        """Should search only emails when types=email."""
        email_result = MagicMock()
        email_result.fetchall.return_value = [
            MagicMock(email_id=sample_emails[0].id, similarity=0.88),
        ]
        mock_db.execute.return_value = email_result

        email_query = MagicMock()
        email_query.filter.return_value.all.return_value = [sample_emails[0]]
        mock_db.query.return_value = email_query

        service = UnifiedSearchService(mock_db)
        results = await service.search(query="test", types="email", top_k=10)

        # Should only return emails
        assert all(r.result_type == ResultType.EMAIL for r in results)

    @pytest.mark.asyncio
    async def test_search_documents_only(self, mock_db, mock_embedding_generator, sample_documents):
        """Should search only documents when types=document."""
        doc_result = MagicMock()
        doc_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=0.75),
        ]
        mock_db.execute.return_value = doc_result

        doc_query = MagicMock()
        doc_query.filter.return_value.all.return_value = [sample_documents[0]]
        mock_db.query.return_value = doc_query

        service = UnifiedSearchService(mock_db)
        results = await service.search(query="invoices", types="document", top_k=10)

        # Should only return documents
        assert all(r.result_type == ResultType.DOCUMENT for r in results)

    @pytest.mark.asyncio
    async def test_search_with_date_filter(self, mock_db, mock_embedding_generator, sample_emails):
        """Should apply date filters."""
        email_result = MagicMock()
        email_result.fetchall.return_value = []
        mock_db.execute.return_value = email_result

        service = UnifiedSearchService(mock_db)
        results = await service.search(
            query="test",
            types="email",
            date_from=date(2024, 1, 1),
            date_to=date(2024, 1, 31),
        )

        # Should have passed date params to SQL
        call_args = mock_db.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert 'date_from' in params
        assert 'date_to' in params

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_similarity(self, mock_db, mock_embedding_generator, sample_emails, sample_documents):
        """Results should be sorted by similarity descending."""
        # Return results with specific scores
        email_result = MagicMock()
        email_result.fetchall.return_value = [
            MagicMock(email_id=sample_emails[0].id, similarity=0.7),
        ]

        doc_result = MagicMock()
        doc_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=0.9),
        ]

        # Execute returns different results (1 for hnsw setup, 2 for searches)
        mock_db.execute.side_effect = [None, email_result, doc_result]

        def query_side_effect(model):
            query_mock = MagicMock()
            if model == Email:
                query_mock.filter.return_value.all.return_value = [sample_emails[0]]
            else:
                query_mock.filter.return_value.all.return_value = [sample_documents[0]]
            return query_mock

        mock_db.query.side_effect = query_side_effect

        service = UnifiedSearchService(mock_db)
        results = await service.search(query="test", types="all", top_k=10)

        # Document with 0.9 should come before email with 0.7
        assert len(results) == 2
        assert results[0].similarity >= results[1].similarity
        assert results[0].result_type == ResultType.DOCUMENT

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, mock_db, mock_embedding_generator, sample_emails, sample_documents):
        """Should limit total results to top_k."""
        # Return multiple results
        email_result = MagicMock()
        email_result.fetchall.return_value = [
            MagicMock(email_id=sample_emails[i].id, similarity=0.8 - i * 0.1)
            for i in range(3)
        ]

        doc_result = MagicMock()
        doc_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[i].id, similarity=0.85 - i * 0.1)
            for i in range(3)
        ]

        mock_db.execute.side_effect = [None, email_result, doc_result]

        def query_side_effect(model):
            query_mock = MagicMock()
            if model == Email:
                query_mock.filter.return_value.all.return_value = sample_emails
            else:
                query_mock.filter.return_value.all.return_value = sample_documents
            return query_mock

        mock_db.query.side_effect = query_side_effect

        service = UnifiedSearchService(mock_db)
        results = await service.search(query="test", types="all", top_k=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_zero_vector_returns_empty(self, mock_db, mock_embedding_generator):
        """Should return empty when embedding generation fails."""
        mock_embedding_generator.generate_query_embedding.return_value = [0.0] * 3072

        service = UnifiedSearchService(mock_db)
        results = await service.search(query="test")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_email_filters(self, mock_db, mock_embedding_generator, sample_emails):
        """Should apply email-specific filters."""
        email_result = MagicMock()
        email_result.fetchall.return_value = []
        mock_db.execute.return_value = email_result

        service = UnifiedSearchService(mock_db)
        results = await service.search(
            query="grants",
            types="email",
            email_category="invitation-speaking",
            email_sender="stanford.edu",
            email_account="work",
        )

        # Should have applied filters in SQL
        call_args = mock_db.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert 'category' in params
        assert 'sender_pattern' in params
        assert 'account_id' in params

    @pytest.mark.asyncio
    async def test_search_with_document_filters(self, mock_db, mock_embedding_generator, sample_documents):
        """Should apply document-specific filters."""
        doc_result = MagicMock()
        doc_result.fetchall.return_value = []
        mock_db.execute.return_value = doc_result

        service = UnifiedSearchService(mock_db)
        results = await service.search(
            query="invoices",
            types="document",
            document_type="invoice",
            document_mime_type="application/pdf",
            document_min_quality=0.8,
        )

        # Should have applied filters in SQL
        call_args = mock_db.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert 'document_type' in params
        assert 'mime_type' in params
        assert 'min_quality' in params


class TestFindRelated:
    """Tests for find_related method."""

    @pytest.mark.asyncio
    async def test_find_related_from_email(self, mock_db, mock_embedding_generator, sample_emails, sample_documents):
        """Should find related items from email reference."""
        ref_email_id = sample_emails[0].id

        # Mock embedding lookup
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.2] * 3072

        emb_query = MagicMock()
        emb_query.filter.return_value.first.return_value = mock_embedding
        mock_db.query.return_value = emb_query

        # Mock search results
        email_result = MagicMock()
        email_result.fetchall.return_value = [
            MagicMock(email_id=sample_emails[1].id, similarity=0.85),
        ]

        doc_result = MagicMock()
        doc_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=0.8),
        ]

        mock_db.execute.side_effect = [email_result, doc_result]

        # Mock item fetches
        email_query = MagicMock()
        email_query.filter.return_value.all.return_value = [sample_emails[1]]

        doc_query = MagicMock()
        doc_query.filter.return_value.all.return_value = [sample_documents[0]]

        mock_db.query.side_effect = [emb_query, email_query, doc_query]

        service = UnifiedSearchService(mock_db)
        results = await service.find_related(email_id=ref_email_id, types="all")

        # Should not include reference email in results
        for r in results:
            if r.result_type == ResultType.EMAIL:
                assert r.item.id != ref_email_id

    @pytest.mark.asyncio
    async def test_find_related_from_document(self, mock_db, mock_embedding_generator, sample_emails, sample_documents):
        """Should find related items from document reference."""
        ref_doc_id = sample_documents[0].id

        # Mock embedding lookup
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.3] * 3072

        emb_query = MagicMock()
        emb_query.filter.return_value.first.return_value = mock_embedding
        mock_db.query.return_value = emb_query

        # Mock search results
        email_result = MagicMock()
        email_result.fetchall.return_value = [
            MagicMock(email_id=sample_emails[0].id, similarity=0.75),
        ]

        doc_result = MagicMock()
        doc_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[1].id, similarity=0.9),
        ]

        mock_db.execute.side_effect = [email_result, doc_result]

        email_query = MagicMock()
        email_query.filter.return_value.all.return_value = [sample_emails[0]]

        doc_query = MagicMock()
        doc_query.filter.return_value.all.return_value = [sample_documents[1]]

        mock_db.query.side_effect = [emb_query, email_query, doc_query]

        service = UnifiedSearchService(mock_db)
        results = await service.find_related(document_id=ref_doc_id, types="all")

        # Should not include reference doc in results
        for r in results:
            if r.result_type == ResultType.DOCUMENT:
                assert r.item.id != ref_doc_id

    @pytest.mark.asyncio
    async def test_find_related_no_reference_raises(self, mock_db, mock_embedding_generator):
        """Should raise error when no reference is provided."""
        service = UnifiedSearchService(mock_db)

        with pytest.raises(ValueError, match="Must provide either email_id or document_id"):
            await service.find_related(types="all")

    @pytest.mark.asyncio
    async def test_find_related_no_embedding_returns_empty(self, mock_db, mock_embedding_generator, sample_emails):
        """Should return empty when reference has no embedding."""
        emb_query = MagicMock()
        emb_query.filter.return_value.first.return_value = None
        mock_db.query.return_value = emb_query

        service = UnifiedSearchService(mock_db)
        results = await service.find_related(email_id=sample_emails[0].id)

        assert results == []


class TestUnifiedSearchResult:
    """Tests for UnifiedSearchResult dataclass."""

    def test_date_property_for_email(self, sample_emails):
        """Should return email date for email results."""
        email = sample_emails[0]
        result = UnifiedSearchResult(
            result_type=ResultType.EMAIL,
            item=email,
            similarity=0.9,
        )
        assert result.date == email.date

    def test_date_property_for_document(self, sample_documents):
        """Should return document_date for document results."""
        doc = sample_documents[0]
        result = UnifiedSearchResult(
            result_type=ResultType.DOCUMENT,
            item=doc,
            similarity=0.85,
        )
        assert result.date == doc.document_date


class TestNaNHandling:
    """Tests for NaN similarity score handling."""

    @pytest.mark.asyncio
    async def test_filters_nan_similarity_emails(self, mock_db, mock_embedding_generator, sample_emails):
        """Should filter out NaN similarity scores from emails."""
        email_result = MagicMock()
        email_result.fetchall.return_value = [
            MagicMock(email_id=sample_emails[0].id, similarity=float('nan')),
            MagicMock(email_id=sample_emails[1].id, similarity=0.8),
        ]
        mock_db.execute.return_value = email_result

        email_query = MagicMock()
        email_query.filter.return_value.all.return_value = sample_emails[:2]
        mock_db.query.return_value = email_query

        service = UnifiedSearchService(mock_db)
        results = await service.search(query="test", types="email")

        # NaN result should be filtered out
        for r in results:
            assert r.similarity == r.similarity  # NaN != NaN

    @pytest.mark.asyncio
    async def test_filters_out_of_range_similarity(self, mock_db, mock_embedding_generator, sample_documents):
        """Should filter out out-of-range similarity scores."""
        doc_result = MagicMock()
        doc_result.fetchall.return_value = [
            MagicMock(document_id=sample_documents[0].id, similarity=1.5),  # Invalid
            MagicMock(document_id=sample_documents[1].id, similarity=-0.1),  # Invalid
            MagicMock(document_id=sample_documents[2].id, similarity=0.8),  # Valid
        ]
        mock_db.execute.return_value = doc_result

        doc_query = MagicMock()
        doc_query.filter.return_value.all.return_value = sample_documents
        mock_db.query.return_value = doc_query

        service = UnifiedSearchService(mock_db)
        results = await service.search(query="test", types="document")

        # Only valid similarity should remain
        for r in results:
            assert 0 <= r.similarity <= 1
