"""
Tests for Unified Search API Endpoints.

Phase 5 tests for:
- GET /api/search/unified
- GET /api/search/unified/related
"""

import pytest
from datetime import datetime, date
from uuid import uuid4
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from backend.api.routes.search import router
from backend.core.database import get_db
from backend.api.auth import verify_api_key
from backend.core.search.unified_search import UnifiedSearchResult, ResultType
from backend.core.database.models import Email
from backend.core.documents.models import Document


# Create test app
app = FastAPI()
app.include_router(router)


# Override dependencies
def override_verify_api_key():
    return True


def override_get_db():
    db = MagicMock()
    yield db


app.dependency_overrides[verify_api_key] = override_verify_api_key
app.dependency_overrides[get_db] = override_get_db


client = TestClient(app)


@pytest.fixture
def sample_email():
    """Create a sample email for testing."""
    email = MagicMock(spec=Email)
    email.id = uuid4()
    email.message_id = "<test@example.com>"
    email.subject = "Test Email"
    email.from_name = "Test Sender"
    email.from_address = "sender@example.com"
    email.to_addresses = ["recipient@example.com"]
    email.cc_addresses = []
    email.date = datetime(2024, 1, 15)
    email.body_text = "Test body"
    email.body_html = None
    email.body_markdown = None
    email.has_attachments = False
    email.folder = "INBOX"
    email.account_id = "test-account"
    email.uid = "12345"  # Must be string
    email.raw_headers = {}
    email.created_at = datetime.now()
    email.email_metadata = None
    email.is_seen = True
    email.is_flagged = False
    email.thread_id = None
    email.classifications = []
    return email


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    doc = MagicMock(spec=Document)
    doc.id = uuid4()
    doc.checksum = "abc123"
    doc.file_size = 1024
    doc.mime_type = "application/pdf"
    doc.original_filename = "invoice.pdf"
    doc.storage_path = "/storage/abc123.pdf"
    doc.document_type = "invoice"
    doc.extraction_status = "completed"
    doc.extraction_quality = 0.95
    doc.extraction_method = "pdfplumber"
    doc.page_count = 2
    doc.title = "Invoice #123"
    doc.summary = "This is an invoice"
    doc.document_date = datetime(2024, 1, 20)  # Must be datetime, not date
    doc.text_content = "Invoice content"
    doc.ai_summary = "This is an invoice"
    doc.ai_category = "financial"
    doc.ai_tags = ["invoice", "financial"]
    doc.is_deleted = False
    doc.created_at = datetime.now()
    doc.first_seen_at = datetime.now()
    doc.last_seen_at = datetime.now()
    doc.source_emails = []
    return doc


class TestUnifiedSearchEndpoint:
    """Tests for GET /api/search/unified endpoint."""

    def test_unified_search_basic(self, sample_email, sample_document):
        """Should return unified search results."""
        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[
                UnifiedSearchResult(
                    result_type=ResultType.EMAIL,
                    item=sample_email,
                    similarity=0.9,
                ),
                UnifiedSearchResult(
                    result_type=ResultType.DOCUMENT,
                    item=sample_document,
                    similarity=0.85,
                ),
            ])
            mock_class.return_value = mock_service

            response = client.get("/api/search/unified?q=test")

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "test"
            assert data["types"] == "all"
            assert data["total"] == 2
            assert len(data["results"]) == 2

    def test_unified_search_emails_only(self, sample_email):
        """Should filter to emails only."""
        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[
                UnifiedSearchResult(
                    result_type=ResultType.EMAIL,
                    item=sample_email,
                    similarity=0.88,
                ),
            ])
            mock_class.return_value = mock_service

            response = client.get("/api/search/unified?q=test&types=email")

            assert response.status_code == 200
            data = response.json()
            assert data["types"] == "email"
            assert all(r["result_type"] == "email" for r in data["results"])

    def test_unified_search_documents_only(self, sample_document):
        """Should filter to documents only."""
        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[
                UnifiedSearchResult(
                    result_type=ResultType.DOCUMENT,
                    item=sample_document,
                    similarity=0.82,
                ),
            ])
            mock_class.return_value = mock_service

            response = client.get("/api/search/unified?q=invoice&types=document")

            assert response.status_code == 200
            data = response.json()
            assert data["types"] == "document"
            assert all(r["result_type"] == "document" for r in data["results"])

    def test_unified_search_with_date_filter(self, sample_email):
        """Should pass date filters to service."""
        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_class.return_value = mock_service

            response = client.get(
                "/api/search/unified?q=test&date_from=2024-01-01&date_to=2024-01-31"
            )

            assert response.status_code == 200
            # Verify search was called with date params
            mock_service.search.assert_called_once()
            call_kwargs = mock_service.search.call_args[1]
            assert call_kwargs["date_from"] is not None
            assert call_kwargs["date_to"] is not None

    def test_unified_search_with_email_filters(self, sample_email):
        """Should pass email-specific filters."""
        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_class.return_value = mock_service

            response = client.get(
                "/api/search/unified?q=test&types=email&email_category=invitation-speaking&email_sender=stanford.edu"
            )

            assert response.status_code == 200
            call_kwargs = mock_service.search.call_args[1]
            assert call_kwargs["email_category"] == "invitation-speaking"
            assert call_kwargs["email_sender"] == "stanford.edu"

    def test_unified_search_with_document_filters(self, sample_document):
        """Should pass document-specific filters."""
        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_class.return_value = mock_service

            response = client.get(
                "/api/search/unified?q=test&types=document&document_type=invoice&document_min_quality=0.8"
            )

            assert response.status_code == 200
            call_kwargs = mock_service.search.call_args[1]
            assert call_kwargs["document_type"] == "invoice"
            assert call_kwargs["document_min_quality"] == 0.8

    def test_unified_search_invalid_date_format(self):
        """Should return 400 for invalid date format."""
        response = client.get("/api/search/unified?q=test&date_from=invalid-date")
        assert response.status_code == 400
        assert "Invalid date_from format" in response.json()["detail"]

    def test_unified_search_result_structure(self, sample_email, sample_document):
        """Result items should have correct structure."""
        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[
                UnifiedSearchResult(
                    result_type=ResultType.EMAIL,
                    item=sample_email,
                    similarity=0.9,
                ),
                UnifiedSearchResult(
                    result_type=ResultType.DOCUMENT,
                    item=sample_document,
                    similarity=0.85,
                ),
            ])
            mock_class.return_value = mock_service

            response = client.get("/api/search/unified?q=test")

            assert response.status_code == 200
            results = response.json()["results"]

            # Email result
            email_result = next(r for r in results if r["result_type"] == "email")
            assert email_result["email"] is not None
            assert email_result["document"] is None
            assert "similarity" in email_result

            # Document result
            doc_result = next(r for r in results if r["result_type"] == "document")
            assert doc_result["document"] is not None
            assert doc_result["email"] is None
            assert "similarity" in doc_result


class TestFindRelatedEndpoint:
    """Tests for GET /api/search/unified/related endpoint."""

    def test_find_related_from_email(self, sample_email, sample_document):
        """Should find items related to email."""
        email_id = str(uuid4())

        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.find_related = AsyncMock(return_value=[
                UnifiedSearchResult(
                    result_type=ResultType.DOCUMENT,
                    item=sample_document,
                    similarity=0.85,
                ),
            ])
            mock_class.return_value = mock_service

            response = client.get(f"/api/search/unified/related?email_id={email_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["reference_type"] == "email"
            assert data["reference_id"] == email_id

    def test_find_related_from_document(self, sample_email):
        """Should find items related to document."""
        doc_id = str(uuid4())

        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.find_related = AsyncMock(return_value=[
                UnifiedSearchResult(
                    result_type=ResultType.EMAIL,
                    item=sample_email,
                    similarity=0.78,
                ),
            ])
            mock_class.return_value = mock_service

            response = client.get(f"/api/search/unified/related?document_id={doc_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["reference_type"] == "document"
            assert data["reference_id"] == doc_id

    def test_find_related_no_reference_error(self):
        """Should return 400 when no reference provided."""
        response = client.get("/api/search/unified/related")
        assert response.status_code == 400
        assert "Must provide either email_id or document_id" in response.json()["detail"]

    def test_find_related_both_references_error(self):
        """Should return 400 when both references provided."""
        email_id = str(uuid4())
        doc_id = str(uuid4())

        response = client.get(
            f"/api/search/unified/related?email_id={email_id}&document_id={doc_id}"
        )
        assert response.status_code == 400
        assert "only one of email_id or document_id" in response.json()["detail"]

    def test_find_related_with_type_filter(self, sample_email):
        """Should filter to specific types."""
        email_id = str(uuid4())

        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.find_related = AsyncMock(return_value=[
                UnifiedSearchResult(
                    result_type=ResultType.EMAIL,
                    item=sample_email,
                    similarity=0.88,
                ),
            ])
            mock_class.return_value = mock_service

            response = client.get(
                f"/api/search/unified/related?email_id={email_id}&types=email"
            )

            assert response.status_code == 200
            call_kwargs = mock_service.find_related.call_args[1]
            assert call_kwargs["types"] == "email"

    def test_find_related_with_threshold(self, sample_email):
        """Should pass similarity threshold."""
        email_id = str(uuid4())

        with patch("backend.api.routes.search.UnifiedSearchService") as mock_class:
            mock_service = MagicMock()
            mock_service.find_related = AsyncMock(return_value=[])
            mock_class.return_value = mock_service

            response = client.get(
                f"/api/search/unified/related?email_id={email_id}&similarity_threshold=0.8"
            )

            assert response.status_code == 200
            call_kwargs = mock_service.find_related.call_args[1]
            assert call_kwargs["similarity_threshold"] == 0.8
