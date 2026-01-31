"""
Tests for Document API Endpoints.

Phase 1 tests for:
- List documents endpoint
- Get document by ID
- Get document by checksum
- Get document origins
- Get document text
- Get document stats
"""

import pytest
import uuid
from datetime import datetime, date
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from backend.core.documents.models import Document, DocumentOrigin, ExtractionStatus
from backend.api.routes.documents import router
from backend.api.auth import verify_api_key
from backend.core.database import get_db


# Create a test app with the documents router
def create_test_app():
    """Create test app with overridden dependencies."""
    app = FastAPI()
    app.include_router(router)

    # Override auth dependency
    app.dependency_overrides[verify_api_key] = lambda: None

    return app


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    return db


@pytest.fixture
def client(mock_db):
    """Create a test client with mocked db."""
    app = create_test_app()
    app.dependency_overrides[get_db] = lambda: mock_db
    return TestClient(app)


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    doc = Document(
        checksum="abc123def456",
        file_size=1024,
        mime_type="application/pdf",
        original_filename="report.pdf",
        page_count=10,
        extraction_status=ExtractionStatus.COMPLETED.value,
        extraction_quality=0.95,
        extraction_method="sandboxed",
        title="Quarterly Report",
        summary="Summary of Q4 results.",
        document_type="report",
        document_date=datetime(2024, 12, 31),
        ai_category="financial",
        ai_tags=["finance", "quarterly"],
        first_seen_at=datetime(2025, 1, 1),
        last_seen_at=datetime(2025, 1, 15),
        created_at=datetime(2025, 1, 1),
        is_deleted=False,
    )
    doc.id = uuid.uuid4()
    return doc


@pytest.fixture
def sample_origin(sample_document):
    """Create a sample origin for testing."""
    origin = DocumentOrigin(
        document_id=sample_document.id,
        origin_type="folder",
        origin_host="localhost",
        origin_path="/documents/finance/",
        origin_filename="report.pdf",
        discovered_at=datetime(2025, 1, 1),
        last_verified_at=datetime(2025, 1, 15),
        is_primary=True,
        is_deleted=False,
    )
    origin.id = uuid.uuid4()
    return origin


class TestListDocuments:
    """Tests for list_documents endpoint."""

    def test_list_documents_success(self, sample_document, mock_db):
        """Should return paginated list of documents."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [sample_document]
        mock_db.query.return_value = mock_query

        client = TestClient(app)
        response = client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "pages" in data

    def test_list_documents_with_filters(self, mock_db):
        """Should apply filters correctly."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_db.query.return_value = mock_query

        client = TestClient(app)
        response = client.get(
            "/api/documents",
            params={
                "extraction_status": "completed",
                "document_type": "report",
                "mime_type": "application/pdf",
                "min_quality": 0.8,
                "search": "quarterly",
            }
        )

        assert response.status_code == 200

    def test_list_documents_pagination(self, mock_db):
        """Should respect pagination parameters."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 100
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_db.query.return_value = mock_query

        client = TestClient(app)
        response = client.get(
            "/api/documents",
            params={"page": 3, "page_size": 25}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 3
        assert data["page_size"] == 25
        assert data["pages"] == 4  # 100 / 25 = 4


class TestGetDocument:
    """Tests for get_document endpoint."""

    def test_get_document_success(self, sample_document, sample_origin, mock_db):
        """Should return document with origins."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=sample_document)
            mock_repo.get_origins = AsyncMock(return_value=[sample_origin])
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(f"/api/documents/{sample_document.id}")

            assert response.status_code == 200
            data = response.json()
            assert "document" in data
            assert "origins" in data
            assert data["document"]["checksum"] == sample_document.checksum

    def test_get_document_not_found(self, mock_db):
        """Should return 404 for missing document."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(f"/api/documents/{uuid.uuid4()}")

            assert response.status_code == 404


class TestGetDocumentByChecksum:
    """Tests for get_document_by_checksum endpoint."""

    def test_get_by_checksum_found(self, sample_document, mock_db):
        """Should return document when found."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_checksum = AsyncMock(return_value=sample_document)
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(f"/api/documents/by-checksum/{sample_document.checksum}")

            assert response.status_code == 200
            data = response.json()
            assert data["checksum"] == sample_document.checksum

    def test_get_by_checksum_not_found(self, mock_db):
        """Should return null when not found."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_checksum = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get("/api/documents/by-checksum/nonexistent123")

            assert response.status_code == 200
            assert response.json() is None


class TestGetDocumentOrigins:
    """Tests for get_document_origins endpoint."""

    def test_get_origins_success(self, sample_document, sample_origin, mock_db):
        """Should return list of origins."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=sample_document)
            mock_repo.get_origins = AsyncMock(return_value=[sample_origin])
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(f"/api/documents/{sample_document.id}/origins")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["origin_type"] == "folder"

    def test_get_origins_include_deleted(self, sample_document, mock_db):
        """Should respect include_deleted parameter."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=sample_document)
            mock_repo.get_origins = AsyncMock(return_value=[])
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(
                f"/api/documents/{sample_document.id}/origins",
                params={"include_deleted": True}
            )

            assert response.status_code == 200
            mock_repo.get_origins.assert_called_once_with(
                sample_document.id,
                include_deleted=True
            )

    def test_get_origins_document_not_found(self, mock_db):
        """Should return 404 when document not found."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(f"/api/documents/{uuid.uuid4()}/origins")

            assert response.status_code == 404


class TestGetDocumentText:
    """Tests for get_document_text endpoint."""

    def test_get_text_success(self, sample_document, mock_db):
        """Should return extracted text."""
        sample_document.extracted_text = "This is the extracted text content."

        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=sample_document)
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(f"/api/documents/{sample_document.id}/text")

            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "This is the extracted text content."
            assert data["title"] == sample_document.title

    def test_get_text_not_extracted(self, sample_document, mock_db):
        """Should return 400 when text not extracted."""
        sample_document.extraction_status = ExtractionStatus.PENDING.value

        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=sample_document)
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(f"/api/documents/{sample_document.id}/text")

            assert response.status_code == 400
            assert "pending" in response.json()["detail"].lower()

    def test_get_text_document_not_found(self, mock_db):
        """Should return 404 when document not found."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(f"/api/documents/{uuid.uuid4()}/text")

            assert response.status_code == 404


class TestGetDocumentStats:
    """Tests for get_document_stats endpoint."""

    def test_get_stats_success(self, mock_db):
        """Should return document statistics."""
        stats = {
            "total_documents": 100,
            "by_extraction_status": {
                "completed": 80,
                "pending": 15,
                "failed": 5,
            },
            "by_document_type": {
                "report": 40,
                "invoice": 30,
                "contract": 20,
                "other": 10,
            },
            "total_embeddings": 250,
            "pending_tasks_by_type": {
                "extract_text": 10,
                "generate_embedding": 5,
            },
        }

        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_stats = AsyncMock(return_value=stats)
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get("/api/documents/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["total_documents"] == 100
            assert "by_extraction_status" in data
            assert "by_document_type" in data


class TestResponseModels:
    """Tests for Pydantic response models."""

    def test_document_response_from_orm(self, sample_document):
        """Should convert ORM model to response model."""
        from backend.api.routes.documents import DocumentResponse

        response = DocumentResponse.model_validate(sample_document)

        assert str(response.id) == str(sample_document.id)
        assert response.checksum == sample_document.checksum
        assert response.file_size == sample_document.file_size
        assert response.mime_type == sample_document.mime_type
        assert response.title == sample_document.title

    def test_document_origin_response_from_orm(self, sample_origin):
        """Should convert ORM origin to response model."""
        from backend.api.routes.documents import DocumentOriginResponse

        response = DocumentOriginResponse.model_validate(sample_origin)

        assert str(response.id) == str(sample_origin.id)
        assert response.origin_type == sample_origin.origin_type
        assert response.origin_path == sample_origin.origin_path
        assert response.is_primary == sample_origin.is_primary


class TestValidation:
    """Tests for input validation."""

    def test_invalid_page_number(self, mock_db):
        """Should reject invalid page numbers."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.get("/api/documents", params={"page": 0})

        assert response.status_code == 422  # Validation error

    def test_invalid_page_size(self, mock_db):
        """Should reject invalid page sizes."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.get("/api/documents", params={"page_size": 200})

        assert response.status_code == 422  # Exceeds max of 100

    def test_invalid_min_quality(self, mock_db):
        """Should reject invalid quality values."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.get("/api/documents", params={"min_quality": 1.5})

        assert response.status_code == 422  # Exceeds 1.0

    def test_invalid_document_id_format(self, mock_db):
        """Should reject invalid UUID format."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.get("/api/documents/not-a-uuid")

        assert response.status_code == 422


# =============================================================================
# Phase 4: Search Endpoints
# =============================================================================

class TestSemanticSearchEndpoint:
    """Tests for /search/semantic endpoint."""

    def test_semantic_search_success(self, sample_document, mock_db):
        """Should return search results."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.core.documents.search.DocumentSearchService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.semantic_search = AsyncMock(return_value=[
                (sample_document, 0.85),
            ])
            mock_service_class.return_value = mock_service

            client = TestClient(app)
            response = client.get(
                "/api/documents/search/semantic",
                params={"query": "quarterly report"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "total" in data
            assert "query" in data
            assert data["total"] == 1
            assert data["query"] == "quarterly report"
            assert data["results"][0]["similarity"] == 0.85

    def test_semantic_search_with_date_filter(self, mock_db):
        """Should accept date range filters."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.core.documents.search.DocumentSearchService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.semantic_search = AsyncMock(return_value=[])
            mock_service_class.return_value = mock_service

            client = TestClient(app)
            response = client.get(
                "/api/documents/search/semantic",
                params={
                    "query": "invoices",
                    "date_from": "2024-01-01",
                    "date_to": "2024-12-31",
                }
            )

            assert response.status_code == 200
            # Verify service was called with date params
            mock_service.semantic_search.assert_called_once()
            call_kwargs = mock_service.semantic_search.call_args[1]
            assert call_kwargs["date_from"] is not None
            assert call_kwargs["date_to"] is not None

    def test_semantic_search_with_type_filter(self, mock_db):
        """Should accept document type filter."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.core.documents.search.DocumentSearchService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.semantic_search = AsyncMock(return_value=[])
            mock_service_class.return_value = mock_service

            client = TestClient(app)
            response = client.get(
                "/api/documents/search/semantic",
                params={
                    "query": "contracts",
                    "document_type": "contract",
                }
            )

            assert response.status_code == 200
            call_kwargs = mock_service.semantic_search.call_args[1]
            assert call_kwargs["document_type"] == "contract"

    def test_semantic_search_with_quality_filter(self, mock_db):
        """Should accept minimum quality filter."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.core.documents.search.DocumentSearchService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.semantic_search = AsyncMock(return_value=[])
            mock_service_class.return_value = mock_service

            client = TestClient(app)
            response = client.get(
                "/api/documents/search/semantic",
                params={
                    "query": "reports",
                    "min_quality": 0.8,
                }
            )

            assert response.status_code == 200
            call_kwargs = mock_service.semantic_search.call_args[1]
            assert call_kwargs["min_quality"] == 0.8

    def test_semantic_search_missing_query(self, mock_db):
        """Should require query parameter."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.get("/api/documents/search/semantic")

        assert response.status_code == 422  # Validation error

    def test_semantic_search_empty_query(self, mock_db):
        """Should reject empty query."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        response = client.get(
            "/api/documents/search/semantic",
            params={"query": ""}
        )

        assert response.status_code == 422

    def test_semantic_search_service_error(self, mock_db):
        """Should return 500 on service error."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.core.documents.search.DocumentSearchService") as mock_service_class:
            mock_service = MagicMock()
            mock_service.semantic_search = AsyncMock(
                side_effect=RuntimeError("pgvector not configured")
            )
            mock_service_class.return_value = mock_service

            client = TestClient(app)
            response = client.get(
                "/api/documents/search/semantic",
                params={"query": "test"}
            )

            assert response.status_code == 500
            assert "pgvector" in response.json()["detail"]


class TestFindSimilarEndpoint:
    """Tests for /{document_id}/similar endpoint."""

    def test_find_similar_success(self, sample_document, mock_db):
        """Should return similar documents."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        similar_doc = Document(
            checksum="xyz789",
            file_size=2048,
            mime_type="application/pdf",
            original_filename="related.pdf",
            extraction_status="completed",
            is_deleted=False,
        )
        similar_doc.id = uuid.uuid4()
        similar_doc.created_at = datetime.now()
        similar_doc.first_seen_at = datetime.now()
        similar_doc.last_seen_at = datetime.now()

        # Patch both Repository and SearchService at the same level
        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class, \
             patch("backend.core.documents.search.DocumentSearchService") as mock_service_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=sample_document)
            mock_repo_class.return_value = mock_repo

            mock_service = MagicMock()
            mock_service.find_similar = AsyncMock(return_value=[
                (similar_doc, 0.78),
            ])
            mock_service_class.return_value = mock_service

            client = TestClient(app)
            response = client.get(f"/api/documents/{sample_document.id}/similar")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert data["results"][0]["similarity"] == 0.78

    def test_find_similar_document_not_found(self, mock_db):
        """Should return 404 when document not found."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=None)
            mock_repo_class.return_value = mock_repo

            client = TestClient(app)
            response = client.get(f"/api/documents/{uuid.uuid4()}/similar")

            assert response.status_code == 404

    def test_find_similar_with_filters(self, sample_document, mock_db):
        """Should accept filter parameters."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        # Patch both Repository and SearchService at the same level
        with patch("backend.api.routes.documents.DocumentRepository") as mock_repo_class, \
             patch("backend.core.documents.search.DocumentSearchService") as mock_service_class:
            mock_repo = MagicMock()
            mock_repo.get_by_id = AsyncMock(return_value=sample_document)
            mock_repo_class.return_value = mock_repo

            mock_service = MagicMock()
            mock_service.find_similar = AsyncMock(return_value=[])
            mock_service_class.return_value = mock_service

            client = TestClient(app)
            response = client.get(
                f"/api/documents/{sample_document.id}/similar",
                params={
                    "top_k": 5,
                    "similarity_threshold": 0.8,
                    "same_type_only": True,
                    "date_from": "2024-01-01",
                    "date_to": "2024-06-30",
                }
            )

            assert response.status_code == 200
            call_kwargs = mock_service.find_similar.call_args[1]
            assert call_kwargs["top_k"] == 5
            assert call_kwargs["similarity_threshold"] == 0.8
            assert call_kwargs["same_type_only"] is True

    def test_find_similar_validation(self, sample_document, mock_db):
        """Should validate parameters."""
        app = create_test_app()
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)

        # Invalid top_k
        response = client.get(
            f"/api/documents/{sample_document.id}/similar",
            params={"top_k": 0}
        )
        assert response.status_code == 422

        # Invalid similarity_threshold
        response = client.get(
            f"/api/documents/{sample_document.id}/similar",
            params={"similarity_threshold": 1.5}
        )
        assert response.status_code == 422
