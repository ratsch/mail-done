"""
Comprehensive tests for Document Write API endpoints:
- GET /api/documents/pending-ocr
- POST /api/documents/{document_id}/ocr
- PATCH /api/documents/{document_id}/metadata
"""
import uuid
from datetime import date
from unittest.mock import patch, AsyncMock

import pytest

from backend.core.documents.models import Document, DocumentOrigin, ExtractionStatus


# =============================================================================
# GET /api/documents/pending-ocr
# =============================================================================

class TestPendingOCR:
    """Tests for the pending-ocr listing endpoint."""

    def test_pending_ocr_returns_needs_ocr_documents(
        self, client, needs_ocr_document, needs_ocr_origin
    ):
        """Documents with extraction_status=needs_ocr should appear."""
        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        doc_ids = [d["id"] for d in data["documents"]]
        assert str(needs_ocr_document.id) in doc_ids

    def test_pending_ocr_returns_ocr_recommended_documents(
        self, client, ocr_recommended_document
    ):
        """Documents with ocr_recommended=True and ocr_applied=False should appear."""
        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        data = response.json()
        doc_ids = [d["id"] for d in data["documents"]]
        assert str(ocr_recommended_document.id) in doc_ids

    def test_pending_ocr_excludes_completed_no_ocr(
        self, client, completed_no_ocr_document
    ):
        """Documents that don't need OCR should NOT appear."""
        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        data = response.json()
        doc_ids = [d["id"] for d in data["documents"]]
        assert str(completed_no_ocr_document.id) not in doc_ids

    def test_pending_ocr_priority_ordering(
        self, client, needs_ocr_document, needs_ocr_origin, ocr_recommended_document
    ):
        """needs_ocr documents should come before ocr_recommended documents."""
        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        data = response.json()
        docs = data["documents"]

        # Find positions
        needs_ocr_idx = None
        recommended_idx = None
        for i, d in enumerate(docs):
            if d["id"] == str(needs_ocr_document.id):
                needs_ocr_idx = i
            if d["id"] == str(ocr_recommended_document.id):
                recommended_idx = i

        assert needs_ocr_idx is not None
        assert recommended_idx is not None
        assert needs_ocr_idx < recommended_idx

    def test_pending_ocr_includes_primary_origin(
        self, client, needs_ocr_document, needs_ocr_origin
    ):
        """Response should include primary origin info for file retrieval."""
        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        data = response.json()

        doc = next(d for d in data["documents"] if d["id"] == str(needs_ocr_document.id))
        assert doc["primary_origin"] is not None
        assert doc["primary_origin"]["host"] == "nvme-pi"
        assert doc["primary_origin"]["path"] == "/data/scans/scanned_doc.pdf"
        assert doc["primary_origin"]["filename"] == "scanned_doc.pdf"

    def test_pending_ocr_no_origin(self, client, needs_ocr_document):
        """Documents without primary origin should have null origin."""
        # No origin fixture loaded
        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        data = response.json()

        doc = next(d for d in data["documents"] if d["id"] == str(needs_ocr_document.id))
        assert doc["primary_origin"] is None

    def test_pending_ocr_filter_mime_type(
        self, client, test_db, needs_ocr_document, needs_ocr_origin
    ):
        """mime_type filter should narrow results."""
        # needs_ocr_document is application/pdf
        response = client.get("/api/documents/pending-ocr?mime_type=application/pdf")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert all(d["mime_type"] == "application/pdf" for d in data["documents"])

        # Filter for non-matching type
        response = client.get("/api/documents/pending-ocr?mime_type=image/png")
        assert response.status_code == 200
        data = response.json()
        doc_ids = [d["id"] for d in data["documents"]]
        assert str(needs_ocr_document.id) not in doc_ids

    def test_pending_ocr_filter_include_needs_ocr_only(
        self, client, needs_ocr_document, needs_ocr_origin, ocr_recommended_document
    ):
        """include_ocr_recommended=false should only return needs_ocr docs."""
        response = client.get(
            "/api/documents/pending-ocr?include_ocr_recommended=false"
        )
        assert response.status_code == 200
        data = response.json()
        doc_ids = [d["id"] for d in data["documents"]]
        assert str(needs_ocr_document.id) in doc_ids
        assert str(ocr_recommended_document.id) not in doc_ids

    def test_pending_ocr_filter_include_ocr_recommended_only(
        self, client, test_db, ocr_recommended_document
    ):
        """include_needs_ocr_status=false should only return ocr_recommended docs.

        Note: needs_ocr documents also have ocr_recommended=True, so they still
        match the ocr_recommended filter. Create a doc that is ONLY needs_ocr status
        (without ocr_recommended) to verify it's excluded.
        """
        # Create a needs_ocr doc WITHOUT ocr_recommended
        doc_no_rec = Document(
            id=uuid.uuid4(),
            checksum="needs_ocr_no_rec" * 4,
            file_size=5000,
            mime_type="application/pdf",
            original_filename="no_rec.pdf",
            extraction_status=ExtractionStatus.NEEDS_OCR.value,
            ocr_recommended=False,
            ocr_applied=False,
        )
        test_db.add(doc_no_rec)
        test_db.commit()

        response = client.get(
            "/api/documents/pending-ocr?include_needs_ocr_status=false"
        )
        assert response.status_code == 200
        data = response.json()
        doc_ids = [d["id"] for d in data["documents"]]
        # The needs_ocr doc without ocr_recommended should be excluded
        assert str(doc_no_rec.id) not in doc_ids
        # The ocr_recommended doc should be included
        assert str(ocr_recommended_document.id) in doc_ids

    def test_pending_ocr_both_filters_false(self, client, needs_ocr_document):
        """Both filters false should return empty results."""
        response = client.get(
            "/api/documents/pending-ocr"
            "?include_needs_ocr_status=false&include_ocr_recommended=false"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["documents"] == []

    def test_pending_ocr_limit(self, client, test_db):
        """Limit parameter should cap results."""
        # Create 5 needs_ocr documents
        for i in range(5):
            doc = Document(
                id=uuid.uuid4(),
                checksum=f"limit_test_{i:04d}" * 4 + f"{i:016d}",
                file_size=1000 * (i + 1),
                mime_type="application/pdf",
                original_filename=f"doc_{i}.pdf",
                extraction_status=ExtractionStatus.NEEDS_OCR.value,
                is_image_only=True,
                ocr_recommended=True,
                ocr_applied=False,
            )
            test_db.add(doc)
        test_db.commit()

        response = client.get("/api/documents/pending-ocr?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 3
        assert data["total"] == 5

    def test_pending_ocr_document_fields(self, client, needs_ocr_document, needs_ocr_origin):
        """Response documents should have all expected fields."""
        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        data = response.json()

        doc = next(d for d in data["documents"] if d["id"] == str(needs_ocr_document.id))
        assert "id" in doc
        assert "original_filename" in doc
        assert "mime_type" in doc
        assert "file_size" in doc
        assert "page_count" in doc
        assert "extraction_status" in doc
        assert "extraction_quality" in doc
        assert "has_native_text" in doc
        assert "is_image_only" in doc
        assert "ocr_recommended" in doc
        assert "ocr_applied" in doc
        assert "primary_origin" in doc

    def test_pending_ocr_excludes_already_ocrd(self, client, test_db):
        """Documents with ocr_applied=True should NOT appear even if ocr_recommended."""
        doc = Document(
            id=uuid.uuid4(),
            checksum="already_ocrd_" * 4 + "1234567890123456",
            file_size=20000,
            mime_type="application/pdf",
            original_filename="already_ocrd.pdf",
            extraction_status=ExtractionStatus.COMPLETED.value,
            extraction_quality=0.9,
            ocr_applied=True,
            ocr_recommended=True,  # Still recommended but already applied
        )
        test_db.add(doc)
        test_db.commit()

        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        data = response.json()
        doc_ids = [d["id"] for d in data["documents"]]
        assert str(doc.id) not in doc_ids

    def test_pending_ocr_excludes_deleted(self, client, test_db):
        """Deleted documents should not appear."""
        doc = Document(
            id=uuid.uuid4(),
            checksum="deleted_doc_" * 4 + "ab12cd34ef567890",
            file_size=5000,
            mime_type="application/pdf",
            original_filename="deleted.pdf",
            extraction_status=ExtractionStatus.NEEDS_OCR.value,
            is_deleted=True,
            ocr_recommended=True,
            ocr_applied=False,
        )
        test_db.add(doc)
        test_db.commit()

        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        data = response.json()
        doc_ids = [d["id"] for d in data["documents"]]
        assert str(doc.id) not in doc_ids


# =============================================================================
# POST /api/documents/{document_id}/ocr
# =============================================================================

class TestSubmitOCR:
    """Tests for the OCR submission endpoint."""

    def test_submit_ocr_basic(self, client, needs_ocr_document):
        """Basic OCR submission should update document."""
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "This is OCR-extracted text from the scanned document. " * 10,
                "method": "ocr_tesseract",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert data["document_id"] == str(needs_ocr_document.id)
        assert data["embeddings_queued"] is True
        assert data["extraction_quality"] is not None
        assert data["previous_quality"] is None  # Was None before

    def test_submit_ocr_with_quality(self, client, needs_ocr_document):
        """OCR submission with explicit quality score."""
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "High quality OCR text content here. " * 20,
                "method": "ocr_claude",
                "quality": 0.95,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert data["extraction_quality"] == pytest.approx(0.95)

    def test_submit_ocr_with_pages(self, client, needs_ocr_document):
        """OCR submission with per-page structure."""
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "Page 1 text. Page 2 text.",
                "method": "ocr_claude",
                "quality": 0.9,
                "pages": [
                    {"page": 1, "text": "Page 1 text."},
                    {"page": 2, "text": "Page 2 text."},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True

    def test_submit_ocr_quality_comparison_rejects_lower(
        self, client, sample_document
    ):
        """OCR with lower quality than existing should be rejected."""
        # sample_document has quality=0.7
        response = client.post(
            f"/api/documents/{sample_document.id}/ocr",
            json={
                "text": "Low quality OCR text.",
                "method": "ocr_tesseract",
                "quality": 0.3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is False
        assert data["embeddings_queued"] is False
        assert data["extraction_quality"] == pytest.approx(0.7)  # Unchanged
        assert data["previous_quality"] == pytest.approx(0.7)

    def test_submit_ocr_quality_comparison_accepts_higher(
        self, client, sample_document
    ):
        """OCR with higher quality than existing should be accepted."""
        # sample_document has quality=0.7
        response = client.post(
            f"/api/documents/{sample_document.id}/ocr",
            json={
                "text": "High quality OCR output with detailed content. " * 20,
                "method": "ocr_claude",
                "quality": 0.95,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert data["extraction_quality"] == pytest.approx(0.95)
        assert data["previous_quality"] == pytest.approx(0.7)

    def test_submit_ocr_force_overrides_quality(self, client, sample_document):
        """force=True should update even with lower quality."""
        response = client.post(
            f"/api/documents/{sample_document.id}/ocr",
            json={
                "text": "Forced low quality OCR text. " * 10,
                "method": "ocr_tesseract",
                "quality": 0.2,
                "force": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert data["extraction_quality"] == pytest.approx(0.2)
        assert data["previous_quality"] == pytest.approx(0.7)

    def test_submit_ocr_empty_text_updates_short_existing(
        self, client, ocr_recommended_document
    ):
        """Document with very short existing text should accept new OCR."""
        # ocr_recommended_document has "Short low quality text." (< 100 chars)
        response = client.post(
            f"/api/documents/{ocr_recommended_document.id}/ocr",
            json={
                "text": "Much better OCR text extracted from document. " * 10,
                "method": "ocr_claude",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True

    def test_submit_ocr_document_not_found(self, client):
        """OCR submission for non-existent document should 404."""
        fake_id = uuid.uuid4()
        response = client.post(
            f"/api/documents/{fake_id}/ocr",
            json={
                "text": "Some OCR text.",
                "method": "ocr_tesseract",
            },
        )
        assert response.status_code == 404

    def test_submit_ocr_auto_quality_calculation(self, client, needs_ocr_document):
        """When quality is not provided, it should be auto-calculated."""
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "This is a reasonably long OCR text that should produce a decent quality score. " * 5,
                "method": "ocr_tesseract",
                # No quality specified
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert data["extraction_quality"] is not None
        assert 0.0 <= data["extraction_quality"] <= 1.0

    def test_submit_ocr_updates_extraction_status_to_completed(
        self, client, test_db, needs_ocr_document
    ):
        """After successful OCR, status should be 'completed'."""
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "OCR extracted text content. " * 10,
                "method": "ocr_claude",
                "quality": 0.85,
            },
        )
        assert response.status_code == 200
        assert response.json()["updated"] is True

        # Verify in DB
        test_db.refresh(needs_ocr_document)
        assert needs_ocr_document.extraction_status == ExtractionStatus.COMPLETED.value
        assert needs_ocr_document.extraction_method == "ocr_claude"
        assert needs_ocr_document.extraction_quality == pytest.approx(0.85)

    def test_submit_ocr_default_method(self, client, needs_ocr_document):
        """Default method should be 'ocr'."""
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "Some OCR text content for testing. " * 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True

    def test_submit_ocr_invalid_document_id(self, client):
        """Invalid UUID should return 422."""
        response = client.post(
            "/api/documents/not-a-uuid/ocr",
            json={"text": "Some text.", "method": "ocr"},
        )
        assert response.status_code == 422

    def test_submit_ocr_missing_text(self, client, needs_ocr_document):
        """Missing required 'text' field should return 422."""
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={"method": "ocr_tesseract"},
        )
        assert response.status_code == 422

    def test_submit_ocr_sets_ocr_applied(self, client, test_db, needs_ocr_document):
        """After successful OCR, ocr_applied should be True."""
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "OCR extracted content. " * 10,
                "method": "ocr_claude",
                "quality": 0.9,
            },
        )
        assert response.status_code == 200
        assert response.json()["updated"] is True

        test_db.refresh(needs_ocr_document)
        assert needs_ocr_document.ocr_applied is True
        assert needs_ocr_document.ocr_pipeline_version == "ocr_claude"
        assert needs_ocr_document.text_source == "ocr"

    def test_submit_ocr_rejected_does_not_set_ocr_applied(
        self, client, test_db, sample_document
    ):
        """When OCR is rejected (lower quality), ocr_applied should remain False."""
        response = client.post(
            f"/api/documents/{sample_document.id}/ocr",
            json={
                "text": "Low quality text.",
                "method": "ocr_tesseract",
                "quality": 0.1,
            },
        )
        assert response.status_code == 200
        assert response.json()["updated"] is False

        test_db.refresh(sample_document)
        assert sample_document.ocr_applied is False

    def test_submit_ocr_quality_out_of_bounds(self, client, needs_ocr_document):
        """Quality values outside 0-1 should return 422."""
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={"text": "Some text.", "quality": 1.5},
        )
        assert response.status_code == 422

        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={"text": "Some text.", "quality": -0.1},
        )
        assert response.status_code == 422


# =============================================================================
# PATCH /api/documents/{document_id}/metadata
# =============================================================================

class TestUpdateMetadata:
    """Tests for the metadata update endpoint."""

    def test_update_title(self, client, sample_document):
        """Updating only title should work."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"title": "Machine Learning Paper on CNNs"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Machine Learning Paper on CNNs"

    def test_update_summary(self, client, sample_document):
        """Updating summary should work."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"summary": "A comprehensive study of convolutional neural networks."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "A comprehensive study of convolutional neural networks."

    def test_update_document_type(self, client, sample_document):
        """Updating document_type should work."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"document_type": "research_paper"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "research_paper"

    def test_update_document_date(self, client, sample_document):
        """Updating document_date should work."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"document_date": "2024-06-15"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["document_date"] is not None
        assert "2024-06-15" in data["document_date"]

    def test_update_language(self, client, sample_document):
        """Updating language should work."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"language": "en"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "en"

    def test_update_ai_category(self, client, sample_document):
        """Updating ai_category should work."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"ai_category": "academic_paper"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ai_category"] == "academic_paper"

    def test_update_ai_tags(self, client, sample_document):
        """Updating ai_tags should work."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"ai_tags": ["machine-learning", "cnn", "deep-learning"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ai_tags"] == ["machine-learning", "cnn", "deep-learning"]

    def test_update_multiple_fields(self, client, sample_document):
        """Updating multiple fields at once should work."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={
                "title": "Updated Title",
                "summary": "Updated summary text.",
                "document_type": "invoice",
                "language": "de",
                "ai_category": "financial",
                "ai_tags": ["invoice", "payment"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["summary"] == "Updated summary text."
        assert data["document_type"] == "invoice"
        assert data["language"] == "de"
        assert data["ai_category"] == "financial"
        assert data["ai_tags"] == ["invoice", "payment"]

    def test_update_preserves_unset_fields(self, client, test_db, sample_document):
        """Fields not in the request should remain unchanged."""
        # First set title
        client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"title": "My Title"},
        )

        # Then set summary (title should remain)
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"summary": "My Summary"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "My Title"  # Preserved
        assert data["summary"] == "My Summary"  # Updated

    def test_update_metadata_document_not_found(self, client):
        """Updating non-existent document should 404."""
        fake_id = uuid.uuid4()
        response = client.patch(
            f"/api/documents/{fake_id}/metadata",
            json={"title": "Some Title"},
        )
        assert response.status_code == 404

    def test_update_metadata_empty_body(self, client, sample_document):
        """Empty update body should return 400."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={},
        )
        assert response.status_code == 400
        assert "No fields" in response.json()["detail"]

    def test_update_metadata_returns_full_document_response(
        self, client, sample_document
    ):
        """Response should be a full DocumentResponse with all standard fields."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"title": "Test Document"},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify standard DocumentResponse fields are present
        assert "id" in data
        assert "checksum" in data
        assert "file_size" in data
        assert "mime_type" in data
        assert "extraction_status" in data
        assert "extraction_quality" in data
        assert "created_at" in data

    def test_update_metadata_invalid_document_id(self, client):
        """Invalid UUID should return 422."""
        response = client.patch(
            "/api/documents/not-a-uuid/metadata",
            json={"title": "Some Title"},
        )
        assert response.status_code == 422

    def test_update_metadata_clear_field_with_null(self, client, test_db, sample_document):
        """Setting a field to null should clear it."""
        # First set title
        client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"title": "To Be Cleared"},
        )

        # Then clear it
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"title": None},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] is None

    def test_update_metadata_does_not_affect_extraction(
        self, client, test_db, sample_document
    ):
        """Metadata updates should not change extraction fields."""
        original_quality = sample_document.extraction_quality
        original_method = sample_document.extraction_method
        original_status = sample_document.extraction_status

        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={
                "title": "New Title",
                "ai_category": "test_category",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["extraction_quality"] == pytest.approx(original_quality)
        assert data["extraction_method"] == original_method
        assert data["extraction_status"] == original_status

    def test_update_metadata_title_too_long(self, client, sample_document):
        """Title exceeding max length should return 422."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"title": "x" * 501},
        )
        assert response.status_code == 422

    def test_update_metadata_language_too_long(self, client, sample_document):
        """Language exceeding max length should return 422."""
        response = client.patch(
            f"/api/documents/{sample_document.id}/metadata",
            json={"language": "x" * 11},
        )
        assert response.status_code == 422


# =============================================================================
# Integration / Cross-endpoint tests
# =============================================================================

class TestWriteAPIIntegration:
    """Cross-endpoint integration tests."""

    def test_pending_ocr_then_submit_ocr_then_verify_gone(
        self, client, test_db, needs_ocr_document, needs_ocr_origin
    ):
        """Full workflow: list pending → submit OCR → verify no longer pending."""
        # Step 1: Verify document appears in pending list
        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        doc_ids = [d["id"] for d in response.json()["documents"]]
        assert str(needs_ocr_document.id) in doc_ids

        # Step 2: Submit OCR
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "Full OCR text from the scanned document. " * 20,
                "method": "ocr_claude",
                "quality": 0.9,
            },
        )
        assert response.status_code == 200
        assert response.json()["updated"] is True

        # Step 3: Document should no longer appear in pending-ocr at all
        # - extraction_status changed from needs_ocr → completed (excludes from needs_ocr filter)
        # - ocr_applied set to True (excludes from ocr_recommended filter)
        response = client.get("/api/documents/pending-ocr")
        assert response.status_code == 200
        doc_ids = [d["id"] for d in response.json()["documents"]]
        assert str(needs_ocr_document.id) not in doc_ids

    def test_submit_ocr_then_update_metadata(
        self, client, test_db, needs_ocr_document
    ):
        """Submit OCR and then enrich with metadata."""
        # Step 1: Submit OCR
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "Invoice from Acme Corp for services rendered. Total: $1,500.00. " * 5,
                "method": "ocr_claude",
                "quality": 0.85,
            },
        )
        assert response.status_code == 200
        assert response.json()["updated"] is True

        # Step 2: Update metadata
        response = client.patch(
            f"/api/documents/{needs_ocr_document.id}/metadata",
            json={
                "title": "Invoice - Acme Corp",
                "document_type": "invoice",
                "language": "en",
                "ai_category": "financial",
                "ai_tags": ["invoice", "acme-corp"],
                "document_date": "2024-01-15",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Invoice - Acme Corp"
        assert data["document_type"] == "invoice"
        assert data["extraction_status"] == "completed"
        assert data["extraction_quality"] == pytest.approx(0.85)

    def test_multiple_ocr_submissions_quality_wins(
        self, client, test_db, needs_ocr_document
    ):
        """Multiple OCR submissions: higher quality should win."""
        # First submission (low quality)
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "Low quality OCR text." * 10,
                "method": "ocr_tesseract",
                "quality": 0.5,
            },
        )
        assert response.status_code == 200
        assert response.json()["updated"] is True

        # Second submission (higher quality, should win)
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "Much better OCR text with proper formatting. " * 20,
                "method": "ocr_claude",
                "quality": 0.9,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert data["extraction_quality"] == pytest.approx(0.9)
        assert data["previous_quality"] == pytest.approx(0.5)

        # Third submission (lower quality, should be rejected)
        response = client.post(
            f"/api/documents/{needs_ocr_document.id}/ocr",
            json={
                "text": "Even worse OCR text." * 5,
                "method": "ocr_tesseract",
                "quality": 0.3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is False
        assert data["extraction_quality"] == pytest.approx(0.9)  # Unchanged
