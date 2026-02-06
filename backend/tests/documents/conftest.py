"""
Test fixtures for Document Write API endpoints.

Handles SQLite compatibility for document tables that use PostgreSQL-specific types
(ARRAY, pgvector Vector). Patches these types before table creation.
"""
import os
import uuid
from pathlib import Path
from datetime import datetime

# Load .env BEFORE any other imports (encryption module needs DB_ENCRYPTION_KEY at import time)
from dotenv import load_dotenv

_repo_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_repo_root / ".env")

# Generate test encryption key if not set
if not os.getenv("DB_ENCRYPTION_KEY"):
    from cryptography.fernet import Fernet
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, JSON, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.dialects.postgresql import ARRAY

from backend.api.main import app
from backend.core.database import get_db
from backend.core.database.models import Base
from backend.core.documents.models import (
    Document, DocumentOrigin, ExtractionStatus,
)


def _patch_array_columns_for_sqlite(target, connection, **kw):
    """Replace ARRAY columns with JSON for SQLite compatibility.

    Note: This permanently mutates the column type on the Table metadata.
    This is intentional — SQLAlchemy ORM operations also need JSON type
    (not ARRAY) when talking to SQLite. Safe for tests since no PostgreSQL
    document table tests exist in the same process.
    """
    if connection.dialect.name == "sqlite":
        for col in target.columns:
            if isinstance(col.type, ARRAY):
                col.type = JSON()


# Register listener on document table
event.listen(Document.__table__, "before_create", _patch_array_columns_for_sqlite)


@pytest.fixture(scope="function")
def test_db():
    """Create test database with document tables."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create only document-related tables + emails (for FK on document_origins)
    doc_tables = [
        t for t in Base.metadata.sorted_tables
        if t.name in {
            "documents", "document_origins", "document_embeddings",
            "document_processing_queue", "emails",
        }
    ]
    Base.metadata.create_all(bind=engine, tables=doc_tables)

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(scope="function")
def client(test_db):
    """Create test client with database and auth overrides."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    # Mock verify_api_key to always pass
    async def mock_verify_api_key():
        return "test-api-key"

    from backend.api.auth import verify_api_key

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_api_key] = mock_verify_api_key

    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_document(test_db):
    """Create a sample completed document."""
    doc = Document(
        id=uuid.uuid4(),
        checksum="abc123def456" * 4 + "abcdef1234567890",
        file_size=50000,
        mime_type="application/pdf",
        original_filename="test_paper.pdf",
        page_count=5,
        extraction_status=ExtractionStatus.COMPLETED.value,
        extraction_quality=0.7,
        extraction_method="sandboxed",
        extracted_text="This is the extracted text from a test paper about machine learning. It covers topics such as convolutional neural networks, recurrent architectures, and transformer models for natural language processing tasks.",
        has_images=True,
        has_native_text=True,
        is_image_only=False,
        ocr_applied=False,
        ocr_recommended=True,
        text_source="native",
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)
    return doc


@pytest.fixture
def sample_origin(test_db, sample_document):
    """Create a primary origin for the sample document."""
    origin = DocumentOrigin(
        id=uuid.uuid4(),
        document_id=sample_document.id,
        origin_type="folder",
        origin_host="nvme-pi",
        origin_path="/data/papers/test_paper.pdf",
        origin_filename="test_paper.pdf",
        is_primary=True,
        is_deleted=False,
    )
    test_db.add(origin)
    test_db.commit()
    test_db.refresh(origin)
    return origin


@pytest.fixture
def needs_ocr_document(test_db):
    """Create a document with needs_ocr status."""
    doc = Document(
        id=uuid.uuid4(),
        checksum="needsocr" * 8,
        file_size=10000,
        mime_type="application/pdf",
        original_filename="scanned_doc.pdf",
        page_count=2,
        extraction_status=ExtractionStatus.NEEDS_OCR.value,
        extraction_quality=None,
        has_images=True,
        has_native_text=False,
        is_image_only=True,
        ocr_applied=False,
        ocr_recommended=True,
        text_source="none",
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)
    return doc


@pytest.fixture
def needs_ocr_origin(test_db, needs_ocr_document):
    """Create a primary origin for needs_ocr document."""
    origin = DocumentOrigin(
        id=uuid.uuid4(),
        document_id=needs_ocr_document.id,
        origin_type="folder",
        origin_host="nvme-pi",
        origin_path="/data/scans/scanned_doc.pdf",
        origin_filename="scanned_doc.pdf",
        is_primary=True,
        is_deleted=False,
    )
    test_db.add(origin)
    test_db.commit()
    test_db.refresh(origin)
    return origin


@pytest.fixture
def ocr_recommended_document(test_db):
    """Create a document with ocr_recommended=True but some existing text."""
    doc = Document(
        id=uuid.uuid4(),
        checksum="ocrrecommended" * 4 + "1234567890123456",
        file_size=80000,
        mime_type="application/pdf",
        original_filename="mixed_doc.pdf",
        page_count=10,
        extraction_status=ExtractionStatus.COMPLETED.value,
        extraction_quality=0.4,
        extraction_method="sandboxed",
        extracted_text="Short low quality text.",
        has_images=True,
        has_native_text=True,
        is_image_only=False,
        ocr_applied=False,
        ocr_recommended=True,
        text_source="native",
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)
    return doc


@pytest.fixture
def completed_no_ocr_document(test_db):
    """Create a document that is completed and does NOT need OCR."""
    doc = Document(
        id=uuid.uuid4(),
        checksum="completednoocr" * 4 + "ab12cd34ef567890",
        file_size=30000,
        mime_type="text/plain",
        original_filename="readme.txt",
        extraction_status=ExtractionStatus.COMPLETED.value,
        extraction_quality=0.95,
        extraction_method="sandboxed",
        extracted_text="This is a plain text file with high quality extraction.",
        has_images=False,
        has_native_text=True,
        is_image_only=False,
        ocr_applied=False,
        ocr_recommended=False,
        text_source="native",
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)
    return doc
