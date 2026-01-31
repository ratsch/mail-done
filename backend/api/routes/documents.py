"""
Document management API endpoints

Provides REST API for document indexing feature:
- List and retrieve documents
- Get document origins
- Get document statistics
- Retrieve document content (from origin)
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime
import logging

from backend.api.auth import verify_api_key
from backend.core.database import get_db
from backend.core.documents.models import Document, DocumentOrigin, ExtractionStatus
from backend.core.documents.repository import DocumentRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


# =============================================================================
# Response Models (Pydantic)
# =============================================================================

from pydantic import BaseModel
from typing import Any


class DocumentOriginResponse(BaseModel):
    """Response model for document origin."""
    id: UUID
    origin_type: str
    origin_host: Optional[str]
    origin_path: Optional[str]
    origin_filename: Optional[str]
    email_id: Optional[UUID]
    attachment_index: Optional[int]
    discovered_at: Optional[datetime]
    last_verified_at: Optional[datetime]
    is_primary: bool
    is_deleted: bool

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Response model for document."""
    id: UUID
    checksum: str
    file_size: int
    mime_type: Optional[str]
    original_filename: Optional[str]
    page_count: Optional[int]
    extraction_status: str
    extraction_quality: Optional[float]
    extraction_method: Optional[str]
    title: Optional[str]
    summary: Optional[str]
    document_type: Optional[str]
    document_date: Optional[datetime]
    ai_category: Optional[str]
    ai_tags: Optional[List[str]]
    first_seen_at: Optional[datetime]
    last_seen_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentDetailResponse(BaseModel):
    """Response model for document with origins."""
    document: DocumentResponse
    origins: List[DocumentOriginResponse]


class DocumentListResponse(BaseModel):
    """Response model for paginated document list."""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    pages: int


class DocumentStatsResponse(BaseModel):
    """Response model for document statistics."""
    total_documents: int
    by_extraction_status: dict
    by_document_type: dict
    total_embeddings: int
    pending_tasks_by_type: dict


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=DocumentListResponse, dependencies=[Depends(verify_api_key)])
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    extraction_status: Optional[str] = Query(None, description="Filter by extraction status"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    mime_type: Optional[str] = Query(None, description="Filter by MIME type"),
    min_quality: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum extraction quality"),
    search: Optional[str] = Query(None, description="Search in title and summary"),
    db: Session = Depends(get_db)
):
    """
    List documents with pagination and filters.

    **Filters:**
    - extraction_status: pending, processing, completed, no_content, failed
    - document_type: invoice, contract, report, etc.
    - mime_type: application/pdf, etc.
    - min_quality: Minimum extraction quality (0.0-1.0)
    - search: Search in title and summary
    """
    query = db.query(Document).filter(Document.is_deleted == False)

    # Apply filters
    if extraction_status:
        query = query.filter(Document.extraction_status == extraction_status)

    if document_type:
        query = query.filter(Document.document_type == document_type)

    if mime_type:
        query = query.filter(Document.mime_type == mime_type)

    if min_quality is not None:
        query = query.filter(Document.extraction_quality >= min_quality)

    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (Document.title.ilike(search_pattern)) |
            (Document.summary.ilike(search_pattern)) |
            (Document.original_filename.ilike(search_pattern))
        )

    # Get total count
    total = query.count()

    # Apply pagination
    offset = (page - 1) * page_size
    documents = query.order_by(Document.created_at.desc()).offset(offset).limit(page_size).all()

    # Calculate total pages
    pages = (total + page_size - 1) // page_size

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in documents],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get("/stats", response_model=DocumentStatsResponse, dependencies=[Depends(verify_api_key)])
async def get_document_stats(
    db: Session = Depends(get_db)
):
    """
    Get document indexing statistics.

    Returns counts by extraction status, document type, and pending tasks.
    """
    repo = DocumentRepository(db)
    stats = await repo.get_stats()
    return DocumentStatsResponse(**stats)


@router.get("/by-checksum/{checksum}", response_model=Optional[DocumentResponse], dependencies=[Depends(verify_api_key)])
async def get_document_by_checksum(
    checksum: str,
    db: Session = Depends(get_db)
):
    """
    Get document by SHA-256 checksum.

    Useful for deduplication checks before uploading.
    """
    repo = DocumentRepository(db)
    document = await repo.get_by_checksum(checksum)
    if not document:
        return None
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}", response_model=DocumentDetailResponse, dependencies=[Depends(verify_api_key)])
async def get_document(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get document with all origins.

    Returns document details and list of all origins (where the file was found).
    """
    repo = DocumentRepository(db)
    document = await repo.get_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    origins = await repo.get_origins(document_id)

    return DocumentDetailResponse(
        document=DocumentResponse.model_validate(document),
        origins=[DocumentOriginResponse.model_validate(o) for o in origins],
    )


@router.get("/{document_id}/origins", response_model=List[DocumentOriginResponse], dependencies=[Depends(verify_api_key)])
async def get_document_origins(
    document_id: UUID,
    include_deleted: bool = Query(False, description="Include deleted origins"),
    db: Session = Depends(get_db)
):
    """
    Get all origins for a document.

    An origin is a location where the document was found
    (folder path, email attachment, Google Drive, etc.)
    """
    repo = DocumentRepository(db)

    # Verify document exists
    document = await repo.get_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    origins = await repo.get_origins(document_id, include_deleted=include_deleted)
    return [DocumentOriginResponse.model_validate(o) for o in origins]


@router.get("/{document_id}/text", dependencies=[Depends(verify_api_key)])
async def get_document_text(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get extracted text content of a document.

    Returns the decrypted extracted text if available.
    """
    repo = DocumentRepository(db)
    document = await repo.get_by_id(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.extraction_status != ExtractionStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Text not available. Extraction status: {document.extraction_status}"
        )

    return {
        "document_id": str(document_id),
        "title": document.title,
        "text": document.extracted_text,
        "extraction_quality": document.extraction_quality,
        "extraction_method": document.extraction_method,
    }


# Note: Document content retrieval endpoint (getting the actual binary file)
# will be implemented in Phase 2 when folder scanning and retrieval are added.
# That requires the host configuration and retrieval service.
