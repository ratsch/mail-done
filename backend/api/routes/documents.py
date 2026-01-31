"""
Document management API endpoints

Provides REST API for document indexing feature:
- List and retrieve documents
- Get document origins
- Get document statistics
- Retrieve document content (from origin)
- Semantic search over documents (Phase 4)
- Find similar documents (Phase 4)
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime, date
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


class DocumentSearchResult(BaseModel):
    """Response model for a single search result."""
    document: DocumentResponse
    similarity: float


class DocumentSearchResponse(BaseModel):
    """Response model for document search results."""
    results: List[DocumentSearchResult]
    total: int
    query: str


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


@router.get("/{document_id}/content", dependencies=[Depends(verify_api_key)])
async def get_document_content(
    document_id: UUID,
    origin_index: int = Query(0, ge=0, description="Origin index to retrieve from (0=primary)"),
    fallback: bool = Query(True, description="Try other origins if first fails"),
    db: Session = Depends(get_db)
):
    """
    Retrieve the original document binary content.

    Fetches the file from its stored origin (filesystem, email attachment, etc.).
    If the primary origin is unavailable, can fall back to secondary origins.

    **Response:**
    - Returns binary file with appropriate Content-Type
    - Content-Disposition header for download with original filename
    """
    from backend.core.documents.retrieval import DocumentRetrievalService, RetrievalError

    repo = DocumentRepository(db)
    document = await repo.get_by_id(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    retrieval = DocumentRetrievalService(repo)

    try:
        content = await retrieval.get_content(
            document=document,
            origin_index=origin_index,
            fallback=fallback,
        )
    except RetrievalError as e:
        logger.warning(f"Failed to retrieve document {document_id}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Document content not accessible: {str(e)}"
        )

    # Get content type
    content_type = retrieval.get_content_type(document)

    # Build filename for Content-Disposition
    filename = document.original_filename or f"document_{document_id}"

    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(content)),
        }
    )


@router.get("/{document_id}/verify-origins", dependencies=[Depends(verify_api_key)])
async def verify_document_origins(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Verify accessibility of all origins for a document.

    Checks each origin to see if it's still accessible and updates
    the last_verified_at timestamp for accessible origins.

    Returns a map of origin_id to accessibility status.
    """
    from backend.core.documents.retrieval import DocumentRetrievalService

    repo = DocumentRepository(db)
    document = await repo.get_by_id(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    retrieval = DocumentRetrievalService(repo)
    results = await retrieval.verify_all_origins(document)

    return {
        "document_id": str(document_id),
        "origins": results,
    }


# =============================================================================
# Search Endpoints (Phase 4)
# =============================================================================

@router.get("/search/semantic", response_model=DocumentSearchResponse, dependencies=[Depends(verify_api_key)])
async def search_documents_semantic(
    query: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    similarity_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum similarity score"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    mime_type: Optional[str] = Query(None, description="Filter by MIME type"),
    min_quality: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum extraction quality"),
    date_from: Optional[date] = Query(None, description="Documents dated on or after (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Documents dated on or before (YYYY-MM-DD)"),
    ai_category: Optional[str] = Query(None, description="Filter by AI category"),
    db: Session = Depends(get_db)
):
    """
    Semantic search over indexed documents.

    Uses vector embeddings to find documents semantically similar to the query.
    Supports filtering by document type, date range, quality, and more.

    **Time Horizon:**
    - Use date_from and date_to to limit search to a specific time period
    - Document dates are based on the document_date field (extracted from content)

    **Example queries:**
    - "invoices from Q4 2024" with date_from=2024-10-01, date_to=2024-12-31
    - "contracts mentioning renewal terms"
    - "research papers on machine learning"
    """
    from backend.core.documents.search import DocumentSearchService

    search_service = DocumentSearchService(db)

    try:
        results = await search_service.semantic_search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_type=document_type,
            mime_type=mime_type,
            min_quality=min_quality,
            date_from=date_from,
            date_to=date_to,
            ai_category=ai_category,
        )
    except RuntimeError as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return DocumentSearchResponse(
        results=[
            DocumentSearchResult(
                document=DocumentResponse.model_validate(doc),
                similarity=similarity,
            )
            for doc, similarity in results
        ],
        total=len(results),
        query=query,
    )


@router.get("/{document_id}/similar", response_model=DocumentSearchResponse, dependencies=[Depends(verify_api_key)])
async def find_similar_documents(
    document_id: UUID,
    top_k: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    similarity_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum similarity score"),
    same_type_only: bool = Query(False, description="Only return documents of the same type"),
    date_from: Optional[date] = Query(None, description="Documents dated on or after (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Documents dated on or before (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """
    Find documents similar to a reference document.

    Returns documents with similar content based on vector embeddings.

    **Use cases:**
    - Find related contracts or invoices
    - Discover duplicate or near-duplicate documents
    - Group similar documents by content

    **Time Horizon:**
    - Use date_from and date_to to limit results to a specific time period
    """
    from backend.core.documents.search import DocumentSearchService

    repo = DocumentRepository(db)

    # Verify document exists
    document = await repo.get_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    search_service = DocumentSearchService(db)

    try:
        results = await search_service.find_similar(
            document_id=document_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            same_type_only=same_type_only,
            date_from=date_from,
            date_to=date_to,
        )
    except RuntimeError as e:
        logger.error(f"Similar search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return DocumentSearchResponse(
        results=[
            DocumentSearchResult(
                document=DocumentResponse.model_validate(doc),
                similarity=similarity,
            )
            for doc, similarity in results
        ],
        total=len(results),
        query=f"similar to {document_id}",
    )
