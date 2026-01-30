"""
Advanced Search API Endpoints

Provides semantic and hybrid search capabilities using pgvector.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
import logging

from backend.api.auth import verify_api_key
from backend.core.database import get_db
from backend.core.search import EmbeddingGenerator, VectorSearch, HybridSearch
from backend.api.schemas import EmailResponse

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/search", tags=["search"])


class SearchResult(BaseModel):
    """Search result with email and score"""
    email: EmailResponse
    score: float
    score_breakdown: Optional[dict] = None
    
    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """Search response"""
    query: str
    mode: str
    total: int
    results: List[SearchResult]
    
    class Config:
        from_attributes = True


@router.get("", dependencies=[Depends(verify_api_key)])
async def search_emails(
    q: str = Query(..., description="Search query"),
    mode: str = Query("hybrid", description="Search mode: keyword/semantic/hybrid"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    vip_only: bool = Query(False, description="Only VIP emails"),
    needs_reply: Optional[bool] = Query(None, description="Filter by reply status"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD or ISO 8601)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD or ISO 8601)"),
    similarity_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Min similarity (semantic mode)"),
    exclude_handled: bool = Query(False, description="Exclude handled emails"),
    exclude_spam: bool = Query(True, description="Exclude spam-marked emails"),
    account_filter: Optional[str] = Query(None, description="Filter by email account (for multi-account setups)"),
    sender: Optional[str] = Query(None, description="Filter by sender email or domain"),
    db: Session = Depends(get_db)
):
    """
    Advanced email search with semantic and hybrid capabilities.
    
    **Modes:**
    - `keyword`: Traditional text search in subject/body
    - `semantic`: Vector similarity search (finds related concepts)
    - `hybrid`: Combines keyword + semantic (recommended)
    
    **Examples:**
    ```
    # Semantic search
    GET /api/search?q=machine learning genomics&mode=semantic
    
    # Hybrid search with filters
    GET /api/search?q=PhD applications&mode=hybrid&category=application-phd&vip_only=true
    
    # Find unanswered emails about topic
    GET /api/search?q=grant invitations&needs_reply=true&date_from=2024-01-01
    ```
    
    **Hybrid Search Benefits:**
    - Finds exact keyword matches (high precision)
    - Finds semantically similar emails (high recall)
    - Combines scores intelligently
    - Boosts emails appearing in both results
    """
    # Validate mode
    valid_modes = ['keyword', 'semantic', 'hybrid']
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {', '.join(valid_modes)}"
        )
    
    try:
        # Parse date strings to datetime objects if provided
        date_from_dt = None
        date_to_dt = None
        
        if date_from:
            try:
                # Try ISO format first
                date_from_dt = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            except:
                try:
                    # Try date only format (YYYY-MM-DD)
                    date_from_dt = datetime.strptime(date_from, '%Y-%m-%d')
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid date_from format: {date_from}")
        
        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            except:
                try:
                    date_to_dt = datetime.strptime(date_to, '%Y-%m-%d')
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid date_to format: {date_to}")
        
        # Create hybrid search
        searcher = HybridSearch(db)
        
        # **CRITICAL OPTIMIZATION: Pass sender and account_id to HybridSearch**
        # This allows the database to filter BEFORE searching (10x faster)
        # The old approach: search 557K emails, then filter → slow
        # The new approach: filter to ~50K emails first, then search → fast!
        
        # **PAGINATION FIX: We need to know the total count for proper pagination**
        # For now, fetch enough results to cover multiple pages
        # When sender/account filters are active, we know the result set is small, so fetch more
        if sender or account_filter:
            # Filtered queries return smaller result sets - safe to fetch more
            total_to_fetch = page * page_size * 5  # Fetch 5x for filtered queries
            total_to_fetch = min(total_to_fetch, 2000)  # Cap at 2000 for safety
        else:
            # Unfiltered queries could be huge - be more conservative
            total_to_fetch = page * page_size * 2
            total_to_fetch = min(total_to_fetch, 500)  # Cap at 500 for unfiltered
        
        # Perform search with all filters applied in the search layer
        # Filters are applied at DB level BEFORE text/semantic search
        results = await searcher.search(
            query=q,
            mode=mode,
            top_k=total_to_fetch,  # Fetch enough for pagination
            category=category,
            vip_only=vip_only,
            needs_reply=needs_reply,
            date_from=date_from_dt,
            date_to=date_to_dt,
            similarity_threshold=similarity_threshold,
            sender=sender,  # ← Applied at DB level (uses index!)
            account_id=account_filter  # ← Applied at DB level (uses index!)
        )
        
        # Filter results based on exclude flags (these can't be done in SQL easily)
        all_filtered_results = []
        for email, score, breakdown in results:
            metadata = email.email_metadata
            
            # Check if should be excluded
            if exclude_handled and metadata:
                is_handled = ('handled' in (metadata.project_tags or [])) or ('handled' in (metadata.user_tags or []))
                if is_handled:
                    continue
            
            if exclude_spam and metadata:
                is_spam = 'user-spam' in (metadata.user_tags or [])
                if is_spam:
                    continue
            
            all_filtered_results.append((email, score, breakdown))
        
        # Apply pagination
        total_results = len(all_filtered_results)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = all_filtered_results[start_idx:end_idx]
        
        # Format results
        search_results = [
            SearchResult(
                email=EmailResponse.model_validate(email),
                score=score,
                score_breakdown=breakdown if mode == "hybrid" else None
            )
            for email, score, breakdown in paginated_results
        ]
        
        # Calculate total pages
        total = total_results
        pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        return {
            "query": q,
            "mode": mode,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "results": search_results
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Search failed. Please try again or contact support."
        )


@router.get("/similar/{email_id}", dependencies=[Depends(verify_api_key)])
async def find_similar_emails(
    email_id: UUID,
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    same_category_only: bool = Query(False, description="Only same category"),
    exclude_same_sender: bool = Query(False, description="Exclude same sender"),
    db: Session = Depends(get_db)
):
    """
    Find emails similar to a given email using vector similarity.
    
    **Use Cases:**
    - "Find other emails like this grant invitation"
    - "Show similar PhD applications"
    - "Group related correspondence"
    
    **Example:**
    ```
    GET /api/search/similar/{email-id}?top_k=10&same_category_only=true
    ```
    """
    try:
        searcher = VectorSearch(db)
        
        results = await searcher.find_similar_emails(
            email_id=email_id,
            top_k=top_k,
            same_category_only=same_category_only,
            exclude_same_sender=exclude_same_sender
        )
        
        search_results = [
            SearchResult(
                email=EmailResponse.model_validate(email),
                score=score,
                score_breakdown={'similarity': score}
            )
            for email, score in results
        ]
        
        return {
            "reference_email_id": str(email_id),
            "total": len(search_results),
            "results": search_results
        }
        
    except Exception as e:
        logger.error(f"Error finding similar emails: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to find similar emails. Please try again."
        )


@router.get("/topics", dependencies=[Depends(verify_api_key)])
async def search_by_topic(
    topic: str = Query(..., description="Research topic or concept"),
    categories: Optional[str] = Query(None, description="Comma-separated categories"),
    top_k: int = Query(20, ge=1, le=100),
    similarity_threshold: float = Query(0.6, ge=0.0, le=1.0),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD or ISO format)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD or ISO format)"),
    db: Session = Depends(get_db)
):
    """
    Search emails by research topic or concept.
    
    **Examples:**
    ```
    # Find all emails about a topic
    GET /api/search/topics?topic=reinforcement learning
    
    # Find topic in specific categories
    GET /api/search/topics?topic=genomics&categories=application-phd,invitation-speaking
    
    # Find topic within date range
    GET /api/search/topics?topic=machine learning&date_from=2024-01-01
    ```
    
    **Semantic Understanding:**
    - "machine learning" finds: deep learning, neural networks, AI
    - "genomics" finds: genetics, DNA sequencing, bioinformatics
    - Understands related concepts and synonyms
    """
    try:
        searcher = VectorSearch(db)
        
        # Parse categories
        category_list = None
        if categories:
            category_list = [c.strip() for c in categories.split(',')]
        
        # Search by topic
        results = await searcher.search_by_topic(
            topic=topic,
            categories=category_list,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            date_from=date_from,
            date_to=date_to
        )
        
        search_results = [
            SearchResult(
                email=EmailResponse.model_validate(email),
                score=score,
                score_breakdown={'similarity': score, 'topic': topic}
            )
            for email, score in results
        ]
        
        return {
            "topic": topic,
            "categories": category_list,
            "total": len(search_results),
            "results": search_results
        }
        
    except Exception as e:
        logger.error(f"Topic search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Topic search failed. Please try again."
        )


@router.post("/embeddings/generate", dependencies=[Depends(verify_api_key)])
async def generate_embeddings(
    background_tasks: BackgroundTasks,
    force_regenerate: bool = Query(False, description="Regenerate all embeddings"),
    batch_size: int = Query(100, ge=10, le=500, description="Batch size"),
    db: Session = Depends(get_db)
):
    """
    Generate embeddings for all emails (admin endpoint).
    
    **Warning:** For 2M emails, this takes ~6 hours and costs ~$40.
    
    **Query Parameters:**
    - force_regenerate: Regenerate even if embeddings exist
    - batch_size: Number of emails per API call (default: 100)
    
    **Process:**
    1. Runs in background
    2. Processes in batches
    3. Skips emails with existing embeddings (unless force_regenerate)
    4. Updates progress in logs
    
    **Cost Estimate:**
    - 10K emails: $0.20, ~5 minutes
    - 100K emails: $2, ~30 minutes
    - 1M emails: $20, ~3 hours
    - 2M emails: $40, ~6 hours
    
    **Returns job info** (actual processing happens in background)
    """
    def embed_task():
        """Background task to generate embeddings."""
        generator = EmbeddingGenerator(batch_size=batch_size)
        stats = generator.embed_all_emails(
            db=db,
            skip_existing=not force_regenerate,
            force_regenerate=force_regenerate
        )
        print(f"Embedding generation complete: {stats}")
    
    # Add to background tasks
    background_tasks.add_task(embed_task)
    
    return {
        "status": "started",
        "message": "Embedding generation started in background",
        "force_regenerate": force_regenerate,
        "batch_size": batch_size,
        "estimate": "Check logs for progress. For 2M emails: ~6 hours, ~$40"
    }


@router.post("/index/create", dependencies=[Depends(verify_api_key)])
async def create_vector_index(
    db: Session = Depends(get_db)
):
    """
    Create HNSW index for fast vector search (admin endpoint).
    
    **Warning:** For 2M emails, this takes ~30 minutes.
    
    **This creates:**
    ```sql
    CREATE INDEX email_embeddings_hnsw_idx 
    ON email_embeddings 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    ```
    
    **Performance:**
    - Without index: 5-10 seconds per query
    - With index: 50-100ms per query (100x faster!)
    
    **Only needs to run once** after initial embedding generation.
    """
    try:
        searcher = VectorSearch(db)
        searcher.create_hnsw_index()
        
        return {
            "status": "success",
            "message": "HNSW index created successfully",
            "note": "This took ~30 minutes for 2M emails"
        }
        
    except Exception as e:
        logger.error(f"Index creation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Index creation failed. Make sure pgvector extension is installed."
        )


@router.get("/index/stats", dependencies=[Depends(verify_api_key)])
async def get_index_stats(db: Session = Depends(get_db)):
    """
    Get vector index statistics.
    
    **Returns:**
    - Total embeddings count
    - Index size on disk
    - Table size
    """
    try:
        searcher = VectorSearch(db)
        stats = searcher.get_index_stats()
        
        return {
            "status": "success",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve index statistics."
        )

