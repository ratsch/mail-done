"""
Debug endpoints for troubleshooting
"""
import os
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from backend.api.auth import verify_api_key
from backend.core.database import get_db
from backend.core.database.models import EmailEmbedding
from backend.core.search.embeddings import EmbeddingGenerator

router = APIRouter(prefix="/api/debug", tags=["debug"])


@router.get("/config", dependencies=[Depends(verify_api_key)])
async def get_config_status():
    """Check configuration status"""
    return {
        "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "database_url_set": bool(os.getenv("DATABASE_URL")),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    }


@router.get("/embeddings/test", dependencies=[Depends(verify_api_key)])
async def test_embedding_generation():
    """Test if embedding generation works"""
    try:
        generator = EmbeddingGenerator()
        test_query = "test query"
        embedding = generator.generate_query_embedding(test_query)
        
        # Check if it's a zero vector (failure)
        is_zero = all(x == 0.0 for x in embedding)
        
        return {
            "status": "success" if not is_zero else "failed",
            "embedding_dimension": len(embedding),
            "is_zero_vector": is_zero,
            "sample_values": embedding[:5].tolist() if hasattr(embedding, 'tolist') else embedding[:5],
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY"))
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY"))
        }


@router.get("/embeddings/count", dependencies=[Depends(verify_api_key)])
async def count_embeddings(db: Session = Depends(get_db)):
    """Count embeddings in database"""
    total = db.query(func.count(EmailEmbedding.id)).scalar() or 0
    
    # Sample some embeddings to verify they're not zero vectors
    sample = db.query(EmailEmbedding).limit(3).all()
    
    sample_info = []
    for emb in sample:
        is_zero = all(x == 0.0 for x in emb.embedding)
        sample_info.append({
            "email_id": str(emb.email_id),
            "dimension": len(emb.embedding),
            "is_zero_vector": is_zero,
            "sample_values": emb.embedding[:5]
        })
    
    return {
        "total_embeddings": total,
        "samples": sample_info
    }

