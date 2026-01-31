"""
Advanced Search module for Phase 3

Provides semantic and hybrid search capabilities using vector embeddings.
"""
from backend.core.search.embeddings import EmbeddingGenerator
from backend.core.search.vector_search import VectorSearch
from backend.core.search.hybrid_search import HybridSearch
from backend.core.search.unified_search import UnifiedSearchService, UnifiedSearchResult, ResultType

__all__ = [
    'EmbeddingGenerator',
    'VectorSearch',
    'HybridSearch',
    'UnifiedSearchService',
    'UnifiedSearchResult',
    'ResultType',
]

