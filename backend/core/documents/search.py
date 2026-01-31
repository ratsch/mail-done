"""
Document Search Service

Provides semantic search over indexed documents using pgvector.
Follows the same patterns as email vector search for consistency.

Features:
- Semantic search with text-embedding-3-large (3072 dims)
- Date range filtering (document_date)
- Filter by document_type, min_quality, mime_type
- Find similar documents
"""

import logging
from typing import List, Optional, Tuple
from uuid import UUID
from datetime import date

from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.core.documents.models import Document, DocumentEmbedding
from backend.core.search.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class DocumentSearchService:
    """
    Semantic search over document embeddings using pgvector.

    Uses the same embedding model as email search (text-embedding-3-large)
    to enable cross-domain searches between emails and documents.
    """

    def __init__(
        self,
        db: Session,
        embedding_model: str = "text-embedding-3-large",
        ef_search: int = 40,
    ):
        """
        Initialize document search service.

        Args:
            db: Database session
            embedding_model: OpenAI embedding model (must match document embeddings)
            ef_search: HNSW ef_search parameter (10-200, higher = more accurate)
        """
        self.db = db
        self.embedding_generator = EmbeddingGenerator(model=embedding_model)
        self._optimize_index_params(ef_search)

    def _optimize_index_params(self, ef_search: int):
        """Set HNSW search parameters for the session."""
        try:
            self.db.execute(
                text("SET hnsw.ef_search = :ef_search").bindparams(ef_search=ef_search)
            )
        except Exception as e:
            logger.debug(f"Could not set HNSW parameters: {e}")

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.6,
        document_type: Optional[str] = None,
        mime_type: Optional[str] = None,
        min_quality: Optional[float] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        ai_category: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search documents by semantic similarity.

        Args:
            query: Natural language search query
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            document_type: Filter by document type (invoice, contract, etc.)
            mime_type: Filter by MIME type
            min_quality: Minimum extraction quality (0.0-1.0)
            date_from: Include documents dated on or after this date
            date_to: Include documents dated on or before this date
            ai_category: Filter by AI-assigned category

        Returns:
            List of (Document, similarity_score) tuples sorted by similarity
        """
        # Generate query embedding
        query_vector = self.embedding_generator.generate_query_embedding(query)

        # Check for failed embedding generation
        if all(x == 0.0 for x in query_vector):
            logger.error(f"Query embedding generation failed for: {query}")
            return []

        return await self._search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_type=document_type,
            mime_type=mime_type,
            min_quality=min_quality,
            date_from=date_from,
            date_to=date_to,
            ai_category=ai_category,
        )

    async def find_similar(
        self,
        document_id: UUID,
        top_k: int = 10,
        similarity_threshold: float = 0.6,
        same_type_only: bool = False,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Find documents similar to a reference document.

        Args:
            document_id: Reference document UUID
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            same_type_only: Only return documents of the same type
            date_from: Include documents dated on or after this date
            date_to: Include documents dated on or before this date

        Returns:
            List of (Document, similarity_score) tuples sorted by similarity
        """
        # Get embedding for reference document
        emb = self.db.query(DocumentEmbedding).filter(
            DocumentEmbedding.document_id == document_id
        ).first()

        if not emb:
            logger.warning(f"No embedding found for document {document_id}")
            return []

        # Get reference document for type filtering
        document_type = None
        if same_type_only:
            ref_doc = self.db.get(Document, document_id)
            if ref_doc:
                document_type = ref_doc.document_type

        # Search, requesting extra to filter out reference
        results = await self._search_by_vector(
            query_vector=emb.embedding,
            top_k=top_k + 1,  # Extra in case reference is in results
            similarity_threshold=similarity_threshold,
            document_type=document_type,
            date_from=date_from,
            date_to=date_to,
        )

        # Filter out the reference document
        filtered = [(doc, score) for doc, score in results if doc.id != document_id]
        return filtered[:top_k]

    async def _search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        similarity_threshold: float = 0.6,
        document_type: Optional[str] = None,
        mime_type: Optional[str] = None,
        min_quality: Optional[float] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        ai_category: Optional[str] = None,
        exclude_document_id: Optional[UUID] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Internal method to search by vector with filters.

        Uses pgvector's cosine distance operator (<=>).
        Strategy: CTE filters, then vector search, then threshold filter.
        """
        # Convert vector to pgvector format
        vector_str = '[' + ','.join(str(float(x)) for x in query_vector) + ']'

        # Build parameters
        params = {
            'query_vector': vector_str,
            'threshold': similarity_threshold,
            'limit': top_k,
        }

        # Build filter conditions
        filter_conditions = []
        join_documents = False

        if document_type:
            filter_conditions.append("d.document_type = :document_type")
            params['document_type'] = document_type
            join_documents = True

        if mime_type:
            filter_conditions.append("d.mime_type = :mime_type")
            params['mime_type'] = mime_type
            join_documents = True

        if min_quality is not None:
            filter_conditions.append("d.extraction_quality >= :min_quality")
            params['min_quality'] = min_quality
            join_documents = True

        if date_from:
            filter_conditions.append("d.document_date >= :date_from")
            params['date_from'] = date_from
            join_documents = True

        if date_to:
            filter_conditions.append("d.document_date <= :date_to")
            params['date_to'] = date_to
            join_documents = True

        if ai_category:
            filter_conditions.append("d.ai_category = :ai_category")
            params['ai_category'] = ai_category
            join_documents = True

        if exclude_document_id:
            filter_conditions.append("emb.document_id != :exclude_id")
            params['exclude_id'] = str(exclude_document_id)

        # Always filter out deleted documents
        filter_conditions.append("d.is_deleted = false")
        join_documents = True

        # Calculate candidate limit (get more than needed for threshold filtering)
        candidate_multiplier = 3 if similarity_threshold > 0.5 else 5
        candidate_limit = min(top_k * candidate_multiplier, 500)
        params['candidate_limit'] = candidate_limit

        has_filters = len(filter_conditions) > 1  # More than just is_deleted

        logger.debug(
            f"Document search: top_k={top_k}, threshold={similarity_threshold}, "
            f"has_filters={has_filters}, filters={len(filter_conditions)}"
        )

        if has_filters or join_documents:
            # Use CTE to filter first, then vector search
            where_clause = " AND ".join(filter_conditions)

            base_sql = f"""
                WITH filtered_docs AS (
                    SELECT emb.document_id, emb.embedding
                    FROM document_embeddings emb
                    JOIN documents d ON emb.document_id = d.id
                    WHERE {where_clause}
                ),
                scored AS (
                    SELECT
                        document_id,
                        1 - (embedding <=> cast(:query_vector as vector)) as similarity
                    FROM filtered_docs
                    ORDER BY embedding <=> cast(:query_vector as vector)
                    LIMIT :candidate_limit
                )
                SELECT document_id, similarity
                FROM scored
                WHERE similarity >= :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """
        else:
            # Pure vector search (optimal for DiskANN index)
            base_sql = """
                WITH scored AS (
                    SELECT
                        document_id,
                        1 - (embedding <=> cast(:query_vector as vector)) as similarity
                    FROM document_embeddings
                    ORDER BY embedding <=> cast(:query_vector as vector)
                    LIMIT :candidate_limit
                )
                SELECT document_id, similarity
                FROM scored
                WHERE similarity >= :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """

        sql = text(base_sql)

        try:
            result = self.db.execute(sql, params)
            rows = result.fetchall()

            logger.info(f"Document search returned {len(rows)} rows")

            if not rows:
                return []

            # Fetch documents in single query (avoids N+1)
            document_ids = [row.document_id for row in rows]
            docs_map = {}
            for doc in self.db.query(Document).filter(Document.id.in_(document_ids)).all():
                docs_map[doc.id] = doc

            # Build results maintaining order
            results = []
            for row in rows:
                doc = docs_map.get(row.document_id)
                if doc:
                    similarity = float(row.similarity)

                    # Filter NaN/Inf
                    if similarity != similarity:  # NaN check
                        logger.warning(f"Skipping document {doc.id} with NaN similarity")
                        continue
                    if similarity < 0 or similarity > 1:
                        logger.warning(f"Skipping document {doc.id} with invalid similarity: {similarity}")
                        continue

                    results.append((doc, similarity))

            logger.info(f"Returning {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Error in document search: {e}", exc_info=True)
            self.db.rollback()
            raise RuntimeError(f"Document search failed - check pgvector configuration: {e}") from e

    async def search_by_content_type(
        self,
        query: str,
        content_types: List[str],
        top_k: int = 20,
        similarity_threshold: float = 0.5,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search documents filtered by content type (MIME type prefixes).

        Convenience method for common searches like "all PDFs" or "all images".

        Args:
            query: Search query
            content_types: List of MIME type prefixes (e.g., ['application/pdf', 'image/'])
            top_k: Maximum results
            similarity_threshold: Minimum similarity
            date_from: Date range start
            date_to: Date range end

        Returns:
            List of (Document, similarity) tuples
        """
        # Generate query embedding
        query_vector = self.embedding_generator.generate_query_embedding(query)

        if all(x == 0.0 for x in query_vector):
            logger.error(f"Query embedding generation failed for: {query}")
            return []

        # Build MIME type filter using LIKE patterns
        vector_str = '[' + ','.join(str(float(x)) for x in query_vector) + ']'

        # Build MIME type conditions
        mime_conditions = []
        for i, ct in enumerate(content_types):
            if ct.endswith('/'):
                mime_conditions.append(f"d.mime_type LIKE :mime_{i}")
            else:
                mime_conditions.append(f"d.mime_type = :mime_{i}")

        mime_where = " OR ".join(mime_conditions)

        params = {
            'query_vector': vector_str,
            'threshold': similarity_threshold,
            'limit': top_k,
            'candidate_limit': min(top_k * 3, 500),
        }

        # Add MIME type params
        for i, ct in enumerate(content_types):
            if ct.endswith('/'):
                params[f'mime_{i}'] = f"{ct}%"
            else:
                params[f'mime_{i}'] = ct

        date_conditions = []
        if date_from:
            date_conditions.append("d.document_date >= :date_from")
            params['date_from'] = date_from
        if date_to:
            date_conditions.append("d.document_date <= :date_to")
            params['date_to'] = date_to

        date_where = " AND ".join(date_conditions) if date_conditions else "TRUE"

        base_sql = f"""
            WITH filtered_docs AS (
                SELECT emb.document_id, emb.embedding
                FROM document_embeddings emb
                JOIN documents d ON emb.document_id = d.id
                WHERE ({mime_where})
                AND d.is_deleted = false
                AND {date_where}
            ),
            scored AS (
                SELECT
                    document_id,
                    1 - (embedding <=> cast(:query_vector as vector)) as similarity
                FROM filtered_docs
                ORDER BY embedding <=> cast(:query_vector as vector)
                LIMIT :candidate_limit
            )
            SELECT document_id, similarity
            FROM scored
            WHERE similarity >= :threshold
            ORDER BY similarity DESC
            LIMIT :limit
        """

        sql = text(base_sql)

        try:
            result = self.db.execute(sql, params)
            rows = result.fetchall()

            if not rows:
                return []

            # Fetch documents
            document_ids = [row.document_id for row in rows]
            docs_map = {doc.id: doc for doc in
                       self.db.query(Document).filter(Document.id.in_(document_ids)).all()}

            return [
                (docs_map[row.document_id], float(row.similarity))
                for row in rows
                if row.document_id in docs_map
            ]

        except Exception as e:
            logger.error(f"Error in content type search: {e}", exc_info=True)
            self.db.rollback()
            raise

    def get_index_stats(self) -> dict:
        """Get statistics about the document vector index."""
        try:
            # Check for index
            result = self.db.execute(text("""
                SELECT indexname, pg_size_pretty(pg_total_relation_size(indexrelid))
                FROM pg_indexes
                JOIN pg_class ON indexrelid = pg_class.oid
                WHERE tablename = 'document_embeddings'
                AND indexname LIKE '%vector%' OR indexname LIKE '%diskann%' OR indexname LIKE '%hnsw%'
            """)).fetchall()

            # Get embedding count
            count = self.db.query(DocumentEmbedding).count()

            return {
                'total_embeddings': count,
                'indexes': [{'name': r[0], 'size': r[1]} for r in result] if result else [],
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}", exc_info=True)
            return {'error': str(e)}
