"""
Unified Search Service

Provides semantic search across both emails and documents using pgvector.
Returns merged and ranked results from both corpora.

Key Features:
- Single query searches both email_embeddings and document_embeddings
- Results merged by similarity score (comparable because same embedding model)
- Type filtering: "email", "document", or "all"
- Date range filtering across both types
"""

import logging
from typing import List, Optional, Tuple, Union, Literal
from uuid import UUID
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.core.database.models import Email
from backend.core.documents.models import Document
from backend.core.search.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class ResultType(str, Enum):
    """Type of search result."""
    EMAIL = "email"
    DOCUMENT = "document"


@dataclass
class UnifiedSearchResult:
    """A single unified search result."""
    result_type: ResultType
    item: Union[Email, Document]
    similarity: float

    # Convenience properties for sorting
    @property
    def date(self) -> Optional[datetime]:
        """Get the relevant date for this result."""
        if self.result_type == ResultType.EMAIL:
            return self.item.date
        else:
            return self.item.document_date


class UnifiedSearchService:
    """
    Unified search across emails and documents.

    Uses the same embedding model for both corpora, so similarity
    scores are directly comparable and can be merged.
    """

    def __init__(
        self,
        db: Session,
        embedding_model: str = "text-embedding-3-large",
        ef_search: int = 40,
    ):
        """
        Initialize unified search service.

        Args:
            db: Database session
            embedding_model: OpenAI embedding model (must match both corpora)
            ef_search: HNSW ef_search parameter
        """
        self.db = db
        self.embedding_generator = EmbeddingGenerator(model=embedding_model)
        self._optimize_index_params(ef_search)

    def _optimize_index_params(self, ef_search: int):
        """Set HNSW search parameters."""
        try:
            self.db.execute(
                text("SET hnsw.ef_search = :ef_search").bindparams(ef_search=ef_search)
            )
        except Exception as e:
            logger.debug(f"Could not set HNSW parameters: {e}")

    async def search(
        self,
        query: str,
        types: Literal["all", "email", "document"] = "all",
        top_k: int = 20,
        similarity_threshold: float = 0.6,
        date_from: Optional[Union[datetime, date]] = None,
        date_to: Optional[Union[datetime, date]] = None,
        # Email-specific filters
        email_category: Optional[str] = None,
        email_sender: Optional[str] = None,
        email_account: Optional[str] = None,
        # Document-specific filters
        document_type: Optional[str] = None,
        document_mime_type: Optional[str] = None,
        document_min_quality: Optional[float] = None,
    ) -> List[UnifiedSearchResult]:
        """
        Search across emails and documents.

        Args:
            query: Natural language search query
            types: What to search - "all", "email", or "document"
            top_k: Maximum total results
            similarity_threshold: Minimum similarity score
            date_from: Start of date range
            date_to: End of date range
            email_category: Filter emails by category
            email_sender: Filter emails by sender
            email_account: Filter emails by account
            document_type: Filter documents by type
            document_mime_type: Filter documents by MIME type
            document_min_quality: Minimum extraction quality

        Returns:
            List of UnifiedSearchResult sorted by similarity (descending)
        """
        # Generate query embedding
        query_vector = self.embedding_generator.generate_query_embedding(query)

        if all(x == 0.0 for x in query_vector):
            logger.error(f"Query embedding generation failed for: {query}")
            return []

        results: List[UnifiedSearchResult] = []

        # Search emails if requested
        if types in ("all", "email"):
            email_results = await self._search_emails(
                query_vector=query_vector,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                date_from=date_from,
                date_to=date_to,
                category=email_category,
                sender=email_sender,
                account_id=email_account,
            )
            results.extend(email_results)

        # Search documents if requested
        if types in ("all", "document"):
            doc_results = await self._search_documents(
                query_vector=query_vector,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                date_from=date_from,
                date_to=date_to,
                document_type=document_type,
                mime_type=document_mime_type,
                min_quality=document_min_quality,
            )
            results.extend(doc_results)

        # Sort by similarity and limit
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    async def _search_emails(
        self,
        query_vector: List[float],
        top_k: int,
        similarity_threshold: float,
        date_from: Optional[Union[datetime, date]] = None,
        date_to: Optional[Union[datetime, date]] = None,
        category: Optional[str] = None,
        sender: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> List[UnifiedSearchResult]:
        """Search email embeddings."""
        vector_str = '[' + ','.join(str(float(x)) for x in query_vector) + ']'

        params = {
            'query_vector': vector_str,
            'threshold': similarity_threshold,
            'limit': top_k,
            'candidate_limit': min(top_k * 3, 500),
        }

        filter_conditions = []
        join_metadata = False

        if category:
            filter_conditions.append("m.ai_category = :category")
            params['category'] = category
            join_metadata = True

        if date_from:
            filter_conditions.append("e.date >= :date_from")
            params['date_from'] = date_from

        if date_to:
            filter_conditions.append("e.date <= :date_to")
            params['date_to'] = date_to

        if sender:
            filter_conditions.append("e.from_address ILIKE :sender_pattern")
            params['sender_pattern'] = f"%{sender}%"

        if account_id:
            filter_conditions.append("e.account_id = :account_id")
            params['account_id'] = account_id

        if filter_conditions:
            metadata_join = "JOIN email_metadata m ON e.id = m.email_id" if join_metadata else ""
            where_clause = " AND ".join(filter_conditions)

            base_sql = f"""
                WITH filtered AS (
                    SELECT emb.email_id, emb.embedding
                    FROM email_embeddings emb
                    JOIN emails e ON emb.email_id = e.id
                    {metadata_join}
                    WHERE {where_clause}
                ),
                scored AS (
                    SELECT
                        email_id,
                        1 - (embedding <=> cast(:query_vector as vector)) as similarity
                    FROM filtered
                    ORDER BY embedding <=> cast(:query_vector as vector)
                    LIMIT :candidate_limit
                )
                SELECT email_id, similarity
                FROM scored
                WHERE similarity >= :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """
        else:
            base_sql = """
                WITH scored AS (
                    SELECT
                        email_id,
                        1 - (embedding <=> cast(:query_vector as vector)) as similarity
                    FROM email_embeddings
                    ORDER BY embedding <=> cast(:query_vector as vector)
                    LIMIT :candidate_limit
                )
                SELECT email_id, similarity
                FROM scored
                WHERE similarity >= :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """

        try:
            result = self.db.execute(text(base_sql), params)
            rows = result.fetchall()

            if not rows:
                return []

            # Fetch emails
            email_ids = [row.email_id for row in rows]
            emails_map = {
                email.id: email
                for email in self.db.query(Email).filter(Email.id.in_(email_ids)).all()
            }

            results = []
            for row in rows:
                email = emails_map.get(row.email_id)
                if email:
                    similarity = float(row.similarity)
                    if similarity == similarity and 0 <= similarity <= 1:  # Filter NaN
                        results.append(UnifiedSearchResult(
                            result_type=ResultType.EMAIL,
                            item=email,
                            similarity=similarity,
                        ))

            return results

        except Exception as e:
            logger.error(f"Email search error: {e}", exc_info=True)
            self.db.rollback()
            return []

    async def _search_documents(
        self,
        query_vector: List[float],
        top_k: int,
        similarity_threshold: float,
        date_from: Optional[Union[datetime, date]] = None,
        date_to: Optional[Union[datetime, date]] = None,
        document_type: Optional[str] = None,
        mime_type: Optional[str] = None,
        min_quality: Optional[float] = None,
    ) -> List[UnifiedSearchResult]:
        """Search document embeddings."""
        vector_str = '[' + ','.join(str(float(x)) for x in query_vector) + ']'

        params = {
            'query_vector': vector_str,
            'threshold': similarity_threshold,
            'limit': top_k,
            'candidate_limit': min(top_k * 3, 500),
        }

        filter_conditions = ["d.is_deleted = false"]

        if document_type:
            filter_conditions.append("d.document_type = :document_type")
            params['document_type'] = document_type

        if mime_type:
            filter_conditions.append("d.mime_type = :mime_type")
            params['mime_type'] = mime_type

        if min_quality is not None:
            filter_conditions.append("d.extraction_quality >= :min_quality")
            params['min_quality'] = min_quality

        if date_from:
            filter_conditions.append("d.document_date >= :date_from")
            params['date_from'] = date_from

        if date_to:
            filter_conditions.append("d.document_date <= :date_to")
            params['date_to'] = date_to

        where_clause = " AND ".join(filter_conditions)

        base_sql = f"""
            WITH filtered AS (
                SELECT emb.document_id, emb.embedding
                FROM document_embeddings emb
                JOIN documents d ON emb.document_id = d.id
                WHERE {where_clause}
            ),
            scored AS (
                SELECT
                    document_id,
                    1 - (embedding <=> cast(:query_vector as vector)) as similarity
                FROM filtered
                ORDER BY embedding <=> cast(:query_vector as vector)
                LIMIT :candidate_limit
            )
            SELECT document_id, similarity
            FROM scored
            WHERE similarity >= :threshold
            ORDER BY similarity DESC
            LIMIT :limit
        """

        try:
            result = self.db.execute(text(base_sql), params)
            rows = result.fetchall()

            if not rows:
                return []

            # Fetch documents
            doc_ids = [row.document_id for row in rows]
            docs_map = {
                doc.id: doc
                for doc in self.db.query(Document).filter(Document.id.in_(doc_ids)).all()
            }

            results = []
            for row in rows:
                doc = docs_map.get(row.document_id)
                if doc:
                    similarity = float(row.similarity)
                    if similarity == similarity and 0 <= similarity <= 1:  # Filter NaN
                        results.append(UnifiedSearchResult(
                            result_type=ResultType.DOCUMENT,
                            item=doc,
                            similarity=similarity,
                        ))

            return results

        except Exception as e:
            logger.error(f"Document search error: {e}", exc_info=True)
            self.db.rollback()
            return []

    async def find_related(
        self,
        email_id: Optional[UUID] = None,
        document_id: Optional[UUID] = None,
        types: Literal["all", "email", "document"] = "all",
        top_k: int = 10,
        similarity_threshold: float = 0.6,
    ) -> List[UnifiedSearchResult]:
        """
        Find emails and documents similar to a reference item.

        Args:
            email_id: Reference email UUID (provide one of email_id or document_id)
            document_id: Reference document UUID
            types: What to search - "all", "email", or "document"
            top_k: Maximum results
            similarity_threshold: Minimum similarity

        Returns:
            List of similar items (excluding the reference)
        """
        from backend.core.database.models import EmailEmbedding
        from backend.core.documents.models import DocumentEmbedding

        # Get reference embedding
        if email_id:
            emb = self.db.query(EmailEmbedding).filter(
                EmailEmbedding.email_id == email_id
            ).first()
            exclude_email = email_id
            exclude_doc = None
        elif document_id:
            emb = self.db.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            ).first()
            exclude_email = None
            exclude_doc = document_id
        else:
            raise ValueError("Must provide either email_id or document_id")

        if not emb:
            logger.warning(f"No embedding found for reference")
            return []

        query_vector = emb.embedding
        results: List[UnifiedSearchResult] = []

        # Search emails
        if types in ("all", "email"):
            email_results = await self._search_emails(
                query_vector=query_vector,
                top_k=top_k + 1,  # Extra for exclusion
                similarity_threshold=similarity_threshold,
            )
            # Exclude reference
            results.extend([r for r in email_results if r.item.id != exclude_email])

        # Search documents
        if types in ("all", "document"):
            doc_results = await self._search_documents(
                query_vector=query_vector,
                top_k=top_k + 1,
                similarity_threshold=similarity_threshold,
            )
            results.extend([r for r in doc_results if r.item.id != exclude_doc])

        # Sort and limit
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]
