"""
Vector Search using pgvector

Provides semantic similarity search over email embeddings.
Optimized for 2M+ emails using HNSW indexing.
"""
import logging
from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import text, desc

from backend.core.database.models import Email, EmailMetadata, EmailEmbedding
from backend.core.search.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class VectorSearch:
    """Semantic search using pgvector and HNSW indexing."""
    
    def __init__(self, db: Session, embedding_model: str = "text-embedding-3-large", ef_search: int = 40):
        """
        Initialize vector search.
        
        Args:
            db: Database session
            embedding_model: OpenAI embedding model to use (default: text-embedding-3-large)
            ef_search: HNSW ef_search parameter for speed/accuracy tradeoff (default: 40)
                      Higher = more accurate but slower. Range: 10-200.
        """
        self.db = db
        self.embedding_generator = EmbeddingGenerator(model=embedding_model)
        # Optimize HNSW search parameters for better performance
        self.optimize_index_params(ef_search=ef_search)
    
    async def search_similar(
        self,
        query: str = None,
        email_id: UUID = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        category: Optional[object] = None,  # str or List[str]
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        account_id: Optional[str] = None,
        sender: Optional[str] = None
    ) -> List[Tuple[Email, float]]:
        """
        Find similar emails using vector similarity.
        
        Args:
            query: Text query to search for (OR)
            email_id: Find emails similar to this one (OR)
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            category: Filter by email category (str or List[str])
            date_from: Filter emails after this date
            date_to: Filter emails before this date
            account_id: Filter by account (parameterized, safe)
            sender: Filter by sender email/domain (parameterized, safe)
        
        Returns:
            List of (Email, similarity_score) tuples, sorted by similarity
        """
        # Get query vector
        if query:
            query_vector = self.embedding_generator.generate_query_embedding(query)
            # Check if embedding generation failed (returns zero vector)
            if all(x == 0.0 for x in query_vector):
                logger.error(f"Query embedding generation failed for: {query}")
                return []  # Return empty results instead of NaN errors
        elif email_id:
            # Get embedding for the reference email
            emb = self.db.query(EmailEmbedding).filter(
                EmailEmbedding.email_id == email_id
            ).first()
            if not emb:
                return []
            query_vector = emb.embedding
        else:
            raise ValueError("Must provide either query or email_id")
        
        # Build SQL query with pgvector
        # Using cosine distance: 1 - (embedding <=> query_vector)
        # <=> is pgvector's cosine distance operator
        
        # Convert query vector to pgvector format (safely)
        # pgvector expects format: '[1.0,2.0,3.0]'::vector
        # Use parameterized query to prevent SQL injection
        vector_str = '[' + ','.join(str(float(x)) for x in query_vector) + ']'
        
        # Build optimized SQL query using pgvector index
        # Strategy: Use ORDER BY with distance operator to leverage HNSW/DiskANN index
        # This allows approximate nearest neighbor search instead of full table scan
        
        # Determine if we have pre-filters (date, account, category, sender)
        has_prefilters = bool(date_from or date_to or account_id or sender or category)
        
        logger.debug(f"Vector search: top_k={top_k}, threshold={similarity_threshold}, "
                    f"has_prefilters={has_prefilters}")
        
        params = {
            'query_vector': vector_str,
            'threshold': similarity_threshold,
            'limit': top_k
        }
        
        # Build filter conditions using PARAMETERIZED queries (safe from SQL injection)
        filter_conditions = []
        join_metadata = False
        
        if category:
            if isinstance(category, list):
                filter_conditions.append("m.ai_category IN :categories")
                params['categories'] = tuple(category)
            else:
                filter_conditions.append("m.ai_category = :category")
                params['category'] = category
            join_metadata = True
        
        if date_from:
            filter_conditions.append("e.date >= :date_from")
            params['date_from'] = date_from
        
        if date_to:
            filter_conditions.append("e.date <= :date_to")
            params['date_to'] = date_to
        
        # Account and sender filters (parameterized - safe from SQL injection)
        if account_id:
            filter_conditions.append("e.account_id = :account_id")
            params['account_id'] = account_id
        
        if sender:
            # Use parameterized ILIKE for safe pattern matching
            filter_conditions.append("e.from_address ILIKE :sender_pattern")
            params['sender_pattern'] = f"%{sender}%"
        
        if has_prefilters:
            # **OPTIMIZED APPROACH: Filter first, then vector search, then threshold filter**
            # Strategy:
            # 1. CTE filters by date/account/category (uses standard indexes)
            # 2. Vector distance computed once in SELECT
            # 3. Threshold applied in outer query (after index scan)
            # 4. ORDER BY uses the raw distance operator for index efficiency
            
            metadata_join = "JOIN email_metadata m ON e.id = m.email_id" if join_metadata else ""
            where_clause = " AND ".join(filter_conditions)
            
            # Request more candidates than needed to account for threshold filtering
            candidate_multiplier = 3 if similarity_threshold > 0.5 else 5
            candidate_limit = min(top_k * candidate_multiplier, 500)
            params['candidate_limit'] = candidate_limit
            
            base_sql = f"""
                WITH filtered_emails AS (
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
                    FROM filtered_emails
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
            # No filters - pure vector search (DiskANN index works optimally)
            # Get more candidates, then filter by threshold in outer query
            candidate_multiplier = 3 if similarity_threshold > 0.5 else 5
            candidate_limit = min(top_k * candidate_multiplier, 500)
            params['candidate_limit'] = candidate_limit
            
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
        
        sql = text(base_sql)
        
        # Execute query
        try:
            # Log query parameters for debugging (without exposing full vector)
            logger.debug(
                f"Vector search: top_k={top_k}, threshold={similarity_threshold}, "
                f"has_prefilters={has_prefilters}, filters={len(filter_conditions)}"
            )
            
            result = self.db.execute(sql, params)
            rows = result.fetchall()
            
            logger.info(f"Vector search returned {len(rows)} rows from database")
            
            if not rows:
                return []
            
            # **OPTIMIZATION: Fetch all emails in a single query (avoids N+1 problem)**
            email_ids = [row.email_id for row in rows]
            
            # Fetch emails with metadata eagerly
            emails_map = {}
            emails_query = self.db.query(Email).filter(Email.id.in_(email_ids))
            
            for email in emails_query.all():
                emails_map[email.id] = email
                
            # Convert to Email objects with similarity scores (maintaining order)
            results = []
            for row in rows:
                email = emails_map.get(row.email_id)
                if email:
                    similarity = float(row.similarity)
                    # Filter out NaN/Inf values (can happen with zero vectors)
                    if similarity != similarity:  # NaN check (NaN != NaN)
                        logger.warning(f"Skipping email {email.id} with NaN similarity")
                        continue
                    if similarity < 0 or similarity > 1:  # Sanity check
                        logger.warning(f"Skipping email {email.id} with out-of-range similarity: {similarity}")
                        continue
                    results.append((email, similarity))
                    logger.debug(f"Found email: {email.subject[:50]} with similarity {similarity:.4f}")
            
            logger.info(f"Returning {len(results)} email objects")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            # Rollback failed transaction to prevent session corruption
            self.db.rollback()
            
            # For production with pgvector, raise error instead of slow fallback
            # Fallback loads ALL embeddings into memory - not suitable for >10K emails
            raise RuntimeError(f"Vector search failed - check pgvector configuration: {e}") from e
    
    async def _fallback_search(
        self,
        query_vector: List[float],
        top_k: int,
        threshold: float
    ) -> List[Tuple[Email, float]]:
        """
        Fallback search using Python-based similarity calculation.
        Slower but works without pgvector support.
        
        WARNING: For large datasets (>100K emails), this will be very slow
        and memory intensive. Use pgvector for production at scale.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check embedding count first
        total_embeddings = self.db.query(EmailEmbedding).count()
        
        if total_embeddings > 100000:
            # Too many for fallback search - would cause OOM
            logger.error(
                f"Fallback search not available for {total_embeddings:,} embeddings. "
                f"Install pgvector for production use."
            )
            return []
        
        logger.warning(
            f"Using fallback search for {total_embeddings:,} embeddings. "
            f"This is slow. Install pgvector for production."
        )
        
        # Process in chunks to avoid OOM
        CHUNK_SIZE = 10000
        all_similarities = []
        
        for offset in range(0, total_embeddings, CHUNK_SIZE):
            embeddings_chunk = self.db.query(EmailEmbedding).offset(offset).limit(CHUNK_SIZE).all()
            
            # Calculate similarities for chunk
            for emb in embeddings_chunk:
                sim = self.embedding_generator.cosine_similarity(
                    query_vector,
                    emb.embedding
                )
                if sim >= threshold:
                    all_similarities.append((emb.email_id, sim))
        
        # Sort by similarity
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k emails
        results = []
        for email_id, sim in all_similarities[:top_k]:
            email = self.db.query(Email).filter(Email.id == email_id).first()
            if email:
                results.append((email, sim))
        
        return results
    
    async def search_by_topic(
        self,
        topic: str,
        categories: Optional[List[str]] = None,
        top_k: int = 20,
        similarity_threshold: float = 0.6,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Tuple[Email, float]]:
        """
        Search emails by research topic or concept.
        
        Args:
            topic: Topic to search for (e.g., "machine learning genomics")
            categories: Filter by categories (e.g., ["application-phd", "invitation-speaking"])
            top_k: Number of results
            similarity_threshold: Minimum similarity
            date_from: Filter emails after this date
            date_to: Filter emails before this date
        
        Returns:
            List of (Email, similarity_score) tuples
        """
        # Enhance query with topic context
        enhanced_query = f"Research topic: {topic}. Related work, papers, and discussions about {topic}."
        
        # Search with category filter and date range (efficient single query)
        return await self.search_similar(
            query=enhanced_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            category=categories,  # Pass list directly
            date_from=date_from,
            date_to=date_to
        )
    
    async def find_similar_emails(
        self,
        email_id: UUID,
        top_k: int = 10,
        same_category_only: bool = False,
        exclude_same_sender: bool = False
    ) -> List[Tuple[Email, float]]:
        """
        Find emails similar to a given email.
        
        Args:
            email_id: Reference email
            top_k: Number of similar emails to find
            same_category_only: Only return emails in same category
            exclude_same_sender: Exclude emails from same sender
        
        Returns:
            List of (Email, similarity_score) tuples
        """
        # Get reference email
        ref_email = self.db.query(Email).filter(Email.id == email_id).first()
        if not ref_email:
            return []
        
        # Get category if filtering
        category = None
        if same_category_only and ref_email.email_metadata:
            category = ref_email.email_metadata.ai_category
        
        # Search
        results = await self.search_similar(
            email_id=email_id,
            top_k=top_k + 10,  # Get extra in case we filter some out
            category=category
        )
        
        # Filter out reference email and optionally same sender
        filtered = []
        for email, score in results:
            if email.id == email_id:
                continue  # Skip reference email itself
            
            if exclude_same_sender and email.from_address == ref_email.from_address:
                continue  # Skip same sender
            
            filtered.append((email, score))
            
            if len(filtered) >= top_k:
                break
        
        return filtered
    
    def create_hnsw_index(self):
        """
        Create HNSW index for fast vector similarity search.
        
        DEPRECATED: Use create_diskann_index() for better performance with pgvectorscale.
        
        HNSW (Hierarchical Navigable Small World) is a good index type
        for approximate nearest neighbor search, but DiskANN is faster at scale.
        
        This should be run after initial embedding generation.
        For 2M vectors, indexing takes ~30 minutes.
        """
        try:
            # Drop existing index if any
            self.db.execute(text("""
                DROP INDEX IF EXISTS email_embeddings_hnsw_idx
            """))
            
            # Create HNSW index
            # m=16: number of connections per layer (default, good balance)
            # ef_construction=64: higher = better quality but slower build
            logger.info("Creating HNSW index (this may take 20-30 minutes for 2M vectors)...")
            
            self.db.execute(text("""
                CREATE INDEX email_embeddings_hnsw_idx 
                ON email_embeddings 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            
            self.db.commit()
            logger.info("HNSW index created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating HNSW index: {e}", exc_info=True)
            logger.error("Note: Make sure pgvector extension is installed in PostgreSQL")
            self.db.rollback()
    
    def create_diskann_index(self):
        """
        Create DiskANN index for fast vector similarity search (pgvectorscale).
        
        DiskANN (StreamingDiskANN) is pgvectorscale's optimized index:
        - Faster than HNSW at scale (2M+ vectors)
        - Lower memory usage (disk-based, not RAM-based)
        - Better for production deployments
        
        This should be run after initial embedding generation.
        For 2M vectors, indexing takes ~15-20 minutes.
        """
        try:
            # Check if vectorscale extension is available
            check_result = self.db.execute(text("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_available_extensions 
                    WHERE name = 'vectorscale' AND installed_version IS NOT NULL
                )
            """))
            has_vectorscale = check_result.scalar()
            
            if not has_vectorscale:
                logger.warning("pgvectorscale not available - falling back to HNSW index")
                return self.create_hnsw_index()
            
            # Drop existing DiskANN index if any
            self.db.execute(text("""
                DROP INDEX IF EXISTS email_embeddings_diskann_idx
            """))
            
            logger.info("Creating DiskANN index (this may take 15-20 minutes for 2M vectors)...")
            
            # Create DiskANN index
            self.db.execute(text("""
                CREATE INDEX email_embeddings_diskann_idx 
                ON email_embeddings 
                USING diskann (embedding)
            """))
            
            self.db.commit()
            logger.info("DiskANN index created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating DiskANN index: {e}", exc_info=True)
            self.db.rollback()
            raise
    
    def optimize_index_params(self, ef_search: int = 40):
        """
        Optimize HNSW index search parameters.
        
        Args:
            ef_search: Controls accuracy vs speed tradeoff
                - 10: Fast but less accurate
                - 40: Good balance (default)
                - 100: More accurate but slower
        
        For 2M vectors:
        - ef_search=40: ~50ms per query, ~95% recall
        - ef_search=100: ~150ms per query, ~99% recall
        """
        try:
            # Set at session level (use parameterized query to prevent SQL injection)
            self.db.execute(text("SET hnsw.ef_search = :ef_search").bindparams(ef_search=ef_search))
            logger.info(f"HNSW ef_search set to {ef_search}")
            
        except Exception as e:
            logger.error(f"Error setting HNSW parameters: {e}", exc_info=True)
    
    def get_index_stats(self) -> dict:
        """Get statistics about the vector index."""
        try:
            # Get index size
            result = self.db.execute(text("""
                SELECT 
                    pg_size_pretty(pg_total_relation_size('email_embeddings_hnsw_idx')) as index_size,
                    pg_size_pretty(pg_total_relation_size('email_embeddings')) as table_size
            """)).fetchone()
            
            # Get row count
            count = self.db.query(EmailEmbedding).count()
            
            return {
                'total_embeddings': count,
                'index_size': result.index_size if result else 'N/A',
                'table_size': result.table_size if result else 'N/A'
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}", exc_info=True)
            return {}

