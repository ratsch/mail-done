"""
Hybrid Search

Combines keyword/full-text search with vector semantic search for best results.
"""
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func

from backend.core.database.models import Email, EmailMetadata
from backend.core.search.vector_search import VectorSearch


class HybridSearch:
    """
    Hybrid search combining keyword matching and semantic similarity.
    
    Strategy:
    1. Keyword search finds exact/partial matches (high precision)
    2. Vector search finds semantically similar (high recall)
    3. Merge and re-rank results using weighted scoring
    """
    
    def __init__(self, db: Session):
        """
        Initialize hybrid search.
        
        Args:
            db: Database session
        """
        self.db = db
        self.vector_search = VectorSearch(db)
    
    async def search(
        self,
        query: str,
        mode: str = "hybrid",  # keyword/semantic/hybrid
        top_k: int = 20,
        category: Optional[str] = None,
        vip_only: bool = False,
        needs_reply: Optional[bool] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
        similarity_threshold: float = 0.6,
        sender: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> List[Tuple[Email, float, Dict]]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            mode: Search mode (keyword/semantic/hybrid)
            top_k: Number of results
            category: Filter by category
            vip_only: Only VIP emails
            needs_reply: Filter by reply status
            date_from: Start date
            date_to: End date
            keyword_weight: Weight for keyword score (0-1)
            semantic_weight: Weight for semantic score (0-1)
            similarity_threshold: Minimum similarity for semantic search (0-1, default: 0.6)
            sender: Filter by sender email/domain (applied at DB level for speed)
            account_id: Filter by account_id (applied at DB level for speed)
        
        Returns:
            List of (Email, combined_score, score_breakdown) tuples
        """
        if mode == "keyword":
            return await self._keyword_search_only(
                query, top_k, category, vip_only, needs_reply, date_from, date_to, sender, account_id
            )
        elif mode == "semantic":
            return await self._semantic_search_only(
                query, top_k, category, vip_only, needs_reply, date_from, date_to, similarity_threshold, sender, account_id
            )
        else:  # hybrid
            return await self._hybrid_search(
                query, top_k, category, vip_only, needs_reply,
                date_from, date_to, keyword_weight, semantic_weight, similarity_threshold, sender, account_id
            )
    
    async def _keyword_search_only(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        vip_only: bool,
        needs_reply: Optional[bool],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
        sender: Optional[str] = None,
        account_id: Optional[str] = None,
        subject_only: bool = True
    ) -> List[Tuple[Email, float, Dict]]:
        """Keyword-only search using PostgreSQL full-text search.

        Args:
            subject_only: If True, only search subject (fast, uses trigram index).
                         Body search is extremely slow (~30s for 582k rows) due to
                         encrypted columns requiring full table scan.
                         Default: True. Use semantic/hybrid mode for body content.
        """
        # Build query
        q = self.db.query(Email).join(
            EmailMetadata, Email.id == EmailMetadata.email_id, isouter=True
        )
        
        # **CRITICAL OPTIMIZATION: Apply account_id and date filters FIRST**
        # This dramatically reduces the search space before text matching
        if account_id:
            q = q.filter(Email.account_id == account_id)
        
        if date_from:
            q = q.filter(Email.date >= date_from)
        
        if date_to:
            q = q.filter(Email.date <= date_to)
        
        # Apply sender filter at DB level (uses index)
        if sender:
            # Use ILIKE for flexible matching (email or domain)
            q = q.filter(Email.from_address.ilike(f'%{sender}%'))
        
        # Keyword search in subject and optionally body
        # **OPTIMIZATION: If query is wildcard (*) and sender filter is active, skip text search**
        # This allows "search by sender only" to be very fast (no text matching needed)
        if query != "*" or not sender:
            search_pattern = f"%{query}%"
            if subject_only:
                # Fast path: only search subject (unencrypted, can be indexed)
                # Body content matching is handled by semantic search via embeddings
                q = q.filter(Email.subject.ilike(search_pattern))
            else:
                # Slow path: search body too (encrypted columns = full table scan)
                # Only use this for pure keyword mode, not hybrid
                q = q.filter(
                    or_(
                        Email.subject.ilike(search_pattern),
                        Email.body_markdown.ilike(search_pattern),
                        Email.body_text.ilike(search_pattern)
                    )
                )
        
        # Apply other filters (category, vip, etc.)
        q = self._apply_filters(q, category, vip_only, needs_reply, None, None)  # Date already applied
        
        # Order by date (most recent first) and limit
        results = q.order_by(Email.date.desc()).limit(top_k).all()
        
        # Calculate simple keyword match scores
        scored_results = []
        for email in results:
            # Simple scoring: count occurrences
            score = self._calculate_keyword_score(email, query)
            breakdown = {
                'keyword_score': score,
                'semantic_score': 0.0,
                'combined_score': score
            }
            scored_results.append((email, score, breakdown))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results
    
    async def _semantic_search_only(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        vip_only: bool,
        needs_reply: Optional[bool],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
        similarity_threshold: float = 0.6,
        sender: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> List[Tuple[Email, float, Dict]]:
        """Semantic-only search using vector similarity."""
        # Request 2x top_k to account for threshold filtering and ensure good results
        vector_top_k = min(top_k * 2, 200)
        
        # Perform vector search with parameterized filters (safe from SQL injection)
        results = await self.vector_search.search_similar(
            query=query,
            top_k=vector_top_k,
            similarity_threshold=similarity_threshold,
            category=category,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
            account_id=account_id,
            sender=sender
        )
        
        # Apply additional filters (VIP, needs_reply)
        filtered_results = []
        for email, similarity in results:
            if vip_only and (not email.email_metadata or not email.email_metadata.vip_level):
                continue
            
            if needs_reply is not None and (not email.email_metadata or email.email_metadata.needs_reply != needs_reply):
                continue
            
            breakdown = {
                'keyword_score': 0.0,
                'semantic_score': similarity,
                'combined_score': similarity
            }
            filtered_results.append((email, similarity, breakdown))
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        vip_only: bool,
        needs_reply: Optional[bool],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
        keyword_weight: float,
        semantic_weight: float,
        similarity_threshold: float = 0.6,
        sender: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> List[Tuple[Email, float, Dict]]:
        """
        True hybrid search: combines keyword and semantic results.
        
        Strategy:
        1. Get top 2*k results from each method
        2. Normalize scores to [0,1]
        3. Combine with weights
        4. Re-rank by combined score
        5. Return top k
        """
        # Get keyword results (subject-only for speed - semantic search handles body content)
        keyword_results = await self._keyword_search_only(
            query, top_k * 2, category, vip_only, needs_reply, date_from, date_to, sender, account_id,
            subject_only=True  # Fast: skip encrypted body columns, let semantic search find body matches
        )
        
        # Get semantic results with passed threshold
        semantic_results = await self._semantic_search_only(
            query, top_k * 2, category, vip_only, needs_reply, date_from, date_to, similarity_threshold, sender, account_id
        )
        
        # Merge results
        email_scores = {}  # email_id -> scores dict
        
        # Add keyword scores
        max_keyword = max([score for _, score, _ in keyword_results], default=1.0)
        for email, score, _ in keyword_results:
            email_scores[email.id] = {
                'email': email,
                'keyword_score': score / max_keyword if max_keyword > 0 else 0,
                'semantic_score': 0.0,
                'keyword_rank_boost': 1.0  # Appeared in keyword results
            }
        
        # Add/merge semantic scores
        for email, score, _ in semantic_results:
            if email.id in email_scores:
                # Update existing
                email_scores[email.id]['semantic_score'] = score
                email_scores[email.id]['semantic_rank_boost'] = 1.0
            else:
                # Add new
                email_scores[email.id] = {
                    'email': email,
                    'keyword_score': 0.0,
                    'semantic_score': score,
                    'keyword_rank_boost': 0.0,
                    'semantic_rank_boost': 1.0
                }
        
        # Calculate combined scores
        combined_results = []
        for email_id, scores in email_scores.items():
            # Base score: weighted combination
            combined = (
                scores['keyword_score'] * keyword_weight +
                scores['semantic_score'] * semantic_weight
            )
            
            # Boost if appeared in both results (high confidence)
            if scores.get('keyword_rank_boost', 0) > 0 and scores.get('semantic_rank_boost', 0) > 0:
                combined *= 1.5  # 50% boost for appearing in both
            
            # Boost recent emails slightly
            email = scores['email']
            if email.date:
                days_old = (datetime.utcnow() - email.date).days
                if days_old < 30:
                    combined *= 1.1  # 10% boost for recent emails
            
            # VIP boost
            if email.email_metadata and email.email_metadata.vip_level:
                vip_boosts = {'urgent': 1.3, 'high': 1.2, 'medium': 1.1}
                combined *= vip_boosts.get(email.email_metadata.vip_level, 1.0)
            
            breakdown = {
                'keyword_score': scores['keyword_score'],
                'semantic_score': scores['semantic_score'],
                'combined_score': combined,
                'in_keyword_results': scores.get('keyword_rank_boost', 0) > 0,
                'in_semantic_results': scores.get('semantic_rank_boost', 0) > 0
            }
            
            combined_results.append((email, combined, breakdown))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results[:top_k]
    
    def _apply_filters(
        self,
        query,
        category: Optional[str],
        vip_only: bool,
        needs_reply: Optional[bool],
        date_from: Optional[datetime],
        date_to: Optional[datetime]
    ):
        """Apply common filters to query."""
        if category:
            query = query.filter(EmailMetadata.ai_category == category)
        
        if vip_only:
            query = query.filter(EmailMetadata.vip_level.in_(['urgent', 'high', 'medium']))
        
        if needs_reply is not None:
            query = query.filter(EmailMetadata.needs_reply == needs_reply)
        
        if date_from:
            query = query.filter(Email.date >= date_from)
        
        if date_to:
            query = query.filter(Email.date <= date_to)
        
        return query
    
    def _calculate_keyword_score(self, email: Email, query: str) -> float:
        """
        Calculate keyword match score.
        
        Simple scoring:
        - Subject match: 3 points per occurrence
        - Body match: 1 point per occurrence
        """
        score = 0.0
        query_lower = query.lower()
        
        # Subject matches (weighted higher)
        if email.subject:
            subject_lower = email.subject.lower()
            matches = subject_lower.count(query_lower)
            score += matches * 3
        
        # Body matches
        body = email.body_markdown or email.body_text or ""
        if body:
            body_lower = body.lower()
            matches = body_lower.count(query_lower)
            score += matches
        
        return float(score)

