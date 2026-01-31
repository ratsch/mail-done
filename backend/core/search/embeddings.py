"""
Embedding Generator

Generates vector embeddings for emails to enable semantic search.
Supports both OpenAI and Azure OpenAI embedding models.
"""
import os
import hashlib
import logging
import numpy as np
import time
from typing import List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from openai import OpenAI, AzureOpenAI, RateLimitError, APIError

from backend.core.database.models import Email, EmailMetadata

logger = logging.getLogger(__name__)

# Attachment text limits for embedding
MAX_ATTACHMENT_TEXT_CHARS = 5000  # Per attachment
MAX_ATTACHMENTS_FOR_EMBEDDING = 5  # Maximum attachments to include


class EmbeddingGenerator:
    """Generate and manage vector embeddings for emails."""
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ):
        """
        Initialize embedding generator.
        
        Args:
            provider: "openai" or "azure" (default: from EMBEDDING_PROVIDER env var, fallback to "openai")
            model: Embedding model to use
                - text-embedding-3-small: 1536 dims, $0.02/1M tokens (recommended)
                - text-embedding-3-large: 3072 dims, $0.13/1M tokens (higher quality)
                - text-embedding-ada-002: 1536 dims, $0.10/1M tokens (legacy)
            batch_size: Number of emails to process at once
        """
        self.model = model
        self.batch_size = batch_size
        
        # Get provider config from llm_endpoints.yaml (or fall back to env vars)
        from backend.core.ai.llm_config import get_model_config
        cfg_provider, cfg_api_key, cfg_endpoint, cfg_api_version = get_model_config(model)
        
        # Use explicit provider if given, otherwise use config, then env var
        self.provider = provider or cfg_provider or os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        
        # Initialize client based on provider
        if self.provider == "azure":
            api_key = cfg_api_key or os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(f"Azure API key not found for embedding model '{model}'")
            
            azure_endpoint = cfg_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError(f"Azure endpoint not found for embedding model '{model}'")
            
            api_version = cfg_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
            
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
            logger.info(f"Using Azure OpenAI for embeddings: {model} (endpoint: {azure_endpoint[:50]}...)")
        else:
            # Default to OpenAI
            api_key = cfg_api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url=cfg_endpoint)
            logger.info(f"Using OpenAI for embeddings: {model}")
        
        # Embedding dimensions by model
        self.embedding_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        self.dims = self.embedding_dims.get(model, 1536)
        
        # Embedding pricing (per 1M tokens)
        self.pricing = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10
        }
        
        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Validate model is recognized
        if model not in self.embedding_dims:
            logger.warning(f"Unknown embedding model '{model}' - defaulting to 1536 dimensions. "
                         f"Known models: {list(self.embedding_dims.keys())}")
        
        # Database cost tracking (optional, set later)
        self.db_cost_tracker = None
    
    def set_cost_tracker_session(self, db_session, source: str = "cli"):
        """Set database session for cost tracking."""
        if db_session and not self.db_cost_tracker:
            try:
                from backend.core.database.cost_tracking import CostTracker
                self.db_cost_tracker = CostTracker(db_session, source=source)
            except Exception as e:
                logger.debug(f"Could not set cost tracker: {e}")
    
    def generate_embedding(
        self,
        email: Email,
        metadata: Optional[EmailMetadata] = None,
        max_retries: int = 3
    ) -> Tuple[List[float], str]:
        """
        Generate embedding vector for a single email with retry logic.
        
        Args:
            email: Email to embed
            metadata: Email metadata (optional, for category context)
            max_retries: Maximum retry attempts for API failures
        
        Returns:
            Tuple of (embedding vector, content hash)
        """
        # Prepare text for embedding
        text = self._prepare_text(email, metadata)
        
        # Generate content hash (for change detection)
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Generate embedding with retry logic
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                embedding = response.data[0].embedding
                
                # Track usage and cost
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens
                    cost = (tokens_used / 1_000_000) * self.pricing.get(self.model, 0.02)
                    self.total_tokens += tokens_used
                    self.total_cost += cost
                    logger.debug(f"Embedding API usage: {tokens_used} tokens, cost: ${cost:.6f}")
                    
                    # Log to database if tracker available
                    if self.db_cost_tracker:
                        try:
                            self.db_cost_tracker.log_usage(
                                model=self.model,
                                task='embedding',
                                prompt_tokens=tokens_used,  # Embeddings don't separate prompt/completion
                                completion_tokens=0,
                                total_tokens=tokens_used,
                                cost_usd=cost,
                                email_id=None,  # Will be set by caller if available
                                context_data={}
                            )
                        except Exception as e:
                            logger.debug(f"Failed to log embedding usage to database: {e}")
                
                # Validate embedding dimension
                if len(embedding) != self.dims:
                    logger.error(f"Embedding dimension mismatch: expected {self.dims}, got {len(embedding)}")
                    return [0.0] * self.dims, content_hash
                
                return embedding, content_hash
                
            except RateLimitError as e:
                # Rate limit - use exponential backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0  # 1s, 2s, 4s
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts for email {email.id}")
                    return [0.0] * self.dims, content_hash
                    
            except APIError as e:
                # API error - retry with backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    logger.warning(f"API error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error after {max_retries} attempts for email {email.id}: {e}")
                    return [0.0] * self.dims, content_hash
                    
            except Exception as e:
                logger.error(f"Unexpected error generating embedding for email {email.id}: {e}", exc_info=True)
                return [0.0] * self.dims, content_hash
        
        # Should never reach here, but return zero vector as safeguard
        return [0.0] * self.dims, content_hash
    
    def generate_batch_embeddings(
        self,
        emails: List[Email],
        metadata_dict: Optional[dict] = None
    ) -> List[Tuple[str, List[float], str]]:
        """
        Generate embeddings for multiple emails in batch.
        
        Args:
            emails: List of emails to embed
            metadata_dict: Dict mapping email.id -> EmailMetadata
        
        Returns:
            List of (email_id, embedding, content_hash) tuples
        """
        results = []
        metadata_dict = metadata_dict or {}
        
        # Process in batches
        for i in range(0, len(emails), self.batch_size):
            batch = emails[i:i + self.batch_size]
            
            # Prepare texts
            texts = []
            email_ids = []
            content_hashes = []
            
            for email in batch:
                metadata = metadata_dict.get(str(email.id))
                text = self._prepare_text(email, metadata)
                content_hash = hashlib.sha256(text.encode()).hexdigest()
                
                texts.append(text)
                email_ids.append(str(email.id))
                content_hashes.append(content_hash)
            
            # Generate embeddings for batch
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                
                # Extract embeddings
                for j, embedding_data in enumerate(response.data):
                    results.append((
                        email_ids[j],
                        embedding_data.embedding,
                        content_hashes[j]
                    )                        )
                        
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
                # Add zero vectors for failed batch
                for j in range(len(batch)):
                    results.append((
                        email_ids[j],
                        [0.0] * self.dims,
                        content_hashes[j]
                    ))
        
        return results
    
    def _prepare_text(
        self,
        email: Email,
        metadata: Optional[EmailMetadata] = None
    ) -> str:
        """
        Prepare email text for embedding.

        Combines subject, sender, category, body, and attachment text for
        comprehensive representation. Weighted to prioritize subject and key metadata.

        Args:
            email: Email to prepare
            metadata: Email metadata

        Returns:
            Prepared text string
        """
        parts = []

        # Subject (weighted 2x by repeating)
        if email.subject:
            parts.append(f"Subject: {email.subject}")
            parts.append(email.subject)  # Repeat for emphasis

        # Category (if available)
        if metadata and metadata.ai_category:
            parts.append(f"Category: {metadata.ai_category}")

        # Sender name/domain (for context)
        if email.from_name:
            parts.append(f"From: {email.from_name}")
        elif email.from_address:
            sender_name = email.from_address.split('@')[0]
            parts.append(f"From: {sender_name}")

        # Body (truncate to ~2000 chars to stay within token limits)
        # OpenAI embedding models have 8191 token limit for text-embedding-3-*
        # Approximate: 1 token ≈ 4 chars, so 2000 chars ≈ 500 tokens
        # Total with subject/metadata: ~600-800 tokens (well within limit)
        body = email.body_markdown or email.body_text or ""
        if body:
            # Clean up body
            body = body.strip()
            # Truncate if too long (leaving room for subject/metadata)
            max_body_length = 2000
            if len(body) > max_body_length:
                body = body[:max_body_length] + "..."
            parts.append(body)

        # Attachment text (Phase 0: include extracted text from attachments)
        # This enables semantic search over attachment content
        attachment_texts = self._extract_attachment_texts(email)
        if attachment_texts:
            parts.extend(attachment_texts)

        # AI summary (if available, very valuable for embedding)
        if metadata and metadata.ai_summary:
            parts.append(f"Summary: {metadata.ai_summary}")

        # Combine all parts
        text = "\n".join(parts)

        # Final length check (OpenAI limit is ~8000 tokens)
        if len(text) > 6000:  # Conservative limit
            text = text[:6000] + "..."

        return text

    def _extract_attachment_texts(self, email: Email) -> List[str]:
        """
        Extract text content from email attachments.

        Reads the attachment_info JSON column which contains extracted_text
        for each attachment that was successfully processed.

        Args:
            email: Email with attachment_info

        Returns:
            List of formatted attachment text strings
        """
        attachment_texts = []

        if not email.attachment_info:
            return attachment_texts

        # attachment_info is a JSON list of AttachmentInfo dicts
        attachments = email.attachment_info
        if not isinstance(attachments, list):
            return attachment_texts

        for i, att in enumerate(attachments[:MAX_ATTACHMENTS_FOR_EMBEDDING]):
            if not isinstance(att, dict):
                continue

            extracted_text = att.get('extracted_text')
            if not extracted_text or not extracted_text.strip():
                continue

            filename = att.get('filename', f'Attachment {i+1}')

            # Truncate long attachment text
            if len(extracted_text) > MAX_ATTACHMENT_TEXT_CHARS:
                extracted_text = extracted_text[:MAX_ATTACHMENT_TEXT_CHARS] + "..."

            attachment_texts.append(f"[Attachment: {filename}]\n{extracted_text}")

        return attachment_texts
    
    def generate_query_embedding(self, query: str, max_retries: int = 5, timeout: int = 60) -> List[float]:
        """
        Generate embedding for a search query with retry logic and timeout handling.
        
        Args:
            query: Search query text
            max_retries: Maximum retry attempts for API failures (default: 5)
            timeout: Timeout per request in seconds (default: 60 for stability)
        
        Returns:
            Embedding vector (or zero vector if all retries fail)
        """
        for attempt in range(max_retries):
            try:
                # Create embedding with timeout
                response = self.client.embeddings.create(
                    model=self.model,
                    input=query,
                    timeout=timeout
                )
                embedding = response.data[0].embedding
                
                # Track usage and cost
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens
                    cost = (tokens_used / 1_000_000) * self.pricing.get(self.model, 0.02)
                    self.total_tokens += tokens_used
                    self.total_cost += cost
                    logger.debug(f"Query embedding API usage: {tokens_used} tokens, cost: ${cost:.6f}")
                    
                    # Log to database if tracker available
                    if self.db_cost_tracker:
                        try:
                            self.db_cost_tracker.log_usage(
                                model=self.model,
                                task='search_query',
                                prompt_tokens=tokens_used,
                                completion_tokens=0,
                                total_tokens=tokens_used,
                                cost_usd=cost,
                                email_id=None,
                                context_data={'query': query[:100]}  # First 100 chars of query
                            )
                        except Exception as e:
                            logger.debug(f"Failed to log query embedding usage to database: {e}")
                
                # Validate dimension
                if len(embedding) != self.dims:
                    logger.error(f"Query embedding dimension mismatch: expected {self.dims}, got {len(embedding)}")
                    return [0.0] * self.dims
                
                logger.info(f"Query embedding generated successfully (attempt {attempt + 1})")
                return embedding
                
            except RateLimitError as e:
                # Rate limit - use longer backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2.0  # 2s, 4s, 8s, 16s, 32s
                    logger.warning(f"Rate limit on query embedding, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts for query: {query}")
                    return [0.0] * self.dims
                    
            except APIError as e:
                # API error (including timeout) - use exponential backoff
                if attempt < max_retries - 1:
                    # Longer backoff for timeouts
                    base_wait = 2.0 if 'timeout' in str(e).lower() else 1.0
                    wait_time = (2 ** attempt) * base_wait  # 2s, 4s, 8s, 16s, 32s or 1s, 2s, 4s, 8s, 16s
                    logger.warning(f"API error on query embedding, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error after {max_retries} attempts for query '{query}': {e}")
                    return [0.0] * self.dims
                    
            except Exception as e:
                # Unexpected error - retry with shorter backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s, 4s, 8s
                    logger.warning(f"Unexpected error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error after {max_retries} attempts for query embedding: {e}", exc_info=True)
                    return [0.0] * self.dims
        
        # Should never reach here, but return zero vector
        logger.error(f"All {max_retries} retry attempts exhausted for query: {query}")
        return [0.0] * self.dims
    
    async def embed_all_emails(
        self,
        db: Session,
        skip_existing: bool = True,
        force_regenerate: bool = False
    ) -> dict:
        """
        Generate embeddings for all emails in database.
        
        Args:
            db: Database session
            skip_existing: Skip emails that already have embeddings
            force_regenerate: Regenerate even if content hasn't changed
        
        Returns:
            Statistics dict (processed, skipped, errors)
        """
        from backend.core.database.models import EmailEmbedding
        
        stats = {
            'total': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'new': 0,
            'updated': 0
        }
        
        # Get all emails
        emails = db.query(Email).all()
        stats['total'] = len(emails)
        
        logger.info(f"Processing {stats['total']:,} emails for embedding generation")
        
        # Get existing embeddings
        existing_embeddings = {}
        if skip_existing:
            embeddings = db.query(EmailEmbedding).all()
            existing_embeddings = {
                str(emb.email_id): emb 
                for emb in embeddings
            }
            logger.info(f"Found {len(existing_embeddings):,} existing embeddings")
        
        # Get metadata for all emails
        metadata_dict = {}
        for email in emails:
            if email.email_metadata:
                metadata_dict[str(email.id)] = email.email_metadata
        
        # Filter emails to process
        emails_to_process = []
        for email in emails:
            email_id_str = str(email.id)
            
            if email_id_str in existing_embeddings and skip_existing:
                existing = existing_embeddings[email_id_str]
                
                # Check if content changed
                if not force_regenerate:
                    text = self._prepare_text(email, metadata_dict.get(email_id_str))
                    content_hash = hashlib.sha256(text.encode()).hexdigest()
                    
                    if existing.content_hash == content_hash:
                        stats['skipped'] += 1
                        continue
            
            emails_to_process.append(email)
        
        logger.info(f"Will process {len(emails_to_process):,} emails (skipped {stats['skipped']:,})")
        
        # Generate embeddings in batches
        results = self.generate_batch_embeddings(emails_to_process, metadata_dict)
        
        # Store embeddings
        for email_id_str, embedding, content_hash in results:
            try:
                # Check if embedding exists
                existing = existing_embeddings.get(email_id_str)
                
                if existing:
                    # Update existing
                    existing.embedding = embedding
                    existing.content_hash = content_hash
                    existing.embedding_model = self.model
                    stats['updated'] += 1
                else:
                    # Create new
                    new_embedding = EmailEmbedding(
                        email_id=email_id_str,
                        embedding=embedding,
                        content_hash=content_hash,
                        embedding_model=self.model
                    )
                    db.add(new_embedding)
                    stats['new'] += 1
                
                stats['processed'] += 1
                
                # Commit periodically
                if stats['processed'] % 100 == 0:
                    db.commit()
                    logger.info(f"Progress: {stats['processed']:,}/{len(emails_to_process):,} emails processed")
                    
            except Exception as e:
                logger.error(f"Error storing embedding for {email_id_str}: {e}", exc_info=True)
                stats['errors'] += 1
        
        # Final commit
        db.commit()
        
        logger.info(f"Embedding generation complete!")
        logger.info(f"Total: {stats['total']:,}, Processed: {stats['processed']:,}, "
                   f"Skipped: {stats['skipped']:,}, Errors: {stats['errors']}")
        logger.info(f"New: {stats['new']:,}, Updated: {stats['updated']:,}")
        
        return stats
    
    def cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Convert to numpy for efficient computation
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, float(similarity)))
    
    def get_usage_stats(self) -> dict:
        """Get cumulative usage and cost statistics."""
        return {
            'model': self.model,
            'total_tokens': self.total_tokens,
            'total_cost_usd': round(self.total_cost, 4),
            'cost_per_1k_tokens': round((self.total_cost / max(1, self.total_tokens)) * 1000, 6) if self.total_tokens > 0 else 0
        }

