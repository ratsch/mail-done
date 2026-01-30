"""
Email Filter

Filters emails based on various criteria for selective processing.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from backend.core.database.models import Email, EmailMetadata, EmailEmbedding


class EmailFilter:
    """Filter emails for selective processing."""
    
    def __init__(self, db: Session):
        """
        Initialize email filter.
        
        Args:
            db: Database session for filter queries
        """
        self.db = db
    
    def should_process_email(
        self,
        email: Email,
        filter_category: Optional[str] = None,
        filter_unclassified: bool = False,
        filter_no_embedding: bool = False,
        filter_needs_reply: bool = False,
        filter_vip_only: bool = False,
        filter_date_after: Optional[datetime] = None,
        filter_date_before: Optional[datetime] = None
    ) -> bool:
        """
        Determine if email should be processed based on filters.
        
        Args:
            email: Email to check
            filter_category: Only process this category
            filter_unclassified: Only process emails without AI classification
            filter_no_embedding: Only process emails without embeddings
            filter_needs_reply: Only process emails needing replies
            filter_vip_only: Only process VIP emails
            filter_date_after: Only process emails after this date
            filter_date_before: Only process emails before this date
        
        Returns:
            True if email passes all filters
        """
        # Date filters (can apply without database)
        # Ensure timezone-aware comparison
        if filter_date_after and email.date:
            email_date = email.date
            if email_date.tzinfo is None:
                from datetime import timezone
                email_date = email_date.replace(tzinfo=timezone.utc)
            if email_date < filter_date_after:
                return False
        
        if filter_date_before and email.date:
            email_date = email.date
            if email_date.tzinfo is None:
                from datetime import timezone
                email_date = email_date.replace(tzinfo=timezone.utc)
            if email_date > filter_date_before:
                return False
        
        # Get metadata if needed for other filters
        metadata = None
        if any([filter_category, filter_unclassified, filter_needs_reply, filter_vip_only]):
            metadata = self.db.query(EmailMetadata).filter(
                EmailMetadata.email_id == email.id
            ).first()
        
        # Category filter
        if filter_category:
            if not metadata or metadata.ai_category != filter_category:
                return False
        
        # Unclassified filter
        if filter_unclassified:
            if metadata and metadata.ai_category:
                return False  # Skip classified emails
        
        # Needs reply filter
        if filter_needs_reply:
            if not metadata or not metadata.needs_reply:
                return False
        
        # VIP filter
        if filter_vip_only:
            if not metadata or not metadata.vip_level:
                return False
        
        # No embedding filter
        if filter_no_embedding:
            embedding = self.db.query(EmailEmbedding).filter(
                EmailEmbedding.email_id == email.id
            ).first()
            if embedding:
                return False  # Skip emails with embeddings
        
        # Passed all filters
        return True
    
    def get_filtered_email_ids(
        self,
        folder: str = "INBOX",
        filter_category: Optional[str] = None,
        filter_unclassified: bool = False,
        filter_no_embedding: bool = False,
        filter_needs_reply: bool = False,
        filter_vip_only: bool = False,
        filter_date_after: Optional[datetime] = None,
        filter_date_before: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> list:
        """
        Get list of email IDs that match filters (database-based).
        More efficient than filtering in memory.
        
        Returns:
            List of email message_ids to process
        """
        # Build query
        query = self.db.query(Email.message_id, Email.uid).filter(
            Email.folder == folder
        )
        
        # Date filters
        if filter_date_after:
            query = query.filter(Email.date >= filter_date_after)
        if filter_date_before:
            query = query.filter(Email.date <= filter_date_before)
        
        # Join metadata if needed
        if any([filter_category, filter_unclassified, filter_needs_reply, filter_vip_only]):
            query = query.join(EmailMetadata, Email.id == EmailMetadata.email_id, isouter=True)
        
        # Category filter
        if filter_category:
            query = query.filter(EmailMetadata.ai_category == filter_category)
        
        # Unclassified filter
        if filter_unclassified:
            query = query.filter(EmailMetadata.ai_category.is_(None))
        
        # Needs reply filter
        if filter_needs_reply:
            query = query.filter(EmailMetadata.needs_reply == True)
        
        # VIP filter
        if filter_vip_only:
            query = query.filter(EmailMetadata.vip_level.in_(['urgent', 'high', 'medium']))
        
        # No embedding filter
        if filter_no_embedding:
            # Left join to find emails WITHOUT embeddings
            query = query.outerjoin(EmailEmbedding, Email.id == EmailEmbedding.email_id)
            query = query.filter(EmailEmbedding.id.is_(None))
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        # Execute and return message_ids
        results = query.all()
        return [(msg_id, uid) for msg_id, uid in results]

