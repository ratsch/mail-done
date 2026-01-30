"""
Email annotations and notes system.
Supports both IMAP keywords (visible in Mail.app) and database storage (private metadata).
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class EmailNote(BaseModel):
    """
    Note/annotation attached to an email.
    Can be stored in database (Phase 2) or as IMAP flag.
    """
    email_uid: str
    note_type: str = Field(..., description="Type: ai_summary, user_note, classification_reason, etc.")
    content: str = Field(..., description="Note content")
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="system", description="Who created: system, user, ai")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional structured data")
    
    # Privacy settings
    private: bool = Field(default=False, description="If True, never sync to IMAP (database only)")
    visible_in_mail: bool = Field(default=False, description="If True, can use IMAP keyword")


class EmailMetadata(BaseModel):
    """
    Extended metadata for an email (stored in database, not IMAP).
    This is private information that should never leak out.
    """
    email_uid: str
    
    # AI-generated metadata (private)
    ai_summary: Optional[str] = Field(None, description="AI-generated summary")
    ai_sentiment: Optional[str] = Field(None, description="Sentiment: positive, negative, neutral")
    ai_action_items: List[str] = Field(default_factory=list, description="Extracted action items")
    ai_entities: Optional[Dict[str, List[str]]] = Field(None, description="Extracted entities (people, orgs, dates)")
    
    # Classification metadata (private)
    classification_reasoning: Optional[str] = Field(None, description="Why it was classified this way")
    classification_confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score 0-1")
    classification_model: Optional[str] = Field(None, description="Model used for classification")
    
    # User annotations (private)
    user_notes: List[str] = Field(default_factory=list, description="User-added notes")
    user_tags: List[str] = Field(default_factory=list, description="User-added tags")
    
    # Processing metadata (private)
    processed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = None
    
    # Business context (private)
    related_project: Optional[str] = None
    priority_level: Optional[int] = Field(None, ge=1, le=5, description="1=lowest, 5=highest")
    deadline: Optional[datetime] = None
    
    # Security/privacy flags
    contains_pii: bool = Field(default=False, description="Contains personal identifiable information")
    contains_sensitive: bool = Field(default=False, description="Contains sensitive content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email_uid": "12345",
                "ai_summary": "Meeting request for Q4 planning session",
                "ai_sentiment": "neutral",
                "ai_action_items": ["Schedule meeting", "Review Q4 goals"],
                "classification_reasoning": "Work email with meeting invite",
                "user_notes": ["Important - need to prepare slides"],
                "related_project": "Q4_Planning",
                "priority_level": 4,
                "contains_pii": False
            }
        }


class AnnotationManager:
    """
    Manages email annotations using both IMAP flags and database storage.
    
    Design:
    - Simple labels → IMAP keywords (visible in Mail.app)
    - Detailed metadata → Database (private, never synced to IMAP)
    """
    
    def __init__(self, imap_monitor=None, database=None):
        """
        Initialize annotation manager.
        
        Args:
            imap_monitor: IMAPMonitor instance for flag operations
            database: Database connection (for Phase 2)
        """
        self.imap = imap_monitor
        self.database = database
        self.pending_metadata: Dict[str, EmailMetadata] = {}  # In-memory until DB ready
        
    def add_simple_note(self, email_uid: str, label: str) -> bool:
        """
        Add a simple note as IMAP keyword (visible in Mail.app).
        
        Examples: 
        - "Important"
        - "FollowUp" 
        - "WaitingReply"
        - "Reviewed"
        
        Args:
            email_uid: Email UID
            label: Simple label (will be converted to IMAP keyword)
            
        Returns:
            True if added successfully
        """
        if not self.imap:
            logger.warning("No IMAP connection - note not added")
            return False
        
        # Convert to safe IMAP keyword
        safe_label = label.replace(' ', '_').replace('-', '_')
        
        return self.imap.add_custom_flag(email_uid, safe_label)
    
    def add_detailed_metadata(self, metadata: EmailMetadata) -> bool:
        """
        Add detailed metadata (stored in database, NOT in IMAP).
        This information stays private and never leaks to email server.
        
        Args:
            metadata: EmailMetadata with all details
            
        Returns:
            True if stored successfully
        """
        if self.database:
            # Phase 2: Store in database
            try:
                self.database.save_metadata(metadata)
                logger.info(f"Saved metadata for email {metadata.email_uid} to database")
                return True
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")
                return False
        else:
            # Phase 1: Store in memory (temporary)
            self.pending_metadata[metadata.email_uid] = metadata
            logger.info(f"Metadata for {metadata.email_uid} stored in memory (database not yet configured)")
            return True
    
    def add_ai_generated_note(self,
                             email_uid: str,
                             summary: Optional[str] = None,
                             action_items: Optional[List[str]] = None,
                             classification_reason: Optional[str] = None,
                             entities: Optional[Dict[str, List[str]]] = None) -> bool:
        """
        Add AI-generated metadata (PRIVATE - never visible in IMAP).
        
        Args:
            email_uid: Email UID
            summary: AI-generated summary
            action_items: Extracted action items
            classification_reason: Why it was classified this way
            entities: Extracted entities (people, orgs, etc.)
            
        Returns:
            True if stored successfully
        """
        metadata = EmailMetadata(
            email_uid=email_uid,
            ai_summary=summary,
            ai_action_items=action_items or [],
            classification_reasoning=classification_reason,
            ai_entities=entities,
            processed_at=datetime.now()
        )
        
        return self.add_detailed_metadata(metadata)
    
    def add_user_note(self,
                     email_uid: str,
                     note: str,
                     visible_in_mail: bool = False,
                     tags: Optional[List[str]] = None) -> bool:
        """
        Add a user note to an email.
        
        Args:
            email_uid: Email UID
            note: Note text
            visible_in_mail: If True, also add as IMAP keyword
            tags: Optional tags
            
        Returns:
            True if added successfully
        """
        # Store detailed note in database/memory
        metadata = self.pending_metadata.get(email_uid, EmailMetadata(email_uid=email_uid))
        metadata.user_notes.append(note)
        if tags:
            metadata.user_tags.extend(tags)
        
        self.add_detailed_metadata(metadata)
        
        # Optionally add simple label to IMAP (visible in Mail.app)
        if visible_in_mail and tags:
            for tag in tags:
                self.add_simple_note(email_uid, tag)
        
        return True
    
    def mark_contains_pii(self, email_uid: str, pii_types: Optional[List[str]] = None) -> bool:
        """
        Mark email as containing PII (for privacy/security tracking).
        This is PRIVATE metadata, never visible in IMAP.
        
        Args:
            email_uid: Email UID
            pii_types: Types of PII found (e.g., ["ssn", "credit_card", "address"])
            
        Returns:
            True if marked successfully
        """
        metadata = self.pending_metadata.get(email_uid, EmailMetadata(email_uid=email_uid))
        metadata.contains_pii = True
        
        # Store PII types in ai_entities for now (we don't have a generic metadata dict)
        if pii_types:
            if not metadata.ai_entities:
                metadata.ai_entities = {}
            metadata.ai_entities['pii_types'] = pii_types
        
        logger.warning(f"Email {email_uid} marked as containing PII: {pii_types}")
        return self.add_detailed_metadata(metadata)
    
    def set_project_context(self,
                           email_uid: str,
                           project: str,
                           priority: Optional[int] = None,
                           deadline: Optional[datetime] = None) -> bool:
        """
        Associate email with a project (PRIVATE metadata).
        
        Args:
            email_uid: Email UID
            project: Project name/ID
            priority: Priority level (1-5)
            deadline: Project deadline
            
        Returns:
            True if set successfully
        """
        metadata = self.pending_metadata.get(email_uid, EmailMetadata(email_uid=email_uid))
        metadata.related_project = project
        metadata.priority_level = priority
        metadata.deadline = deadline
        
        # Optionally add visible flag in Mail.app
        if self.imap:
            self.add_simple_note(email_uid, f"Project_{project}")
        
        return self.add_detailed_metadata(metadata)
    
    def get_metadata(self, email_uid: str) -> Optional[EmailMetadata]:
        """
        Retrieve metadata for an email.
        
        Args:
            email_uid: Email UID
            
        Returns:
            EmailMetadata if found, None otherwise
        """
        if self.database:
            # Phase 2: Query database
            return self.database.get_metadata(email_uid)
        else:
            # Phase 1: Return from memory
            return self.pending_metadata.get(email_uid)
    
    def export_metadata_for_database(self) -> List[EmailMetadata]:
        """
        Export all pending metadata for database migration.
        Useful when transitioning from Phase 1 to Phase 2.
        
        Returns:
            List of all EmailMetadata objects
        """
        return list(self.pending_metadata.values())
    
    def clear_pending(self):
        """Clear pending metadata (after database migration)"""
        count = len(self.pending_metadata)
        self.pending_metadata.clear()
        logger.info(f"Cleared {count} pending metadata entries")


# Predefined IMAP keywords for common use cases
class IMAPKeywords:
    """Standard IMAP keywords that work across email clients"""
    
    # Standard flags
    IMPORTANT = "$Important"
    FOLLOW_UP = "$FollowUp"
    REVIEWED = "$Reviewed"
    WAITING_REPLY = "$WaitingReply"
    DELEGATED = "$Delegated"
    
    # Custom project/context flags
    @staticmethod
    def project(name: str) -> str:
        """Generate project flag"""
        safe_name = name.replace(' ', '_').replace('-', '_')[:20]
        return f"$Project_{safe_name}"
    
    @staticmethod
    def priority(level: int) -> str:
        """Generate priority flag (1-5)"""
        return f"$Priority{level}"

