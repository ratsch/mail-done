"""
Pydantic schemas for FastAPI endpoints
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class EmailMetadataResponse(BaseModel):
    """Email metadata response"""
    vip_level: Optional[str] = None
    intended_color: Optional[int] = None
    priority_score: Optional[int] = None
    ai_category: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_reasoning: Optional[str] = None
    ai_summary: Optional[str] = None
    ai_sentiment: Optional[str] = None
    ai_urgency: Optional[str] = None
    ai_urgency_score: Optional[int] = None
    needs_reply: Optional[bool] = None
    awaiting_reply: Optional[bool] = None
    project_tags: Optional[List[str]] = None
    user_notes: Optional[str] = None
    is_cold_email: Optional[bool] = None
    deadline: Optional[str] = None
    
    # Application-specific fields
    applicant_name: Optional[str] = None
    applicant_institution: Optional[str] = None
    scientific_excellence_score: Optional[int] = None
    scientific_excellence_reason: Optional[str] = None
    recommendation_score: Optional[int] = None
    recommendation_reason: Optional[str] = None
    
    # Invitation/Review-specific fields
    relevance_score: Optional[int] = None
    relevance_reason: Optional[str] = None
    prestige_score: Optional[int] = None
    prestige_reason: Optional[str] = None
    
    # Category-specific data (JSON field with event_date, location, etc.)
    category_specific_data: Optional[Dict[str, Any]] = None
    suggested_labels: Optional[List[str]] = None
    
    class Config:
        from_attributes = True


class ClassificationResponse(BaseModel):
    """Classification response"""
    id: UUID
    classifier_type: str
    category: str
    confidence: float
    reasoning: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class SenderHistoryResponse(BaseModel):
    """Sender history response"""
    email_address: str
    sender_name: Optional[str] = None
    domain: str
    sender_type: Optional[str] = None
    email_count: int
    last_seen: Optional[datetime] = None
    avg_reply_time_hours: Optional[float] = None
    is_frequent: bool
    
    class Config:
        from_attributes = True


class EmailResponse(BaseModel):
    """Email response"""
    id: UUID
    message_id: str
    uid: Optional[str] = None
    folder: str
    from_address: str
    to_addresses: List[str]
    subject: str
    date: datetime
    body_markdown: Optional[str] = None
    has_attachments: bool
    is_seen: bool
    is_flagged: bool
    thread_id: Optional[str] = None
    created_at: datetime
    
    # Relationships
    email_metadata: Optional[EmailMetadataResponse] = None
    classifications: Optional[List[ClassificationResponse]] = None
    
    class Config:
        from_attributes = True


class EmailListResponse(BaseModel):
    """Paginated email list response"""
    emails: List[EmailResponse]
    total: int
    page: int
    page_size: int
    pages: int


class EmailDetailResponse(EmailResponse):
    """Detailed email response with sender history"""
    sender_history: Optional[SenderHistoryResponse] = None
    raw_headers: Optional[Dict[str, Any]] = None


class UpdateMetadataRequest(BaseModel):
    """Update email metadata request"""
    user_notes: Optional[str] = None
    project_tags: Optional[List[str]] = None
    awaiting_reply: Optional[bool] = None
    needs_reply: Optional[bool] = None


class SenderStatsResponse(BaseModel):
    """Sender statistics response"""
    email_address: str
    sender_name: Optional[str] = None
    email_count: int
    last_seen: Optional[datetime] = None
    sender_type: Optional[str] = None
    categories: Dict[str, int]  # category -> count
    avg_reply_time_hours: Optional[float] = None


class StatsResponse(BaseModel):
    """System statistics response"""
    total_emails: int
    emails_today: int
    emails_this_week: int
    vips_configured: int
    rules_configured: int
    ai_classifications: int
    needs_reply_count: int
    flagged_count: int
    unread_count: int
    top_senders: List[Dict[str, Any]]
    categories_breakdown: Dict[str, int]
    folders_breakdown: Dict[str, int] = {}  # Add folders breakdown field
    with_embeddings: int = 0  # Add embeddings count field


class TriggerProcessingRequest(BaseModel):
    """Trigger email processing request"""
    folder: str = Field(default="INBOX", description="Folder to process")
    limit: Optional[int] = Field(default=100, description="Max emails to process")
    new_only: bool = Field(default=True, description="Process only unseen emails")
    use_ai: bool = Field(default=True, description="Use AI classification")


class ProcessingStatusResponse(BaseModel):
    """Processing status response"""
    status: str
    processed: int
    vip_detected: int
    rule_matched: int
    ai_classified: int
    errors: List[str]

