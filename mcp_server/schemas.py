"""
Pydantic schemas for MCP tool responses.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class EmailResult(BaseModel):
    """Single email in search results."""
    id: str = Field(description="Email UUID")
    message_id: str = Field(description="RFC822 Message-ID")
    subject: str = Field(description="Email subject")
    from_address: str = Field(description="Sender email address")
    from_name: Optional[str] = Field(None, description="Sender display name")
    date: str = Field(description="Email date (ISO format)")
    
    # Category info
    category: Optional[str] = Field(None, description="AI-assigned category")
    subcategory: Optional[str] = Field(None, description="AI-assigned subcategory")
    
    # Summary (useful for search results)
    summary: Optional[str] = Field(None, description="AI-generated summary")
    
    # Search relevance
    similarity_score: Optional[float] = Field(None, description="Semantic similarity score (0-1)")
    
    # Flags
    is_vip: bool = Field(False, description="Whether sender is VIP")
    needs_reply: bool = Field(False, description="Whether email needs a reply")
    
    class Config:
        from_attributes = True


class EmailDetail(EmailResult):
    """Full email details including body."""
    body: str = Field(description="Email body text")
    to_addresses: List[str] = Field(description="Recipient email addresses")
    cc_addresses: Optional[List[str]] = Field(None, description="CC email addresses")
    has_attachments: bool = Field(False, description="Whether email has attachments")
    attachment_count: int = Field(0, description="Number of attachments")
    
    # Extended metadata
    urgency: Optional[str] = Field(None, description="Urgency level")
    urgency_score: Optional[int] = Field(None, description="Urgency score 1-10")
    action_items: Optional[List[str]] = Field(None, description="Detected action items")
    
    # Application-specific (for PhD applications etc.)
    applicant_name: Optional[str] = Field(None, description="Applicant name (for applications)")
    research_fit_score: Optional[int] = Field(None, description="Research fit score 1-10")
    recommendation_score: Optional[int] = Field(None, description="Overall recommendation 1-10")


class SearchResponse(BaseModel):
    """Response for search operations."""
    query: str = Field(description="Original search query")
    mode: str = Field(description="Search mode used")
    total: int = Field(description="Total results found")
    results: List[EmailResult] = Field(description="Search results")
    
    # Metadata
    search_time_ms: Optional[int] = Field(None, description="Search time in milliseconds")


class SenderInfo(BaseModel):
    """Information about an email sender."""
    email_address: str = Field(description="Email address")
    sender_name: Optional[str] = Field(None, description="Display name")
    domain: Optional[str] = Field(None, description="Email domain")
    email_count: int = Field(description="Total emails from sender")
    is_vip: bool = Field(False, description="Whether sender is VIP")
    typical_category: Optional[str] = Field(None, description="Most common email category")
    first_seen: Optional[str] = Field(None, description="First email date")
    last_seen: Optional[str] = Field(None, description="Most recent email date")


class CategoryInfo(BaseModel):
    """Information about an email category."""
    name: str = Field(description="Category name")
    count: int = Field(description="Number of emails in category")
    description: Optional[str] = Field(None, description="Category description")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
