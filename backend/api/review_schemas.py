"""
Pydantic schemas for Lab Application Review System
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class ReviewRequest(BaseModel):
    """Request to submit/update a review"""
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5 stars")
    comment: Optional[str] = Field(None, max_length=2000, description="Optional comment")


class ReviewResponse(BaseModel):
    """Review response"""
    id: UUID
    email_id: UUID
    lab_member_id: UUID
    rater_name: str
    rating: int
    comment: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class ReviewSummaryResponse(BaseModel):
    """Review summary (avg rating, count)"""
    avg_rating: Optional[float]
    num_ratings: int


class DecisionRequest(BaseModel):
    """Request to make a decision"""
    decision: str = Field(..., description="Decision: accept, reject, interview, request_more_info, refer_to_direct_doctorate, delete")
    notes: Optional[str] = Field(None, max_length=5000, description="Optional admin notes")


class DecisionResponse(BaseModel):
    """Decision response"""
    id: UUID
    email_id: UUID
    admin_id: UUID
    admin_name: str
    decision: str
    notes: Optional[str]
    decided_at: datetime
    
    class Config:
        from_attributes = True


class ApplicationListItem(BaseModel):
    """Application list item (summary)"""
    email_id: UUID
    applicant_name: Optional[str]
    applicant_institution: Optional[str]
    date: datetime
    category: Optional[str]
    scientific_excellence_score: Optional[int]
    research_fit_score: Optional[int]
    overall_recommendation_score: Optional[int]
    relevance_score: Optional[int]
    avg_rating: Optional[float]
    num_ratings: int
    my_rating: Optional[int]
    status: str
    review_deadline: Optional[datetime]
    decision: Optional[str]
    received_date: datetime
    email_text_link: Optional[str] = None
    folder_path: Optional[str] = None
    # Technical experience scores
    coding_experience_score: Optional[int] = None
    medical_data_experience_score: Optional[int] = None
    omics_genomics_experience_score: Optional[int] = None
    sequence_analysis_experience_score: Optional[int] = None
    image_analysis_experience_score: Optional[int] = None
    profile_tags: Optional[List[Dict[str, Any]]] = None  # List of tag objects with tag, confidence, reason
    highest_degree: Optional[str] = None
    application_source: Optional[str] = None  # Source of application (e.g., 'ai_center' for AI Center applications)
    
    class Config:
        from_attributes = True


class ApplicationDetailResponse(BaseModel):
    """Full application detail response - includes all fields from reprocess_applications.py export"""
    # Basic email info
    email_id: UUID
    date: datetime
    from_address: Optional[str]
    from_name: Optional[str]
    curated_sender_name: Optional[str]  # Cleaned/extracted sender name
    subject: Optional[str]
    message_id: Optional[str] = None  # RFC822 Message-ID for opening in mail clients
    
    # Classification
    category: Optional[str]
    subcategory: Optional[str]
    confidence: Optional[float]
    application_source: Optional[str] = None  # Source of application (e.g., 'ai_center' for AI Center applications)
    
    # Scores and their explanations
    scientific_excellence_score: Optional[int]
    scientific_excellence_reason: Optional[str]
    research_fit_score: Optional[int]
    research_fit_reason: Optional[str]
    overall_recommendation_score: Optional[int]
    recommendation_reason: Optional[str]  # Explanation for recommendation
    relevance_score: Optional[int]
    relevance_reason: Optional[str]
    prestige_score: Optional[int]  # For invitations/reviews
    prestige_reason: Optional[str]
    urgency_score: Optional[int]
    urgency: Optional[str]  # urgent|normal|low
    urgency_reason: Optional[str]
    sentiment: Optional[str]  # positive|neutral|negative
    
    # AI summary and reasoning
    summary: Optional[str]  # 1-2 sentence summary
    reasoning: Optional[str]  # Classification reasoning
    
    # Status flags
    needs_reply: Optional[bool]
    reply_deadline: Optional[str]  # YYYY-MM-DD
    reply_suggestion: Optional[str]  # Suggested reply content
    action_items: Optional[List[str]]  # Extracted action items
    is_followup: Optional[bool]
    followup_to_date: Optional[str]  # Date of original email for followups
    reprocessed: Optional[bool]
    reprocessed_at: Optional[datetime]
    was_already_processed: Optional[bool]
    passed_filters: Optional[bool]
    
    # Metadata (JSON fields)
    profile_tags: Optional[List[Dict[str, Any]]]
    red_flags: Optional[Dict[str, Any]]
    information_used: Optional[Dict[str, Any]]
    
    # Individual red flags
    is_mass_email: Optional[bool]
    no_research_background: Optional[bool]
    irrelevant_field: Optional[bool]
    possible_spam: Optional[bool]
    insufficient_materials: Optional[bool]
    prompt_manipulation_detected: Optional[bool]
    prompt_manipulation_indicators: Optional[List[str]]
    is_not_application: Optional[bool]
    is_not_application_reason: Optional[str]
    correct_category: Optional[str]
    is_cold_email: Optional[bool]
    
    # Technical experience (scores and evidence)
    coding_experience_score: Optional[int]
    coding_experience_evidence: Optional[str]
    omics_genomics_experience_score: Optional[int]
    omics_genomics_experience_evidence: Optional[str]
    medical_data_experience_score: Optional[int]
    medical_data_experience_evidence: Optional[str]
    sequence_analysis_experience_score: Optional[int]
    sequence_analysis_experience_evidence: Optional[str]
    image_analysis_experience_score: Optional[int]
    image_analysis_experience_evidence: Optional[str]
    
    # Event/invitation specific fields
    event_date: Optional[str]  # YYYY-MM-DD
    deadline: Optional[str]  # Review/response deadline
    location: Optional[str]  # Event location
    time_commitment_hours: Optional[int]
    time_commitment_reason: Optional[str]
    
    # Additional info request
    should_request_additional_info: Optional[bool]
    missing_information_items: Optional[List[str]]
    potential_recommendation_score: Optional[int]
    
    # Applicant info (PII)
    applicant_name: Optional[str]
    applicant_email: Optional[str]  # Applicant's email (from forwarded content if forwarded)
    applicant_institution: Optional[str]
    nationality: Optional[str]
    highest_degree: Optional[str]
    current_situation: Optional[str]
    recent_thesis_title: Optional[str]
    recommendation_source: Optional[str]
    github_account: Optional[str]
    linkedin_account: Optional[str]
    google_scholar_account: Optional[str]
    
    # Evaluation details
    key_strengths: Optional[List[str]]
    concerns: Optional[List[str]]
    next_steps: Optional[str]
    additional_notes: Optional[str]
    ai_reasoning: Optional[str]
    score_reasoning: Optional[Dict[str, Any]]
    
    # AI suggestions
    suggested_folder: Optional[str]
    suggested_labels: Optional[List[str]]
    answer_options: Optional[List[Dict[str, str]]]  # Draft responses [{text, tone}]
    
    # Receipt-specific fields
    vendor: Optional[str]
    amount: Optional[str]
    currency: Optional[str]  # CHF/USD/EUR
    
    # Google Drive files
    folder_path: Optional[str]  # google_drive_folder_id
    email_text_file: Optional[str]
    email_text_link: Optional[str]
    email_text: Optional[str]  # Most recent email text content
    attachments_list: Optional[List[Dict[str, str]]]  # From google_drive_links (includes source metadata)
    llm_response_file: Optional[str]
    llm_response_link: Optional[str]
    
    # Consolidated attachments (from category_specific_data - includes all attachments with full source tracking)
    consolidated_attachments: Optional[List[Dict[str, Any]]] = None  # All attachments from current + prior emails
    reference_letter_attachments: Optional[List[Dict[str, Any]]] = None  # Attachments from reference letters
    
    # Review system fields
    reviews: List[ReviewResponse]
    avg_rating: Optional[float]
    num_ratings: int
    my_review: Optional[ReviewResponse]
    my_private_notes: Optional[str] = None  # Current user's private notes (encrypted, decrypted automatically)
    decision: Optional[DecisionResponse]
    review_deadline: Optional[datetime]
    application_status: str
    status_updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class NotificationResponse(BaseModel):
    """Notification response"""
    id: UUID
    email_id: UUID
    applicant_name: Optional[str]
    category: Optional[str]
    notification_type: str
    deadline: Optional[datetime]
    created_at: datetime
    read_at: Optional[datetime] = None  # When notification was marked as read (null = unread)
    
    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    """User/lab member response"""
    id: UUID
    email: str
    full_name: Optional[str]
    role: str
    can_review: bool
    is_active: bool
    avatar_url: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class CreateUserRequest(BaseModel):
    """Request to create a new user"""
    email: str = Field(..., description="User email")
    full_name: Optional[str] = Field(None, description="Full name")
    role: str = Field("member", description="Role: member or admin")
    can_review: bool = Field(False, description="Can review applications")
    avatar_url: Optional[str] = Field(None, description="Avatar URL from Google Directory API")


class UpdateUserRequest(BaseModel):
    """Request to update a user"""
    can_review: Optional[bool] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    avatar_url: Optional[str] = None


class BatchDecisionRequest(BaseModel):
    """Request for batch decisions"""
    email_ids: List[UUID] = Field(..., description="List of email IDs")
    decision: str = Field(..., description="Decision: accept, reject, interview, request_more_info, refer_to_direct_doctorate, delete")
    notes: Optional[str] = Field(None, description="Optional notes")


class UpdateIsNotApplicationRequest(BaseModel):
    """Request to update is_not_application flag"""
    is_not_application: bool = Field(..., description="Whether this is NOT an application")
    reason: Optional[str] = Field(None, max_length=500, description="Reason for the flag change")


class PrivateNotesRequest(BaseModel):
    """Request to save/update private notes"""
    notes: str = Field(..., max_length=10000, description="Private notes (not visible to others)")


class BatchUserRequest(BaseModel):
    """Request for batch user creation"""
    users: List[CreateUserRequest] = Field(..., description="List of users to create")


class BatchUserResponse(BaseModel):
    """Response for batch user creation"""
    created: int
    failed: int
    results: List[dict] = Field(..., description="List of creation results with status and user_id or error")


class AuditLogResponse(BaseModel):
    """Audit log entry response"""
    id: UUID
    user_id: UUID
    user_name: Optional[str]
    email_id: UUID
    applicant_name: Optional[str]
    action_type: str
    action_details: Optional[Dict[str, Any]]
    ip_address: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class AvailableTagsResponse(BaseModel):
    """Response for available tags endpoint"""
    tags: List[str] = Field(..., description="List of unique tag names")

