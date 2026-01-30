"""
SQLAlchemy Database Models for Phase 2

Stores:
- Email metadata (colors, priorities, AI insights)
- Sender history (patterns, statistics)
- Classifications (audit trail)
- Reply tracking
- API usage and cost tracking

Encryption:
- Sensitive fields are encrypted at rest using Fernet (AES-128 CBC)
- Encrypted fields: email bodies, AI summaries/reasoning, user notes, embeddings
- Unencrypted fields: subjects, email addresses, names, scores (for search/filtering)
- See backend/core/database/encryption.py for implementation
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index, Date
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

# Import encryption types
from backend.core.database.encryption import EncryptedText, EncryptedJSON

# pgvector for native vector support
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    # Fallback to JSON if pgvector not installed
    HAS_PGVECTOR = False
    print("WARNING: pgvector not installed. Install with: pip install pgvector")

Base = declarative_base()


class Email(Base):
    """
    Core email record - mirrors IMAP email but with database persistence.
    Links to rich metadata via email_metadata table.
    
    Multi-account support: account_id tracks which account this email belongs to.
    """
    __tablename__ = "emails"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(String(500), nullable=False, index=True)  # RFC822 Message-ID (unique per account)
    uid = Column(String(100), nullable=False)  # IMAP UID
    folder = Column(String(200), nullable=False, default="INBOX")
    
    # Multi-account support
    account_id = Column(String(50), nullable=False, default="work", index=True)
    original_account_id = Column(String(50))  # Where email first appeared
    moved_from_email_id = Column(UUID(as_uuid=True))  # If moved from another account
    
    # Headers (UNENCRYPTED for search/filtering)
    from_address = Column(String(500), nullable=False, index=True)
    from_name = Column(String(500))
    to_addresses = Column(JSON, nullable=False)  # List of email addresses
    cc_addresses = Column(JSON, default=list)
    subject = Column(Text, nullable=False)  # UNENCRYPTED for subject search
    date = Column(DateTime, nullable=False, index=True)
    
    # Content (ENCRYPTED - most sensitive data)
    body_markdown = Column(EncryptedText)  # ENCRYPTED
    body_text = Column(EncryptedText)      # ENCRYPTED
    body_html = Column(EncryptedText)      # ENCRYPTED
    
    # Metadata
    has_attachments = Column(Boolean, default=False)
    attachment_count = Column(Integer, default=0)
    attachment_info = Column(JSON, default=list)  # List of attachment metadata
    
    # Headers for preprocessing
    raw_headers = Column(JSON, default=dict)
    
    # Threading
    thread_id = Column(String(500))  # Index defined in __table_args__
    references = Column(Text)  # Full References header
    in_reply_to = Column(String(500))
    
    # Flags
    is_seen = Column(Boolean, default=False)
    is_flagged = Column(Boolean, default=False)
    is_answered = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    email_metadata = relationship(
        "EmailMetadata",
        back_populates="email",
        foreign_keys="EmailMetadata.email_id",
        uselist=False,
        cascade="all, delete-orphan"
    )
    classifications = relationship("Classification", back_populates="email", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_emails_from_date', 'from_address', 'date'),
        Index('ix_emails_folder_date', 'folder', 'date'),
        Index('ix_emails_thread_id', 'thread_id'),  # For thread lookups
        Index('ix_emails_is_seen_date', 'is_seen', 'date'),  # For new email queries
        Index('ix_emails_account_message_id', 'account_id', 'message_id', unique=True),  # Unique per account
    )


class EmailMetadata(Base):
    """
    Rich metadata for emails - bypasses IMAP limitations.
    Stores VIP levels, colors, AI analysis, notes.
    """
    __tablename__ = "email_metadata"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), unique=True, nullable=False)
    
    # VIP and Priority (UNENCRYPTED - for filtering)
    vip_level = Column(String(20))  # urgent/high/medium/null
    intended_color = Column(Integer)  # 1-7 for UI display (bypasses IMAP!)
    priority_score = Column(Integer)  # UNENCRYPTED - quantitative score for filtering
    
    # AI Classification (UNENCRYPTED - for filtering/display)
    ai_category = Column(String(50))
    ai_subcategory = Column(String(50))
    ai_confidence = Column(Float)  # 0-1
    ai_reasoning = Column(EncryptedText)  # ENCRYPTED - may quote email content
    ai_summary = Column(EncryptedText)    # ENCRYPTED - contains confidential summary
    ai_sentiment = Column(String(20))  # positive/neutral/negative
    ai_urgency = Column(String(20))  # urgent/normal/low
    ai_urgency_score = Column(Integer)  # 1-10
    
    # Full LLM response (ENCRYPTED - may include sensitive reasoning/details)
    ai_full_response = Column(EncryptedJSON, default=dict)  # Store the complete structured JSON output
    
    # Curated Sender Name (UNENCRYPTED - extracted via LLM during classification, better than raw from_name)
    curated_sender_name = Column(String(500))  # Curated sender name extracted from email body/attachments
    
    # Action Items & Replies
    # Note: Changed from JSON to Text for encryption (encrypted JSON stored as encrypted text)
    ai_action_items = Column(EncryptedJSON, default=list)  # ENCRYPTED - stored as TEXT
    needs_reply = Column(Boolean, default=False)
    awaiting_reply = Column(Boolean, default=False)
    reply_deadline = Column(DateTime)
    replied_at = Column(DateTime)
    
    # Relevance Scores (UNENCRYPTED - quantitative scores for filtering)
    relevance_score = Column(Integer)  # UNENCRYPTED - quantitative score
    relevance_reason = Column(EncryptedText)  # ENCRYPTED - may reference sensitive context
    prestige_score = Column(Integer)  # UNENCRYPTED - quantitative score
    prestige_reason = Column(EncryptedText)  # ENCRYPTED - may reference sensitive context
    
    # Application-Specific Scores (UNENCRYPTED - for filtering/sorting applications)
    scientific_excellence_score = Column(Integer)  # UNENCRYPTED - 1-10 academic credentials
    research_fit_score = Column(Integer)  # UNENCRYPTED - 1-10 research alignment (applications only)
    overall_recommendation_score = Column(Integer)  # UNENCRYPTED - 1-10 overall recommendation (applications only)
    applicant_name = Column(String(500))  # UNENCRYPTED - applicant name extracted during classification (applications only)
    applicant_email = Column(String(500))  # UNENCRYPTED - applicant email (may differ from from_address for forwarded emails)
    applicant_institution = Column(String(500))  # UNENCRYPTED - applicant institution (applications only)
    
    # Category-Specific Fields - Split into queryable vs. encrypted
    # Non-PII structured metadata (UNENCRYPTED - for querying tags, flags, scores)
    category_metadata = Column(JSON, default=dict)  # profile_tags, information_used, red_flags, technical_scores
    # PII and sensitive evaluation details (ENCRYPTED - names, accounts, reasoning, notes)
    category_specific_data = Column(EncryptedJSON, default=dict)  # Now used for sensitive data only
    
    # Two-Stage Classification Tracking (NEW - for analysis and debugging)
    two_stage_used = Column(Boolean, default=False)  # Was Stage 2 triggered?
    stage_1_model = Column(String(50))  # Model used for Stage 1 (e.g., "gpt-5-mini")
    stage_1_category = Column(String(50))  # Stage 1 category (may differ from final)
    stage_1_confidence = Column(Float)  # Stage 1 confidence
    stage_1_urgency_score = Column(Integer)  # Stage 1 urgency (for comparison)
    stage_1_recommendation_score = Column(Integer)  # Stage 1 rec score (for applications)
    stage_1_scientific_excellence_score = Column(Integer)  # Stage 1 sci score
    stage_2_model = Column(String(50))  # Model used for Stage 2 (e.g., "gpt-5.1")
    stage_2_reason = Column(String(200))  # Why Stage 2 was triggered
    # Note: Stage 2 scores are in the main ai_* fields (final result)
    
    # Followup Tracking
    is_followup = Column(Boolean, default=False)
    followup_to_email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'))
    followup_to_date = Column(DateTime)
    
    # Categorization
    is_cold_email = Column(Boolean, default=False)
    project_tags = Column(JSON, default=list)  # UNENCRYPTED - for filtering (not too sensitive)
    suggested_folder = Column(String(200))
    suggested_labels = Column(JSON, default=list)
    
    # User Annotations
    user_notes = Column(EncryptedText)     # ENCRYPTED - user's personal notes
    user_priority = Column(Integer)  # User override 1-5
    user_tags = Column(JSON, default=list)  # UNENCRYPTED - needed for filtering (handled, user-spam)
    
    # Lifecycle Status (NEW - track email journey)
    email_status = Column(String(20), default='active')  # 'active', 'archived', 'deleted', 'handled'
    status_updated_at = Column(DateTime)
    
    # Folder History Summary (NEW - for quick queries without joining history table)
    first_seen_folder = Column(String(200))  # Usually INBOX or Sent Items
    first_seen_at = Column(DateTime)
    current_folder_since = Column(DateTime)
    folder_move_count = Column(Integer, default=0)
    
    # Google Drive Integration (NEW - for exporting to Google Sheets/Drive)
    google_drive_folder_id = Column(String(200))  # Google Drive folder ID containing exported files
    google_drive_links = Column(JSON, default=dict)  # Links to email.txt, attachments, llm_response.json
    google_sheet_url = Column(String(500))  # Link to Google Sheet (if exported to Sheets)
    
    # Inquiry Handler (for #info emails - separate from applications)
    inquiry_types = Column(JSON, default=None)  # ["phd", "postdoc"] - types of positions being inquired about
    extracted_name = Column(String(255))  # Sender's full name extracted for greeting
    name_extraction_source = Column(String(50))  # "ai" or "fallback"
    inquiry_classification_source = Column(String(50))  # "tag" (from subject tags) or "ai"
    
    # Inquiry Draft Lifecycle
    draft_status = Column(String(20))  # "created", "sent", "skipped"
    draft_created_at = Column(DateTime)  # When the draft was created
    draft_message_id = Column(String(500))  # Message-ID of the draft we created in IMAP
    inquiry_message_id = Column(String(500))  # Message-ID of the original inquiry email (for sent detection)
    
    # Application Review System (NEW - for lab member review workflow)
    review_deadline = Column(DateTime)  # Deadline for reviews (e.g., 1 week from receipt for high-recommendation apps)
    application_status = Column(String(20), default='pending', server_default='pending')  # 'pending', 'under_review', 'decided', 'archived'
    application_status_updated_at = Column(DateTime)  # When application_status was last updated
    gdrive_archive_folder_id = Column(String(200))  # ID of archived Google Drive folder after decision
    gdrive_archive_path = Column(Text)  # Human-readable archive path: 'archive/phd/2024-01/john_smith'
    archived_at = Column(DateTime)  # When GDrive archiving was completed
    
    # User Action Signals (NEW - understand user behavior)
    user_archived = Column(Boolean, default=False)  # User moved to Archive
    user_deleted = Column(Boolean, default=False)   # User moved to Trash
    auto_filed = Column(Boolean, default=False)     # Rule/AI auto-organized
    
    # Importance Signals (NEW - derived from user actions)
    time_to_action_seconds = Column(Integer)  # How quickly user acted (first move from inbox)
    importance_score = Column(Integer)  # 0-10, derived from actions (trash=0, archive=5, organized=8)
    
    # Timestamps
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    email = relationship("Email", foreign_keys=[email_id], back_populates="email_metadata")
    
    __table_args__ = (
        Index('ix_metadata_vip_level', 'vip_level'),  # For VIP filtering
        Index('ix_metadata_needs_reply', 'needs_reply', 'reply_deadline'),  # For reply tracking
        Index('ix_metadata_urgency', 'ai_urgency', 'ai_urgency_score'),  # For urgent email queries
        Index('ix_metadata_category', 'ai_category', 'ai_subcategory'),  # For category filtering
        Index('ix_metadata_status', 'email_status'),  # For lifecycle filtering
        Index('ix_metadata_user_actions', 'user_archived', 'user_deleted'),  # User behavior analysis
        Index('ix_metadata_curated_sender_name', 'curated_sender_name'),  # For sender name search/filtering
        # Application Review System indexes
        Index('ix_metadata_application_status', 'application_status'),  # For application status filtering
        Index('ix_metadata_review_deadline', 'review_deadline'),  # For deadline queries
        Index('ix_metadata_applicant_name', 'applicant_name'),  # For name search (case-insensitive via LOWER() in queries)
        Index('ix_metadata_applicant_email', 'applicant_email'),  # For related emails lookup in forwarded applications
        Index('ix_metadata_pending_review', 'application_status', 'review_deadline'),  # Partial index for pending/under_review
        Index('ix_metadata_deadline_approaching', 'review_deadline'),  # Partial index for approaching deadlines
        # Inquiry Handler indexes
        Index('ix_metadata_draft_status', 'draft_status'),  # For pending draft queries
    )


class SenderHistory(Base):
    """
    Track sender patterns and statistics for smart categorization.
    Helps AI make better decisions based on past interactions.
    """
    __tablename__ = "sender_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_address = Column(String(500), unique=True, nullable=False, index=True)
    sender_name = Column(String(500))
    domain = Column(String(200), index=True)
    
    # Categorization
    sender_type = Column(String(50))  # colleague/client/newsletter/spam/unknown
    is_vip = Column(Boolean, default=False)
    vip_level = Column(String(20))  # urgent/high/medium
    
    # Statistics
    email_count = Column(Integer, default=0)
    first_seen = Column(DateTime, nullable=False)
    last_seen = Column(DateTime, nullable=False)
    
    # Interaction Patterns
    reply_count = Column(Integer, default=0)  # How many times you replied
    avg_reply_time_hours = Column(Float)  # Your typical response time
    always_replies = Column(Boolean, default=False)  # You always reply to this sender
    always_archives = Column(Boolean, default=False)  # You always archive
    
    # Cold Email Detection
    is_cold_sender = Column(Boolean, default=False)  # No prior interaction
    is_frequent = Column(Boolean, default=False)  # > 10 emails
    
    # Classification History
    typical_category = Column(String(50))  # Most common category
    typical_urgency = Column(String(20))  # Most common urgency
    
    # Notes (ENCRYPTED - personal annotations)
    notes = Column(EncryptedText)  # ENCRYPTED - user notes about sender
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_sender_domain_type', 'domain', 'sender_type'),
    )


class Classification(Base):
    """
    Classification audit log - tracks how emails were classified.
    Useful for improving prompts and understanding AI decisions.
    """
    __tablename__ = "classifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=False)
    
    # Classification Type
    classifier_type = Column(String(20), nullable=False)  # rule/ai/vip/manual
    classifier_name = Column(String(100))  # Which rule or AI model
    
    # Results
    category = Column(String(50), nullable=False)
    subcategory = Column(String(50))
    confidence = Column(Float)  # 0-1
    reasoning = Column(EncryptedText)  # ENCRYPTED - may quote email content
    
    # AI-specific
    model_used = Column(String(50))  # gpt-4, claude-3-opus, etc.
    prompt_version = Column(String(20))  # Track prompt iterations
    tokens_used = Column(Integer)  # API cost tracking
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    email = relationship("Email", back_populates="classifications")
    
    __table_args__ = (
        Index('ix_classifications_email_type', 'email_id', 'classifier_type'),
    )


class ReplyTracking(Base):
    """
    Track which emails need replies and response status.
    From Inbox-Zero feature.
    """
    __tablename__ = "reply_tracking"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), unique=True, nullable=False)
    
    # Reply Status
    needs_reply = Column(Boolean, default=False)
    awaiting_reply = Column(Boolean, default=False)  # You sent, waiting for response
    reply_deadline = Column(DateTime)
    
    # Tracking
    reminded_at = Column(DateTime)  # When user was reminded
    replied_at = Column(DateTime)  # When you replied
    reply_email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'))  # Your reply
    
    # AI Detection
    detected_by = Column(String(20))  # ai/rule/manual
    question_detected = Column(Boolean, default=False)
    action_requested = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReplyDraft(Base):
    """
    Generated draft replies for emails (Phase 3).
    Stores template-based and AI-generated drafts.
    """
    __tablename__ = "reply_drafts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=False)
    
    # Draft content
    subject = Column(Text, nullable=False)  # UNENCRYPTED - for draft search
    body = Column(EncryptedText, nullable=False)  # ENCRYPTED - draft content
    tone = Column(String(20))  # professional/friendly/formal/enthusiastic/cautious
    option_number = Column(Integer, default=1)  # Which variation (1, 2, 3)
    
    # Generation metadata
    generated_by = Column(String(20))  # template/ai/hybrid
    template_used = Column(String(100))  # Template key if used
    model_used = Column(String(50))  # gpt-4o-mini, claude-3-haiku, etc.
    confidence = Column(Float)  # 0-1, quality estimate
    reasoning = Column(EncryptedText)  # ENCRYPTED - may reference email content
    
    # Decision context
    decision = Column(String(20))  # accept/decline/maybe/acknowledge
    category = Column(String(50))  # Email category this is for
    
    # Status
    status = Column(String(20), default='draft')  # draft/edited/sent/discarded
    edited_by_user = Column(Boolean, default=False)
    sent_at = Column(DateTime)
    
    # IMAP integration
    imap_draft_uid = Column(String(100))  # UID in Drafts folder (if saved to IMAP)
    imap_message_id = Column(String(500))  # Message-ID if sent
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    email = relationship("Email", foreign_keys=[email_id])
    
    __table_args__ = (
        Index('ix_drafts_email_id', 'email_id'),
        Index('ix_drafts_status', 'status'),
        Index('ix_drafts_created_at', 'created_at'),
    )


class EmailEmbedding(Base):
    """
    Vector embeddings for semantic search (Phase 3).
    Uses pgvector for efficient similarity search at scale.
    """
    __tablename__ = "email_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), unique=True, nullable=False)
    
    # Vector embedding (3072 dimensions for text-embedding-3-large)
    # Using pgvector's Vector type for native PostgreSQL vector operations
    # UNENCRYPTED - required for vector similarity search (pgvector HNSW index)
    # Note: Encrypting would break semantic search completely (need native vector types)
    if HAS_PGVECTOR:
        embedding = Column(Vector(3072), nullable=False)  # UNENCRYPTED - for pgvector search
    else:
        # Fallback to JSON if pgvector not available (dev/testing)
        embedding = Column(JSON, nullable=False)  # UNENCRYPTED
    
    # Metadata
    embedding_model = Column(String(50), nullable=False)  # text-embedding-3-small, etc.
    content_hash = Column(String(64), nullable=False)  # SHA256 of content (for change detection)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    email = relationship("Email", foreign_keys=[email_id])
    
    __table_args__ = (
        Index('ix_embeddings_email_id', 'email_id'),
        # Vector index will be created via migration:
        # CREATE INDEX ON email_embeddings USING hnsw (embedding vector_cosine_ops);
    )


class APIUsage(Base):
    """
    Track individual OpenAI API calls for cost monitoring.
    
    Records every API call with tokens, cost, and context.
    Enables detailed cost analysis and debugging.
    """
    __tablename__ = 'api_usage'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Model & Task
    model = Column(String(50), nullable=False, index=True)  # gpt-4o, gpt-4o-mini, text-embedding-3-small
    task = Column(String(50), nullable=False, index=True)  # classification, embedding, search_query
    source = Column(String(20), nullable=False, index=True)  # cli, api
    
    # Token Usage
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    
    # Cost
    cost_usd = Column(Float, nullable=False)
    
    # Context (optional)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=True)
    context_data = Column(JSON, default={})  # Additional context (category, confidence, etc.)
    
    # Relationships
    email = relationship("Email", foreign_keys=[email_id])
    
    __table_args__ = (
        Index('ix_api_usage_date', 'timestamp'),
        Index('ix_api_usage_model_task', 'model', 'task'),
    )


class DailyAPIUsage(Base):
    """
    Daily aggregated API usage for cost reporting.
    
    Pre-aggregated data for fast daily/monthly cost queries.
    Updated automatically when APIUsage records are created.
    """
    __tablename__ = 'daily_api_usage'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Aggregation key
    usage_date = Column(Date, nullable=False, index=True)
    model = Column(String(50), nullable=False, index=True)
    task = Column(String(50), nullable=False, index=True)
    source = Column(String(20), nullable=False, index=True)
    
    # Aggregated metrics
    total_calls = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    total_cost_usd = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_daily_usage_date_model', 'usage_date', 'model', 'task', 'source', unique=True),
    )


class FolderSyncState(Base):
    """
    Track incremental sync state per IMAP folder AND account.
    Stores the last processed UID to enable efficient incremental processing.
    
    Multi-account: Each account+folder combination has independent sync state.
    """
    __tablename__ = "folder_sync_state"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id = Column(String(50), nullable=False, default="work", index=True)
    folder = Column(String(200), nullable=False, index=True)
    
    # Last processed UID (highest UID we've seen)
    last_processed_uid = Column(Integer, nullable=False, default=0)
    
    # Total emails processed from this folder
    total_processed = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    first_sync_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_sync_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Optional: track UIDVALIDITY to detect folder resets
    uid_validity = Column(Integer, nullable=True)
    
    __table_args__ = (
        Index('ix_folder_sync_state_account_folder', 'account_id', 'folder', unique=True),
    )


class EmailLocationHistory(Base):
    """
    Track email location movements over time (folder AND account tracking).
    
    Records every time an email moves between folders or accounts to understand:
    - User behavior (archive = handled, trash = unimportant)
    - Cross-account movements
    - Time to action (how quickly user responds/files)
    - Rule accuracy (do users move emails after auto-filing?)
    - Email importance signals
    
    NOTE: Renamed from EmailFolderHistory to EmailLocationHistory for multi-account support.
    """
    __tablename__ = "email_location_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=False)
    
    # Location Movement (account + folder)
    from_account_id = Column(String(50))  # Previous account (NULL if first seen)
    from_folder = Column(String(200))  # Previous folder (NULL if first seen)
    to_account_id = Column(String(50), nullable=False, default="work")  # New account
    to_folder = Column(String(200), nullable=False)  # New folder
    moved_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Movement Context
    moved_by = Column(String(20), nullable=False)  # 'rule', 'ai', 'user', 'system', 'cross_account_rule', 'ui'
    move_reason = Column(Text)  # Rule name, AI category, "User action", etc.
    
    # Cross-account tracking
    is_cross_account = Column(Boolean, default=False, nullable=False)  # Quick filter for cross-account moves
    
    # Derived Insights
    time_in_previous_location_seconds = Column(Integer)  # How long in previous location
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    email = relationship("Email", foreign_keys=[email_id])
    
    __table_args__ = (
        Index('ix_location_history_email_moved', 'email_id', 'moved_at'),  # Timeline queries
        Index('ix_location_history_folder', 'to_folder', 'moved_at'),  # Folder activity
        Index('ix_location_history_moved_by', 'moved_by', 'moved_at'),  # Action analysis
        Index('ix_location_history_cross_account', 'is_cross_account'),  # Cross-account filter
        Index('ix_location_history_accounts', 'from_account_id', 'to_account_id'),  # Account movement
    )


class CrossAccountMove(Base):
    """
    Audit trail and retry management for cross-account email moves.
    
    Tracks every cross-account move attempt, including failures and retries.
    Enables orphan detection and duplicate cleanup.
    """
    __tablename__ = "cross_account_moves"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=False)
    message_id = Column(String(500), nullable=False)
    
    # Move details
    from_account_id = Column(String(50), nullable=False)
    from_folder = Column(String(200), nullable=False)
    to_account_id = Column(String(50), nullable=False)
    to_folder = Column(String(200), nullable=False)
    
    # Method and status
    move_method = Column(String(20), nullable=False)  # 'imap_copy', 'download_upload', 'pending'
    status = Column(String(20), nullable=False)  # 'pending', 'in_progress', 'completed', 'failed', 'retrying'
    error_message = Column(Text)
    
    # Retry management
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)
    last_retry_at = Column(DateTime)
    next_retry_at = Column(DateTime)  # For scheduled retries
    
    # Timing
    initiated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    
    # Who initiated
    initiated_by = Column(String(20), nullable=False)  # 'rule', 'ai', 'manual', 'ui'
    rule_name = Column(String(200))
    
    # Duplicate detection
    duplicate_detected = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime)
    
    # Relationships
    email = relationship("Email", foreign_keys=[email_id])
    
    __table_args__ = (
        Index('ix_cross_moves_email', 'email_id'),
        Index('ix_cross_moves_status', 'status', 'initiated_at'),
        Index('ix_cross_moves_retry', 'status', 'next_retry_at'),  # For scheduled retry queries
        Index('ix_cross_moves_accounts', 'from_account_id', 'to_account_id'),
        Index('ix_cross_moves_duplicates', 'duplicate_detected'),  # For duplicate cleanup
    )


# ============================================================================
# Lab Application Review System Models
# ============================================================================

class LabMember(Base):
    """
    Lab members who can review applications.
    Linked to GSuite for authentication.
    """
    __tablename__ = "lab_members"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gsuite_id = Column(String(500), unique=True, nullable=True)  # GSuite user ID (nullable for manually created users)
    email = Column(String(500), unique=True, nullable=False)  # Lab member's email
    full_name = Column(String(500))
    role = Column(String(50), default='member')  # 'member', 'admin'
    can_review = Column(Boolean, default=False)  # Admin-controlled flag for review permission
    is_active = Column(Boolean, default=True)  # Account active status (for soft delete/suspension)
    avatar_url = Column(String(500))  # Google Directory API thumbnail photo URL
    last_login_at = Column(DateTime)  # Track last successful login
    failed_login_attempts = Column(Integer, default=0)  # Track failed attempts for security monitoring
    locked_until = Column(DateTime)  # Account lockout after too many failed attempts
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    reviews = relationship("ApplicationReview", back_populates="lab_member", cascade="all, delete-orphan")
    decisions = relationship("ApplicationDecision", back_populates="admin")
    private_notes = relationship("ApplicationPrivateNote", back_populates="lab_member", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    __table_args__ = (
        Index('ix_lab_members_gsuite_id', 'gsuite_id'),
        Index('ix_lab_members_email', 'email'),
        Index('ix_lab_members_role', 'role'),
        Index('ix_lab_members_is_active', 'is_active'),
    )


class ApplicationReview(Base):
    """
    Individual reviews (ratings and comments) by lab members for each application.
    """
    __tablename__ = "application_reviews"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id', ondelete='CASCADE'), nullable=False)
    lab_member_id = Column(UUID(as_uuid=True), ForeignKey('lab_members.id', ondelete='CASCADE'), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    comment = Column(EncryptedText)  # Encrypted comment (may contain sensitive info)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    email = relationship("Email", foreign_keys=[email_id])
    lab_member = relationship("LabMember", back_populates="reviews")
    
    __table_args__ = (
        Index('ix_app_reviews_email_member', 'email_id', 'lab_member_id', unique=True),  # One review per member per application
        Index('ix_app_reviews_rating', 'rating'),
        Index('ix_app_reviews_created_at', 'created_at'),
    )


class ApplicationDecision(Base):
    """
    Final decisions made by admins on applications.
    """
    __tablename__ = "application_decisions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id', ondelete='CASCADE'), unique=True, nullable=False)  # One decision per application
    admin_id = Column(UUID(as_uuid=True), ForeignKey('lab_members.id', ondelete='SET NULL'), nullable=False)  # Admin who made the decision
    decision = Column(String(50), nullable=False)  # 'accept', 'reject', 'interview', 'request_more_info', 'refer_to_direct_doctorate', 'delete'
    notes = Column(EncryptedText)  # Encrypted admin notes
    decided_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    email = relationship("Email", foreign_keys=[email_id])
    admin = relationship("LabMember", back_populates="decisions")
    
    __table_args__ = (
        Index('ix_app_decisions_email', 'email_id', unique=True),
        Index('ix_app_decisions_decision', 'decision'),
        Index('ix_app_decisions_decided_at', 'decided_at'),
    )


class ApplicationPrivateNote(Base):
    """
    Private notes by lab members for applications.
    Each user can have their own private notes per application (not visible to others).
    """
    __tablename__ = "application_private_notes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id', ondelete='CASCADE'), nullable=False)
    lab_member_id = Column(UUID(as_uuid=True), ForeignKey('lab_members.id', ondelete='CASCADE'), nullable=False)
    notes = Column(EncryptedText)  # Encrypted private notes (not visible to other users)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    email = relationship("Email", foreign_keys=[email_id])
    lab_member = relationship("LabMember", back_populates="private_notes")
    
    __table_args__ = (
        Index('ix_app_private_notes_email_member', 'email_id', 'lab_member_id', unique=True),  # One note per member per application
        Index('ix_app_private_notes_created_at', 'created_at'),
    )


class NotificationLog(Base):
    """
    Tracks notifications sent to lab members to prevent duplicates.
    """
    __tablename__ = "notification_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lab_member_id = Column(UUID(as_uuid=True), ForeignKey('lab_members.id', ondelete='CASCADE'), nullable=False)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id', ondelete='CASCADE'), nullable=False)
    notification_type = Column(String(50), nullable=False)  # 'new_application', 'deadline_approaching', 'deadline_passed'
    sent_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    read_at = Column(DateTime)  # When notification was marked as read (null = unread)
    
    # Relationships
    lab_member = relationship("LabMember")
    email = relationship("Email", foreign_keys=[email_id])
    
    __table_args__ = (
        Index('ix_notif_log_member_email_type', 'lab_member_id', 'email_id', 'notification_type', unique=True),  # Prevent duplicate notifications
        Index('ix_notif_log_sent_at', 'sent_at'),
        Index('ix_notif_log_type', 'notification_type'),
    )


class SystemSettings(Base):
    """
    Configurable system settings (thresholds, rate limits, etc.).
    """
    __tablename__ = "system_settings"
    
    key = Column(String(100), primary_key=True)
    value = Column(JSON, nullable=False)  # JSON format: {"value": <actual_value>, "type": "integer|string|boolean|float"}
    description = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(UUID(as_uuid=True), ForeignKey('lab_members.id', ondelete='SET NULL'))
    
    # Relationships
    updater = relationship("LabMember")


class SecurityLog(Base):
    """
    Tracks security events for monitoring and auditing.
    """
    __tablename__ = "security_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('lab_members.id', ondelete='SET NULL'))
    event_type = Column(String(50), nullable=False)  # 'failed_auth', 'rate_limit', 'unauthorized_access', 'admin_action'
    endpoint = Column(String(200))
    ip_address = Column(String(50))  # Using String instead of INET for compatibility
    user_agent = Column(Text)
    details = Column(JSON)  # Additional context (e.g., limit exceeded, action taken)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("LabMember")
    
    __table_args__ = (
        Index('ix_security_log_user_id', 'user_id'),
        Index('ix_security_log_event_type', 'event_type'),
        Index('ix_security_log_created_at', 'created_at'),
        Index('ix_security_log_endpoint', 'endpoint'),
    )


class AuditLog(Base):
    """
    Tracks user actions on applications for audit purposes (view, comment, decision).
    """
    __tablename__ = "audit_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('lab_members.id', ondelete='CASCADE'), nullable=False)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id', ondelete='CASCADE'), nullable=False)
    action_type = Column(String(50), nullable=False)  # 'view', 'review_submit', 'review_update', 'review_delete', 'decision_make', 'decision_update'
    action_details = Column(JSON)  # Additional context: rating (for reviews), decision type (for decisions), comment length, etc. Never stores full PII
    ip_address = Column(String(50))  # Using String instead of INET for compatibility
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("LabMember", back_populates="audit_logs")
    email = relationship("Email", foreign_keys=[email_id])
    
    __table_args__ = (
        Index('ix_audit_log_user_id', 'user_id'),
        Index('ix_audit_log_email_id', 'email_id'),
        Index('ix_audit_log_action_type', 'action_type'),
        Index('ix_audit_log_created_at', 'created_at'),
    )


class JWTBlacklist(Base):
    """
    Tracks invalidated JWT tokens for secure logout functionality.
    """
    __tablename__ = "jwt_blacklist"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_jti = Column(String(500), unique=True, nullable=False)  # JWT ID (jti claim)
    user_id = Column(UUID(as_uuid=True), ForeignKey('lab_members.id', ondelete='CASCADE'))
    invalidated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)  # When the token naturally expires (for cleanup)
    
    # Relationships
    user = relationship("LabMember")
    
    __table_args__ = (
        Index('ix_jwt_blacklist_token_jti', 'token_jti', unique=True),
        Index('ix_jwt_blacklist_expires_at', 'expires_at'),
    )


class ApplicationCollection(Base):
    """
    User-created collections for organizing applications.
    Examples: "Shortlist for interview", "Discuss in meeting", "Project X candidates"
    """
    __tablename__ = "application_collections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    items = relationship("ApplicationCollectionItem", back_populates="collection", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_application_collections_name', 'name', unique=True),
    )


class ApplicationCollectionItem(Base):
    """
    Many-to-many relationship between collections and applications.
    An application can be in multiple collections.
    """
    __tablename__ = "application_collection_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey('application_collections.id', ondelete='CASCADE'), nullable=False)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id', ondelete='CASCADE'), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    collection = relationship("ApplicationCollection", back_populates="items")
    email = relationship("Email", foreign_keys=[email_id])
    
    __table_args__ = (
        Index('ix_collection_items_collection_id', 'collection_id'),
        Index('ix_collection_items_email_id', 'email_id'),
        Index('uq_collection_item', 'collection_id', 'email_id', unique=True),
    )


