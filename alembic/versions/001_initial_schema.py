"""Initial schema - creates all tables and indexes for mail-done.

Revision ID: 001_initial
Revises:
Create Date: 2026-01-30

This migration creates the complete database schema from scratch.
For existing databases, use `alembic stamp head` instead of running this.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables and indexes."""

    # ==========================================================================
    # Core Tables
    # ==========================================================================

    # emails - Core email records
    op.create_table('emails',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('message_id', sa.String(500), nullable=False),
        sa.Column('uid', sa.String(100), nullable=False),
        sa.Column('folder', sa.String(200), nullable=False, server_default='INBOX'),
        sa.Column('account_id', sa.String(50), nullable=False, server_default='work'),
        sa.Column('original_account_id', sa.String(50), nullable=True),
        sa.Column('moved_from_email_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('from_address', sa.String(500), nullable=False),
        sa.Column('from_name', sa.String(500), nullable=True),
        sa.Column('to_addresses', postgresql.JSON(), nullable=False),
        sa.Column('cc_addresses', postgresql.JSON(), nullable=True),
        sa.Column('subject', sa.Text(), nullable=False),
        sa.Column('date', sa.DateTime(), nullable=False),
        sa.Column('body_markdown', sa.Text(), nullable=True),  # Encrypted
        sa.Column('body_text', sa.Text(), nullable=True),  # Encrypted
        sa.Column('body_html', sa.Text(), nullable=True),  # Encrypted
        sa.Column('has_attachments', sa.Boolean(), server_default='false'),
        sa.Column('attachment_count', sa.Integer(), server_default='0'),
        sa.Column('attachment_info', postgresql.JSON(), nullable=True),
        sa.Column('raw_headers', postgresql.JSON(), nullable=True),
        sa.Column('thread_id', sa.String(500), nullable=True),
        sa.Column('references', sa.Text(), nullable=True),
        sa.Column('in_reply_to', sa.String(500), nullable=True),
        sa.Column('is_seen', sa.Boolean(), server_default='false'),
        sa.Column('is_flagged', sa.Boolean(), server_default='false'),
        sa.Column('is_answered', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_emails_message_id', 'emails', ['message_id'])
    op.create_index('ix_emails_account_id', 'emails', ['account_id'])
    op.create_index('ix_emails_from_address', 'emails', ['from_address'])
    op.create_index('ix_emails_date', 'emails', ['date'])
    op.create_index('ix_emails_from_date', 'emails', ['from_address', 'date'])
    op.create_index('ix_emails_folder_date', 'emails', ['folder', 'date'])
    op.create_index('ix_emails_thread_id', 'emails', ['thread_id'])
    op.create_index('ix_emails_is_seen_date', 'emails', ['is_seen', 'date'])
    op.create_index('ix_emails_account_message_id', 'emails', ['account_id', 'message_id'], unique=True)
    # Performance indexes
    op.create_index('idx_emails_account_date_id', 'emails', ['account_id', sa.text('date DESC'), 'id'])
    op.create_index('idx_emails_account_sender_date', 'emails', ['account_id', 'from_address', sa.text('date DESC')])
    # Trigram index for subject search (requires pg_trgm extension)
    op.execute("CREATE INDEX idx_emails_subject_trgm ON emails USING gin (subject gin_trgm_ops)")

    # email_metadata - Rich metadata for emails
    op.create_table('email_metadata',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('vip_level', sa.String(20), nullable=True),
        sa.Column('intended_color', sa.Integer(), nullable=True),
        sa.Column('priority_score', sa.Integer(), nullable=True),
        sa.Column('ai_category', sa.String(50), nullable=True),
        sa.Column('ai_subcategory', sa.String(50), nullable=True),
        sa.Column('ai_confidence', sa.Float(), nullable=True),
        sa.Column('ai_reasoning', sa.Text(), nullable=True),  # Encrypted
        sa.Column('ai_summary', sa.Text(), nullable=True),  # Encrypted
        sa.Column('ai_sentiment', sa.String(20), nullable=True),
        sa.Column('ai_urgency', sa.String(20), nullable=True),
        sa.Column('ai_urgency_score', sa.Integer(), nullable=True),
        sa.Column('ai_full_response', sa.Text(), nullable=True),  # Encrypted JSON
        sa.Column('curated_sender_name', sa.String(500), nullable=True),
        sa.Column('ai_action_items', sa.Text(), nullable=True),  # Encrypted JSON
        sa.Column('needs_reply', sa.Boolean(), server_default='false'),
        sa.Column('awaiting_reply', sa.Boolean(), server_default='false'),
        sa.Column('reply_deadline', sa.DateTime(), nullable=True),
        sa.Column('replied_at', sa.DateTime(), nullable=True),
        sa.Column('relevance_score', sa.Integer(), nullable=True),
        sa.Column('relevance_reason', sa.Text(), nullable=True),  # Encrypted
        sa.Column('prestige_score', sa.Integer(), nullable=True),
        sa.Column('prestige_reason', sa.Text(), nullable=True),  # Encrypted
        sa.Column('scientific_excellence_score', sa.Integer(), nullable=True),
        sa.Column('research_fit_score', sa.Integer(), nullable=True),
        sa.Column('overall_recommendation_score', sa.Integer(), nullable=True),
        sa.Column('applicant_name', sa.String(500), nullable=True),
        sa.Column('applicant_email', sa.String(500), nullable=True),
        sa.Column('applicant_institution', sa.String(500), nullable=True),
        sa.Column('category_metadata', postgresql.JSON(), server_default='{}'),
        sa.Column('category_specific_data', sa.Text(), nullable=True),  # Encrypted JSON
        sa.Column('two_stage_used', sa.Boolean(), server_default='false'),
        sa.Column('stage_1_model', sa.String(50), nullable=True),
        sa.Column('stage_1_category', sa.String(50), nullable=True),
        sa.Column('stage_1_confidence', sa.Float(), nullable=True),
        sa.Column('stage_1_urgency_score', sa.Integer(), nullable=True),
        sa.Column('stage_1_recommendation_score', sa.Integer(), nullable=True),
        sa.Column('stage_1_scientific_excellence_score', sa.Integer(), nullable=True),
        sa.Column('stage_2_model', sa.String(50), nullable=True),
        sa.Column('stage_2_reason', sa.String(200), nullable=True),
        sa.Column('is_followup', sa.Boolean(), server_default='false'),
        sa.Column('followup_to_email_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('followup_to_date', sa.DateTime(), nullable=True),
        sa.Column('is_cold_email', sa.Boolean(), server_default='false'),
        sa.Column('project_tags', postgresql.JSON(), nullable=True),
        sa.Column('suggested_folder', sa.String(200), nullable=True),
        sa.Column('suggested_labels', postgresql.JSON(), nullable=True),
        sa.Column('user_notes', sa.Text(), nullable=True),  # Encrypted
        sa.Column('user_priority', sa.Integer(), nullable=True),
        sa.Column('user_tags', postgresql.JSON(), nullable=True),
        sa.Column('email_status', sa.String(20), server_default='active'),
        sa.Column('status_updated_at', sa.DateTime(), nullable=True),
        sa.Column('first_seen_folder', sa.String(200), nullable=True),
        sa.Column('first_seen_at', sa.DateTime(), nullable=True),
        sa.Column('current_folder_since', sa.DateTime(), nullable=True),
        sa.Column('folder_move_count', sa.Integer(), server_default='0'),
        sa.Column('google_drive_folder_id', sa.String(200), nullable=True),
        sa.Column('google_drive_links', postgresql.JSON(), server_default='{}'),
        sa.Column('google_sheet_url', sa.String(500), nullable=True),
        sa.Column('inquiry_types', postgresql.JSON(), nullable=True),
        sa.Column('extracted_name', sa.String(255), nullable=True),
        sa.Column('name_extraction_source', sa.String(50), nullable=True),
        sa.Column('inquiry_classification_source', sa.String(50), nullable=True),
        sa.Column('draft_status', sa.String(20), nullable=True),
        sa.Column('draft_created_at', sa.DateTime(), nullable=True),
        sa.Column('draft_message_id', sa.String(500), nullable=True),
        sa.Column('inquiry_message_id', sa.String(500), nullable=True),
        sa.Column('review_deadline', sa.DateTime(), nullable=True),
        sa.Column('application_status', sa.String(20), server_default='pending'),
        sa.Column('application_status_updated_at', sa.DateTime(), nullable=True),
        sa.Column('gdrive_archive_folder_id', sa.String(200), nullable=True),
        sa.Column('gdrive_archive_path', sa.Text(), nullable=True),
        sa.Column('archived_at', sa.DateTime(), nullable=True),
        sa.Column('user_archived', sa.Boolean(), server_default='false'),
        sa.Column('user_deleted', sa.Boolean(), server_default='false'),
        sa.Column('auto_filed', sa.Boolean(), server_default='false'),
        sa.Column('time_to_action_seconds', sa.Integer(), nullable=True),
        sa.Column('importance_score', sa.Integer(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['followup_to_email_id'], ['emails.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email_id')
    )
    op.create_index('ix_metadata_vip_level', 'email_metadata', ['vip_level'])
    op.create_index('ix_metadata_needs_reply', 'email_metadata', ['needs_reply', 'reply_deadline'])
    op.create_index('ix_metadata_urgency', 'email_metadata', ['ai_urgency', 'ai_urgency_score'])
    op.create_index('ix_metadata_category', 'email_metadata', ['ai_category', 'ai_subcategory'])
    op.create_index('ix_metadata_status', 'email_metadata', ['email_status'])
    op.create_index('ix_metadata_user_actions', 'email_metadata', ['user_archived', 'user_deleted'])
    op.create_index('ix_metadata_curated_sender_name', 'email_metadata', ['curated_sender_name'])
    op.create_index('ix_metadata_application_status', 'email_metadata', ['application_status'])
    op.create_index('ix_metadata_review_deadline', 'email_metadata', ['review_deadline'])
    op.create_index('ix_metadata_applicant_name', 'email_metadata', ['applicant_name'])
    op.create_index('ix_metadata_applicant_email', 'email_metadata', ['applicant_email'])
    op.create_index('ix_metadata_draft_status', 'email_metadata', ['draft_status'])
    # Partial indexes for application review
    op.execute("CREATE INDEX idx_metadata_pending_review ON email_metadata (application_status, review_deadline) WHERE application_status IN ('pending', 'under_review')")
    op.execute("CREATE INDEX idx_metadata_deadline_approaching ON email_metadata (review_deadline) WHERE review_deadline IS NOT NULL")
    # Case-insensitive applicant name search
    op.execute("CREATE INDEX ix_metadata_applicant_name_lower ON email_metadata (lower(applicant_name))")

    # sender_history - Track sender patterns
    op.create_table('sender_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_address', sa.String(500), nullable=False),
        sa.Column('sender_name', sa.String(500), nullable=True),
        sa.Column('domain', sa.String(200), nullable=True),
        sa.Column('sender_type', sa.String(50), nullable=True),
        sa.Column('is_vip', sa.Boolean(), server_default='false'),
        sa.Column('vip_level', sa.String(20), nullable=True),
        sa.Column('email_count', sa.Integer(), server_default='0'),
        sa.Column('first_seen', sa.DateTime(), nullable=False),
        sa.Column('last_seen', sa.DateTime(), nullable=False),
        sa.Column('reply_count', sa.Integer(), server_default='0'),
        sa.Column('avg_reply_time_hours', sa.Float(), nullable=True),
        sa.Column('always_replies', sa.Boolean(), server_default='false'),
        sa.Column('always_archives', sa.Boolean(), server_default='false'),
        sa.Column('is_cold_sender', sa.Boolean(), server_default='false'),
        sa.Column('is_frequent', sa.Boolean(), server_default='false'),
        sa.Column('typical_category', sa.String(50), nullable=True),
        sa.Column('typical_urgency', sa.String(20), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),  # Encrypted
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email_address')
    )
    op.create_index('ix_sender_history_email_address', 'sender_history', ['email_address'])
    op.create_index('ix_sender_history_domain', 'sender_history', ['domain'])
    op.create_index('ix_sender_domain_type', 'sender_history', ['domain', 'sender_type'])

    # classifications - Audit log for classifications
    op.create_table('classifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('classifier_type', sa.String(20), nullable=False),
        sa.Column('classifier_name', sa.String(100), nullable=True),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('subcategory', sa.String(50), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('reasoning', sa.Text(), nullable=True),  # Encrypted
        sa.Column('model_used', sa.String(50), nullable=True),
        sa.Column('prompt_version', sa.String(20), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_classifications_email_type', 'classifications', ['email_id', 'classifier_type'])

    # reply_tracking - Track reply status
    op.create_table('reply_tracking',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('needs_reply', sa.Boolean(), server_default='false'),
        sa.Column('awaiting_reply', sa.Boolean(), server_default='false'),
        sa.Column('reply_deadline', sa.DateTime(), nullable=True),
        sa.Column('reminded_at', sa.DateTime(), nullable=True),
        sa.Column('replied_at', sa.DateTime(), nullable=True),
        sa.Column('reply_email_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('detected_by', sa.String(20), nullable=True),
        sa.Column('question_detected', sa.Boolean(), server_default='false'),
        sa.Column('action_requested', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['reply_email_id'], ['emails.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email_id')
    )

    # reply_drafts - Generated draft replies
    op.create_table('reply_drafts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('subject', sa.Text(), nullable=False),
        sa.Column('body', sa.Text(), nullable=False),  # Encrypted
        sa.Column('tone', sa.String(20), nullable=True),
        sa.Column('option_number', sa.Integer(), server_default='1'),
        sa.Column('generated_by', sa.String(20), nullable=True),
        sa.Column('template_used', sa.String(100), nullable=True),
        sa.Column('model_used', sa.String(50), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('reasoning', sa.Text(), nullable=True),  # Encrypted
        sa.Column('decision', sa.String(20), nullable=True),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('status', sa.String(20), server_default='draft'),
        sa.Column('edited_by_user', sa.Boolean(), server_default='false'),
        sa.Column('sent_at', sa.DateTime(), nullable=True),
        sa.Column('imap_draft_uid', sa.String(100), nullable=True),
        sa.Column('imap_message_id', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_drafts_email_id', 'reply_drafts', ['email_id'])
    op.create_index('ix_drafts_status', 'reply_drafts', ['status'])
    op.create_index('ix_drafts_created_at', 'reply_drafts', ['created_at'])

    # email_embeddings - Vector embeddings for semantic search
    op.create_table('email_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('embedding_model', sa.String(50), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email_id')
    )
    op.create_index('ix_embeddings_email_id', 'email_embeddings', ['email_id'])
    # Add vector column using raw SQL (pgvector)
    op.execute("ALTER TABLE email_embeddings ADD COLUMN embedding vector(3072) NOT NULL")
    # Create vector index (DiskANN for large datasets, HNSW as fallback)
    op.execute("CREATE INDEX email_embeddings_diskann_idx ON email_embeddings USING diskann (embedding)")

    # ==========================================================================
    # API Usage & Tracking Tables
    # ==========================================================================

    # api_usage - Track individual API calls
    op.create_table('api_usage',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('model', sa.String(50), nullable=False),
        sa.Column('task', sa.String(50), nullable=False),
        sa.Column('source', sa.String(20), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=False),
        sa.Column('completion_tokens', sa.Integer(), nullable=False),
        sa.Column('total_tokens', sa.Integer(), nullable=False),
        sa.Column('cost_usd', sa.Float(), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('context_data', postgresql.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_api_usage_timestamp', 'api_usage', ['timestamp'])
    op.create_index('ix_api_usage_model', 'api_usage', ['model'])
    op.create_index('ix_api_usage_task', 'api_usage', ['task'])
    op.create_index('ix_api_usage_source', 'api_usage', ['source'])
    op.create_index('ix_api_usage_date', 'api_usage', ['timestamp'])
    op.create_index('ix_api_usage_model_task', 'api_usage', ['model', 'task'])

    # daily_api_usage - Aggregated daily usage
    op.create_table('daily_api_usage',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('usage_date', sa.Date(), nullable=False),
        sa.Column('model', sa.String(50), nullable=False),
        sa.Column('task', sa.String(50), nullable=False),
        sa.Column('source', sa.String(20), nullable=False),
        sa.Column('total_calls', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_cost_usd', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_daily_api_usage_usage_date', 'daily_api_usage', ['usage_date'])
    op.create_index('ix_daily_api_usage_model', 'daily_api_usage', ['model'])
    op.create_index('ix_daily_api_usage_task', 'daily_api_usage', ['task'])
    op.create_index('ix_daily_api_usage_source', 'daily_api_usage', ['source'])
    op.create_index('ix_daily_usage_date_model', 'daily_api_usage', ['usage_date', 'model', 'task', 'source'], unique=True)

    # ==========================================================================
    # Folder & Location Tracking Tables
    # ==========================================================================

    # folder_sync_state - Track incremental sync state
    op.create_table('folder_sync_state',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('account_id', sa.String(50), nullable=False, server_default='work'),
        sa.Column('folder', sa.String(200), nullable=False),
        sa.Column('last_processed_uid', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_processed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('first_sync_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_sync_at', sa.DateTime(), nullable=True),
        sa.Column('uid_validity', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_folder_sync_state_account_id', 'folder_sync_state', ['account_id'])
    op.create_index('ix_folder_sync_state_folder', 'folder_sync_state', ['folder'])
    op.create_index('ix_folder_sync_state_account_folder', 'folder_sync_state', ['account_id', 'folder'], unique=True)

    # email_location_history - Track email movements
    op.create_table('email_location_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('from_account_id', sa.String(50), nullable=True),
        sa.Column('from_folder', sa.String(200), nullable=True),
        sa.Column('to_account_id', sa.String(50), nullable=False, server_default='work'),
        sa.Column('to_folder', sa.String(200), nullable=False),
        sa.Column('moved_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('moved_by', sa.String(20), nullable=False),
        sa.Column('move_reason', sa.Text(), nullable=True),
        sa.Column('is_cross_account', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('time_in_previous_location_seconds', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_email_location_history_moved_at', 'email_location_history', ['moved_at'])
    op.create_index('ix_location_history_email_moved', 'email_location_history', ['email_id', 'moved_at'])
    op.create_index('ix_location_history_folder', 'email_location_history', ['to_folder', 'moved_at'])
    op.create_index('ix_location_history_moved_by', 'email_location_history', ['moved_by', 'moved_at'])
    op.create_index('ix_location_history_cross_account', 'email_location_history', ['is_cross_account'])
    op.create_index('ix_location_history_accounts', 'email_location_history', ['from_account_id', 'to_account_id'])

    # cross_account_moves - Audit trail for cross-account moves
    op.create_table('cross_account_moves',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('message_id', sa.String(500), nullable=False),
        sa.Column('from_account_id', sa.String(50), nullable=False),
        sa.Column('from_folder', sa.String(200), nullable=False),
        sa.Column('to_account_id', sa.String(50), nullable=False),
        sa.Column('to_folder', sa.String(200), nullable=False),
        sa.Column('move_method', sa.String(20), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_retries', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('last_retry_at', sa.DateTime(), nullable=True),
        sa.Column('next_retry_at', sa.DateTime(), nullable=True),
        sa.Column('initiated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('initiated_by', sa.String(20), nullable=False),
        sa.Column('rule_name', sa.String(200), nullable=True),
        sa.Column('duplicate_detected', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_cross_moves_email', 'cross_account_moves', ['email_id'])
    op.create_index('ix_cross_moves_status', 'cross_account_moves', ['status', 'initiated_at'])
    op.create_index('ix_cross_moves_retry', 'cross_account_moves', ['status', 'next_retry_at'])
    op.create_index('ix_cross_moves_accounts', 'cross_account_moves', ['from_account_id', 'to_account_id'])
    op.create_index('ix_cross_moves_duplicates', 'cross_account_moves', ['duplicate_detected'])

    # ==========================================================================
    # Lab Application Review System Tables
    # ==========================================================================

    # lab_members - Lab members who can review applications
    op.create_table('lab_members',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('gsuite_id', sa.String(500), nullable=True),
        sa.Column('email', sa.String(500), nullable=False),
        sa.Column('full_name', sa.String(500), nullable=True),
        sa.Column('role', sa.String(50), server_default='member'),
        sa.Column('can_review', sa.Boolean(), server_default='false'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('avatar_url', sa.String(500), nullable=True),
        sa.Column('last_login_at', sa.DateTime(), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), server_default='0'),
        sa.Column('locked_until', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('gsuite_id')
    )
    op.create_index('ix_lab_members_gsuite_id', 'lab_members', ['gsuite_id'])
    op.create_index('ix_lab_members_email', 'lab_members', ['email'])
    op.create_index('ix_lab_members_role', 'lab_members', ['role'])
    op.create_index('ix_lab_members_is_active', 'lab_members', ['is_active'])

    # application_reviews - Individual reviews
    op.create_table('application_reviews',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('lab_member_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('comment', sa.Text(), nullable=True),  # Encrypted
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['lab_member_id'], ['lab_members.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_app_reviews_email_member', 'application_reviews', ['email_id', 'lab_member_id'], unique=True)
    op.create_index('ix_app_reviews_rating', 'application_reviews', ['rating'])
    op.create_index('ix_app_reviews_created_at', 'application_reviews', ['created_at'])

    # application_decisions - Final decisions by admins
    op.create_table('application_decisions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('admin_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('decision', sa.String(50), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),  # Encrypted
        sa.Column('decided_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['admin_id'], ['lab_members.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email_id')
    )
    op.create_index('ix_app_decisions_email', 'application_decisions', ['email_id'], unique=True)
    op.create_index('ix_app_decisions_decision', 'application_decisions', ['decision'])
    op.create_index('ix_app_decisions_decided_at', 'application_decisions', ['decided_at'])

    # application_private_notes - Private notes per user per application
    op.create_table('application_private_notes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('lab_member_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),  # Encrypted
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['lab_member_id'], ['lab_members.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_app_private_notes_email_member', 'application_private_notes', ['email_id', 'lab_member_id'], unique=True)
    op.create_index('ix_app_private_notes_created_at', 'application_private_notes', ['created_at'])

    # notification_log - Track sent notifications
    op.create_table('notification_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('lab_member_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('notification_type', sa.String(50), nullable=False),
        sa.Column('sent_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('read_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['lab_member_id'], ['lab_members.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_notif_log_member_email_type', 'notification_log', ['lab_member_id', 'email_id', 'notification_type'], unique=True)
    op.create_index('ix_notif_log_sent_at', 'notification_log', ['sent_at'])
    op.create_index('ix_notif_log_type', 'notification_log', ['notification_type'])
    op.create_index('ix_notif_log_read_at', 'notification_log', ['read_at'])

    # ==========================================================================
    # Security & System Tables
    # ==========================================================================

    # system_settings - Configurable system settings
    op.create_table('system_settings',
        sa.Column('key', sa.String(100), nullable=False),
        sa.Column('value', postgresql.JSON(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['updated_by'], ['lab_members.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('key')
    )

    # security_log - Track security events
    op.create_table('security_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('endpoint', sa.String(200), nullable=True),
        sa.Column('ip_address', sa.String(50), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('details', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['user_id'], ['lab_members.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_security_log_user_id', 'security_log', ['user_id'])
    op.create_index('ix_security_log_event_type', 'security_log', ['event_type'])
    op.create_index('ix_security_log_created_at', 'security_log', ['created_at'])
    op.create_index('ix_security_log_endpoint', 'security_log', ['endpoint'])

    # audit_log - Track user actions
    op.create_table('audit_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('action_type', sa.String(50), nullable=False),
        sa.Column('action_details', postgresql.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(50), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['lab_members.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audit_log_user_id', 'audit_log', ['user_id'])
    op.create_index('ix_audit_log_email_id', 'audit_log', ['email_id'])
    op.create_index('ix_audit_log_action_type', 'audit_log', ['action_type'])
    op.create_index('ix_audit_log_created_at', 'audit_log', ['created_at'])

    # jwt_blacklist - Track invalidated JWTs
    op.create_table('jwt_blacklist',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token_jti', sa.String(500), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('invalidated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['lab_members.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_jti')
    )
    op.create_index('ix_jwt_blacklist_token_jti', 'jwt_blacklist', ['token_jti'], unique=True)
    op.create_index('ix_jwt_blacklist_expires_at', 'jwt_blacklist', ['expires_at'])

    # ==========================================================================
    # Collection Tables
    # ==========================================================================

    # application_collections - User-created collections
    op.create_table('application_collections',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('ix_application_collections_name', 'application_collections', ['name'], unique=True)

    # application_collection_items - Many-to-many for collections
    op.create_table('application_collection_items',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('collection_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('added_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['collection_id'], ['application_collections.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_collection_items_collection_id', 'application_collection_items', ['collection_id'])
    op.create_index('ix_collection_items_email_id', 'application_collection_items', ['email_id'])
    op.create_index('uq_collection_item', 'application_collection_items', ['collection_id', 'email_id'], unique=True)


def downgrade() -> None:
    """Drop all tables."""
    # Drop in reverse order due to foreign key constraints
    op.drop_table('application_collection_items')
    op.drop_table('application_collections')
    op.drop_table('jwt_blacklist')
    op.drop_table('audit_log')
    op.drop_table('security_log')
    op.drop_table('system_settings')
    op.drop_table('notification_log')
    op.drop_table('application_private_notes')
    op.drop_table('application_decisions')
    op.drop_table('application_reviews')
    op.drop_table('lab_members')
    op.drop_table('cross_account_moves')
    op.drop_table('email_location_history')
    op.drop_table('folder_sync_state')
    op.drop_table('daily_api_usage')
    op.drop_table('api_usage')
    op.drop_table('email_embeddings')
    op.drop_table('reply_drafts')
    op.drop_table('reply_tracking')
    op.drop_table('classifications')
    op.drop_table('sender_history')
    op.drop_table('email_metadata')
    op.drop_table('emails')
