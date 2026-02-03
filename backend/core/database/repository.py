"""
Database Repository - High-level database operations for email processing.

Provides simple interface to store and retrieve emails, metadata, and classifications.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_
from sqlalchemy.exc import IntegrityError, OperationalError
import logging
import json

from .models import Email, EmailMetadata, SenderHistory, Classification, ReplyTracking, FolderSyncState, EmailLocationHistory, CrossAccountMove
from backend.core.email.models import ProcessedEmail
from backend.core.ai.classifier import AIClassificationResult

logger = logging.getLogger(__name__)


def sanitize_for_postgres(text: Optional[str], field_name: str = "text", max_length: Optional[int] = None) -> Optional[str]:
    """
    Remove NUL bytes and other problematic characters for PostgreSQL.
    
    PostgreSQL text fields cannot contain NUL (0x00) characters.
    Some emails (especially with corrupted attachments or binary content) may contain these.
    Also handles UTF-8 surrogate pairs and enforces field length limits.
    
    Args:
        text: Input text that may contain NUL bytes or surrogates
        field_name: Name of field being sanitized (for logging)
        max_length: Maximum length for field (truncates if longer)
        
    Returns:
        Sanitized text safe for PostgreSQL, or None if input was None
    """
    if text is None:
        return None
    
    # Check for NUL bytes and log if found
    if '\x00' in text:
        nul_count = text.count('\x00')
        logger.debug(f"Sanitized {nul_count} NUL byte(s) from {field_name}")
    
    # Remove NUL bytes (0x00)
    sanitized = text.replace('\x00', '')
    
    # Remove UTF-8 surrogate characters (can't be encoded in valid UTF-8)
    # Surrogates are in range U+D800 to U+DFFF
    try:
        # Try to encode to catch surrogate issues
        sanitized.encode('utf-8', errors='strict')
    except UnicodeEncodeError:
        # Contains surrogates - replace them
        sanitized = sanitized.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        logger.debug(f"Removed surrogate characters from {field_name}")
    
    # Truncate if max_length specified
    if max_length and len(sanitized) > max_length:
        logger.debug(f"Truncated {field_name} from {len(sanitized)} to {max_length} characters")
        sanitized = sanitized[:max_length]
    
    return sanitized


def sanitize_dict_for_postgres(data: Optional[Dict]) -> Optional[Dict]:
    """
    Recursively sanitize a dictionary to remove NUL bytes from all string values.
    
    Args:
        data: Dictionary that may contain strings with NUL bytes
        
    Returns:
        Sanitized dictionary safe for PostgreSQL JSON storage
    """
    if data is None:
        return None
    
    if not isinstance(data, dict):
        return data
    
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = sanitize_for_postgres(value)
        elif isinstance(value, dict):
            result[key] = sanitize_dict_for_postgres(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_for_postgres(item) if isinstance(item, str)
                else sanitize_dict_for_postgres(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value
    
    return result


class EmailRepository:
    """
    Repository pattern for email database operations.
    Simplifies database access for process_inbox.py and API.
    """
    
    def __init__(self, db: Session):
        """
        Initialize repository with database session.
        
        Args:
            db: SQLAlchemy session
        """
        self.db = db
    
    def bulk_update_folder_lifecycle(self, message_ids_to_folder: List[tuple], current_folder: str, account_id: str = 'work') -> Dict[str, int]:
        """
        Bulk update folders for lifecycle tracking (ultra-fast, batched processing).
        
        Args:
            message_ids_to_folder: List of (message_id, uid) tuples from IMAP
            current_folder: The folder these emails are currently in
            account_id: Account ID to filter emails (required for multi-account support)
            
        Returns:
            Dict with stats: {'updated': N, 'unchanged': N, 'not_in_db': N}
        """
        stats = {'updated': 0, 'unchanged': 0, 'not_in_db': 0}
        
        try:
            # Process in batches of 500 to avoid overwhelming database
            batch_size = 500
            total_items = len(message_ids_to_folder)
            total_batches = (total_items + batch_size - 1) // batch_size
            
            if total_batches > 1:
                logger.info(f"   Processing {total_items} emails in {total_batches} batches...")
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_items)
                batch_items = message_ids_to_folder[start_idx:end_idx]
                
                # Normalize message IDs
                normalized_ids = []
                for msg_id, uid in batch_items:
                    if not msg_id.startswith('<'):
                        normalized_ids.append(f'<{msg_id}>')
                    else:
                        normalized_ids.append(msg_id)
                
                # Bulk fetch emails that exist in DB (filtered by account_id)
                emails = self.db.query(Email.id, Email.message_id, Email.folder).filter(
                    Email.message_id.in_(normalized_ids),
                    Email.account_id == account_id
                ).all()
                
                email_map = {msg_id: (email_id, folder) for email_id, msg_id, folder in emails}
                stats['not_in_db'] += len(normalized_ids) - len(email_map)
                
                # Find emails that changed folders
                emails_to_update = []
                for msg_id in normalized_ids:
                    if msg_id in email_map:
                        email_id, old_folder = email_map[msg_id]
                        if old_folder != current_folder:
                            emails_to_update.append((email_id, msg_id, old_folder))
                        else:
                            stats['unchanged'] += 1
                
                # Bulk create folder history records for this batch
                if emails_to_update:
                    now = datetime.utcnow()
                    
                    # Get account_id for these emails (all should be same account due to filter)
                    account_ids = self.db.query(Email.account_id).filter(
                        Email.id.in_([eid for eid, _, _ in emails_to_update])
                    ).distinct().all()
                    account_id_for_history = account_ids[0][0] if account_ids else account_id
                    
                    # Prepare all records for bulk insert
                    history_records = [
                        {
                            'email_id': email_id,
                            'from_folder': old_folder,
                            'to_folder': current_folder,
                            'from_account_id': account_id_for_history,
                            'to_account_id': account_id_for_history,
                            'is_cross_account': False,
                            'moved_at': now,
                            'moved_by': 'user',
                            'move_reason': 'Detected during lifecycle scan',
                            'created_at': now
                        }
                        for email_id, msg_id, old_folder in emails_to_update
                    ]
                    
                    # Single bulk insert for history
                    if history_records:
                        self.db.execute(EmailLocationHistory.__table__.insert(), history_records)
                    
                    # Update email folders using bulk update
                    email_ids_to_update = [eid for eid, _, _ in emails_to_update]
                    if email_ids_to_update:
                        self.db.execute(
                            Email.__table__.update().where(
                                Email.id.in_(email_ids_to_update)
                            ).values(folder=current_folder, updated_at=now)
                        )
                    
                    # Bulk update metadata for Archive/Trash
                    folder_lower = current_folder.lower()
                    
                    if 'archive' in folder_lower or 'trash' in folder_lower or 'deleted' in folder_lower:
                        email_ids = email_ids_to_update
                        
                        # Get or create metadata records
                        existing_metadata = self.db.query(EmailMetadata.email_id).filter(
                            EmailMetadata.email_id.in_(email_ids)
                        ).all()
                        existing_ids = {m.email_id for m in existing_metadata}
                        
                        # Create missing metadata records
                        new_metadata = [
                            {'email_id': email_id}
                            for email_id in email_ids
                            if email_id not in existing_ids
                        ]
                        
                        if new_metadata:
                            self.db.execute(EmailMetadata.__table__.insert(), new_metadata)
                        
                        # Update all metadata based on folder in one query
                        update_values = {
                            'folder_move_count': EmailMetadata.folder_move_count + 1,
                            'current_folder_since': now,
                            'status_updated_at': now,
                            'updated_at': now
                        }
                        
                        if 'archive' in folder_lower:
                            update_values.update({
                                'user_archived': True,
                                'email_status': 'archived',
                                'importance_score': 5
                            })
                        elif 'trash' in folder_lower or 'deleted' in folder_lower:
                            update_values.update({
                                'user_deleted': True,
                                'email_status': 'deleted',
                                'importance_score': 0
                            })
                        
                        self.db.execute(
                            EmailMetadata.__table__.update().where(
                                EmailMetadata.email_id.in_(email_ids)
                            ).values(update_values)
                        )
                    
                    stats['updated'] += len(emails_to_update)
                
                # Commit this batch
                self.db.commit()
                
                if total_batches > 1 and (batch_num + 1) % 5 == 0:
                    logger.info(f"   Batch {batch_num + 1}/{total_batches}: {stats['updated']} folder changes tracked")
            
            return stats
            
        except Exception as e:
            logger.error(f"Bulk folder lifecycle update failed: {e}")
            self.db.rollback()
            return stats
    
    def get_or_create_email(self, processed_email: ProcessedEmail, account_id: str = "work") -> tuple[Optional[Email], bool]:
        """
        Get email from database or create if doesn't exist.
        Uses message_id + account_id as unique identifier.
        
        Args:
            processed_email: ProcessedEmail from email processor
            account_id: Account nickname (e.g., 'work', 'personal', 'work_archive')
            
        Returns:
            Tuple of (Email database model or None if error, is_new: bool)
            is_new is True if email was just created, False if already existed
        """
        try:
            # Try to find existing by message_id and account_id
            email = self.db.query(Email).filter(
                Email.message_id == processed_email.message_id,
                Email.account_id == account_id
            ).first()
            
            if email:
                # Detect folder change and track it
                old_folder = email.folder
                new_folder = processed_email.folder
                
                if old_folder != new_folder:
                    # Email moved to a different folder - track it
                    logger.info(f"Detected folder change for {email.message_id} in {account_id}: {old_folder} → {new_folder}")
                    
                    # Track the movement (assume user action since we're discovering it during scan)
                    self.track_location_change(
                        email, 
                        new_folder=new_folder,
                        new_account=account_id,
                        moved_by='user',  # Discovered during scan, likely user action
                        move_reason='Detected during folder scan'
                    )
                    
                    email.folder = new_folder
                else:
                    # Same folder, just update flags
                    email.folder = new_folder
                
                email.is_seen = '\\Seen' in processed_email.flags
                email.is_flagged = '\\Flagged' in processed_email.flags
                email.updated_at = datetime.utcnow()
                return email, False  # Not new
            
            # Create new (sanitize text fields to remove NUL bytes and enforce length limits)
            email = Email(
                message_id=processed_email.message_id,
                uid=sanitize_for_postgres(processed_email.uid, max_length=100),
                folder=sanitize_for_postgres(processed_email.folder, max_length=200),
                account_id=account_id,  # NEW: Multi-account support
                original_account_id=account_id,  # NEW: Track where email first appeared
                from_address=sanitize_for_postgres(processed_email.from_address, max_length=500),
                from_name=sanitize_for_postgres(processed_email.from_name, max_length=500),
                to_addresses=processed_email.to_addresses,
                cc_addresses=[],
                subject=sanitize_for_postgres(processed_email.subject),
                date=processed_email.date,
                body_markdown=sanitize_for_postgres(processed_email.body_markdown),
                has_attachments=processed_email.has_attachments,
                attachment_count=processed_email.attachment_count,
                attachment_info=[
                    {
                        'filename': sanitize_for_postgres(att.filename),
                        'content_type': att.content_type,
                        'size': att.size,
                        'extracted': att.extracted_text is not None
                    }
                    for att in processed_email.attachment_info
                ],
                raw_headers=sanitize_dict_for_postgres(processed_email.raw_headers),
                thread_id=sanitize_for_postgres(processed_email.thread_id, max_length=500),
                references=sanitize_for_postgres(processed_email.references),  # Text field, no limit
                is_seen='\\Seen' in processed_email.flags,
                is_flagged='\\Flagged' in processed_email.flags,
            )
            
            self.db.add(email)
            self.db.flush()  # Get ID
            
            # Track initial location (first time seeing this email)
            self.track_location_change(
                email,
                new_folder=processed_email.folder,
                new_account=account_id,  # NEW: Track account
                moved_by='system',
                move_reason='First discovery'
            )
            
            logger.debug(f"Created email in database: {email.message_id}")
            return email, True  # New email
            
        except IntegrityError as e:
            # Duplicate message_id - expected when processing multiple folders or parallel workers
            logger.debug(f"Email already exists in database: {processed_email.message_id}")
            self.db.rollback()
            # Try to fetch the existing one and update folder if needed
            existing = self.db.query(Email).filter(
                Email.message_id == processed_email.message_id
            ).first()
            if existing and existing.folder != processed_email.folder:
                # Update folder if this email is now in a different folder
                logger.debug(f"Updating folder for {processed_email.message_id}: {existing.folder} -> {processed_email.folder}")
                existing.folder = processed_email.folder
                existing.updated_at = datetime.utcnow()
            return existing, False
        except Exception as e:
            logger.error(f"Failed to create/update email {processed_email.message_id}: {e}")
            self.db.rollback()
            return None, False
    
    def store_metadata(self, email: Email, vip_level: Optional[str] = None, 
                      intended_color: Optional[int] = None,
                      ai_result: Optional[AIClassificationResult] = None,
                      two_stage_metadata: Optional[Dict] = None,
                      curated_sender_name: Optional[str] = None,
                      rule_category: Optional[str] = None,
                      application_source: Optional[str] = None) -> Optional[EmailMetadata]:
        """
        Store or update email metadata.
        
        Args:
            email: Email database model
            vip_level: VIP level (urgent/high/medium)
            intended_color: Color code 1-7
            ai_result: AI classification result (final result from Stage 2 if used)
            two_stage_metadata: Optional dict with:
                - stage_2_triggered: bool
                - stage_1_result: AIClassificationResult
                - stage_2_result: AIClassificationResult
                - stage_1_model: str (e.g., "gpt-5-mini")
                - stage_2_model: str (e.g., "gpt-5.1")
                - stage_2_reason: str (why triggered)
                - improvements: dict (what changed)
            rule_category: If set, this category takes precedence over AI category
                (used when rule-based classification should be authoritative)
            application_source: Optional source identifier (e.g., "ai_center") for filtering
            
        Returns:
            EmailMetadata model or None if error
        """
        if not email:
            return None
            
        try:
            # Get or create metadata
            metadata = self.db.query(EmailMetadata).filter(
                EmailMetadata.email_id == email.id
            ).first()
            
            if not metadata:
                metadata = EmailMetadata(email_id=email.id)
                self.db.add(metadata)
            
            # Update curated sender name (extracted via LLM during classification)
            if curated_sender_name:
                metadata.curated_sender_name = curated_sender_name
            
            # Update VIP info
            if vip_level:
                metadata.vip_level = vip_level
            if intended_color:
                metadata.intended_color = intended_color
            
            # Update AI results
            if ai_result:
                # Rule category takes precedence if set (for authoritative rule-based classification)
                metadata.ai_category = rule_category if rule_category else ai_result.category
                # Subcategory is now part of category (flattened: e.g., "application-phd")
                metadata.ai_confidence = ai_result.confidence
                metadata.ai_reasoning = ai_result.reasoning
                metadata.ai_summary = ai_result.summary
                metadata.ai_sentiment = ai_result.sentiment
                metadata.ai_urgency = ai_result.urgency
                metadata.ai_urgency_score = ai_result.urgency_score
                
                # Persist full LLM response (encrypted) for auditability
                try:
                    if hasattr(ai_result, "model_dump"):
                        full_resp = ai_result.model_dump()
                    elif hasattr(ai_result, "dict"):
                        full_resp = ai_result.dict()
                    else:
                        # Fallback if already a dict-like
                        full_resp = ai_result  # type: ignore[assignment]
                    # Only store if it's JSON-serializable
                    json.dumps(full_resp)
                    metadata.ai_full_response = full_resp
                except Exception as e:
                    logger.warning(f"Could not persist full AI response for email {email.id}: {e}")
                
                metadata.ai_action_items = ai_result.action_items
                metadata.needs_reply = ai_result.needs_reply
                metadata.is_cold_email = ai_result.is_cold_email
                metadata.is_followup = ai_result.is_followup
                metadata.relevance_score = ai_result.relevance_score
                metadata.relevance_reason = ai_result.relevance_reason
                metadata.prestige_score = ai_result.prestige_score
                metadata.prestige_reason = ai_result.prestige_reason
                metadata.suggested_folder = ai_result.suggested_folder
                metadata.suggested_labels = ai_result.suggested_labels
                
                # Store two-stage metadata if provided
                if two_stage_metadata:
                    metadata.two_stage_used = two_stage_metadata.get('two_stage_used', False)
                    
                    # Always store Stage 1 results when using two-stage classifier
                    # The metadata dict already has extracted fields, not the full object
                    if two_stage_metadata.get('stage_1_model'):
                        metadata.stage_1_model = two_stage_metadata.get('stage_1_model')
                        metadata.stage_1_category = two_stage_metadata.get('stage_1_category')
                        metadata.stage_1_confidence = two_stage_metadata.get('stage_1_confidence')
                        metadata.stage_1_urgency_score = two_stage_metadata.get('stage_1_urgency_score')
                        metadata.stage_1_recommendation_score = two_stage_metadata.get('stage_1_recommendation_score')
                        metadata.stage_1_scientific_excellence_score = two_stage_metadata.get('stage_1_scientific_excellence_score')
                    
                    # Store Stage 2 info only if it was triggered
                    if two_stage_metadata.get('stage_2_triggered', False):
                        metadata.stage_2_model = two_stage_metadata.get('stage_2_model', 'unknown')
                        metadata.stage_2_reason = two_stage_metadata.get('stage_2_reason', '')[:200]
                        
                        # Log category changes
                        stage_1_cat = two_stage_metadata.get('stage_1_category')
                        if stage_1_cat and stage_1_cat != ai_result.category:
                            logger.warning(
                                f"Two-stage category change: {stage_1_cat} → {ai_result.category} "
                                f"(Reason: {metadata.stage_2_reason})"
                            )
                
                # Store category-specific fields - split into three tiers
                # Tier 1: Direct columns (already set above for some fields)
                # Tier 2: category_metadata (non-PII, queryable)
                # Tier 3: category_specific_data (PII/sensitive, encrypted)
                
                category_metadata = {}  # Non-PII queryable data
                category_data = {}      # PII/sensitive encrypted data
                
                # Event/Deadline info (invitations/reviews) - mostly in metadata
                if ai_result.event_date:
                    category_metadata['event_date'] = ai_result.event_date
                if ai_result.deadline:
                    category_metadata['deadline'] = ai_result.deadline
                if ai_result.location:
                    category_data['location'] = ai_result.location  # Could contain PII
                if ai_result.time_commitment_hours:
                    category_metadata['time_commitment_hours'] = ai_result.time_commitment_hours
                if ai_result.time_commitment_reason:
                    category_data['time_commitment_reason'] = ai_result.time_commitment_reason
                
                # Application-specific fields - THREE-TIER SPLIT
                if ai_result.category and ai_result.category.startswith('application-'):
                    # Tier 1: Direct columns (already in EmailMetadata model)
                    if ai_result.scientific_excellence_score:
                        metadata.scientific_excellence_score = ai_result.scientific_excellence_score
                    if ai_result.research_fit_score:
                        metadata.research_fit_score = ai_result.research_fit_score
                    if ai_result.recommendation_score:
                        metadata.overall_recommendation_score = ai_result.recommendation_score
                    # Also store in columns that already exist
                    if ai_result.applicant_name:
                        metadata.applicant_name = ai_result.applicant_name
                    if ai_result.applicant_email:
                        metadata.applicant_email = ai_result.applicant_email
                    if ai_result.applicant_institution:
                        metadata.applicant_institution = ai_result.applicant_institution
                    
                    # Tier 2: category_metadata (non-PII, queryable)
                    if ai_result.profile_tags:
                        # Convert Pydantic models to dicts for JSON storage
                        if isinstance(ai_result.profile_tags, list):
                            tags_list = []
                            for tag in ai_result.profile_tags:
                                if hasattr(tag, 'model_dump'):  # Pydantic v2
                                    tags_list.append(tag.model_dump())
                                elif hasattr(tag, 'dict'):  # Pydantic v1
                                    tags_list.append(tag.dict())
                                else:  # Already a dict
                                    tags_list.append(tag)
                            category_metadata['profile_tags'] = tags_list
                        else:
                            category_metadata['profile_tags'] = ai_result.profile_tags
                    if ai_result.information_used:
                        # Convert Pydantic model to dict for JSON storage
                        if hasattr(ai_result.information_used, 'model_dump'):  # Pydantic v2
                            category_metadata['information_used'] = ai_result.information_used.model_dump()
                        elif hasattr(ai_result.information_used, 'dict'):  # Pydantic v1
                            category_metadata['information_used'] = ai_result.information_used.dict()
                        else:  # Already a dict
                            category_metadata['information_used'] = ai_result.information_used
                    
                    # Red flags in metadata (non-PII)
                    category_metadata['red_flags'] = {
                        'is_mass_email': ai_result.is_mass_email or False,
                        'no_research_background': ai_result.no_research_background or False,
                        'irrelevant_field': ai_result.irrelevant_field or False,
                        'possible_spam': ai_result.possible_spam or False,
                        'insufficient_materials': ai_result.insufficient_materials or False,
                        'is_followup': ai_result.is_followup or False,
                        'is_not_application': ai_result.is_not_application or False,
                        'prompt_manipulation_detected': ai_result.prompt_manipulation_detected or False,
                        'prompt_manipulation_indicators': ai_result.prompt_manipulation_indicators or []
                    }
                    # Store correct_category and reason separately (not in red_flags) for easier querying
                    if ai_result.is_not_application:
                        if ai_result.correct_category:
                            category_metadata['correct_category'] = ai_result.correct_category
                        if ai_result.is_not_application_reason:
                            category_metadata['is_not_application_reason'] = ai_result.is_not_application_reason
                    
                    # Technical experience scores (just numbers, non-PII)
                    # Handle both Pydantic models and dicts for backward compatibility
                    def _get_score(exp):
                        """Extract score from TechnicalExperience model or dict."""
                        if exp is None:
                            return None
                        if hasattr(exp, 'score'):  # Pydantic model
                            return exp.score
                        if isinstance(exp, dict):  # Dict (backward compat)
                            return exp.get('score', 0)
                        return 0
                    
                    tech_scores = {}
                    if ai_result.coding_experience:
                        tech_scores['coding_experience'] = _get_score(ai_result.coding_experience)
                    if ai_result.omics_genomics_experience:
                        tech_scores['omics_genomics_experience'] = _get_score(ai_result.omics_genomics_experience)
                    if ai_result.medical_data_experience:
                        tech_scores['medical_data_experience'] = _get_score(ai_result.medical_data_experience)
                    if ai_result.sequence_analysis_algorithms_experience:
                        tech_scores['sequence_analysis_algorithms_experience'] = _get_score(ai_result.sequence_analysis_algorithms_experience)
                    if ai_result.image_analysis_experience:
                        tech_scores['image_analysis_experience'] = _get_score(ai_result.image_analysis_experience)
                    if tech_scores:
                        category_metadata['technical_experience_scores'] = tech_scores
                    
                    # Tier 3: category_specific_data (PII/sensitive, encrypted)
                    if ai_result.nationality:
                        category_data['nationality'] = ai_result.nationality
                    if ai_result.highest_degree_completed:
                        category_data['highest_degree_completed'] = ai_result.highest_degree_completed
                    if ai_result.current_situation:
                        category_data['current_situation'] = ai_result.current_situation
                    if ai_result.recent_thesis_title:
                        category_data['recent_thesis_title'] = ai_result.recent_thesis_title
                    if ai_result.recommendation_source:
                        category_data['recommendation_source'] = ai_result.recommendation_source
                    
                    # Online profiles (PII)
                    online_profiles = {}
                    if ai_result.github_account:
                        online_profiles['github_account'] = ai_result.github_account
                    if ai_result.linkedin_account:
                        online_profiles['linkedin_account'] = ai_result.linkedin_account
                    if ai_result.google_scholar_account:
                        online_profiles['google_scholar_account'] = ai_result.google_scholar_account
                    if online_profiles:
                        category_data['online_profiles'] = online_profiles
                    
                    # Technical experience evidence (contains quotes - sensitive)
                    def _get_evidence(exp):
                        """Extract evidence from TechnicalExperience model or dict."""
                        if exp is None:
                            return None
                        if hasattr(exp, 'evidence'):  # Pydantic model
                            return exp.evidence
                        if isinstance(exp, dict):  # Dict (backward compat)
                            return exp.get('evidence')
                        return None
                    
                    tech_evidence = {}
                    if ai_result.coding_experience:
                        tech_evidence['coding_experience'] = _get_evidence(ai_result.coding_experience)
                    if ai_result.omics_genomics_experience:
                        tech_evidence['omics_genomics_experience'] = _get_evidence(ai_result.omics_genomics_experience)
                    if ai_result.medical_data_experience:
                        tech_evidence['medical_data_experience'] = _get_evidence(ai_result.medical_data_experience)
                    if ai_result.sequence_analysis_algorithms_experience:
                        tech_evidence['sequence_analysis_algorithms_experience'] = _get_evidence(ai_result.sequence_analysis_algorithms_experience)
                    if ai_result.image_analysis_experience:
                        tech_evidence['image_analysis_experience'] = _get_evidence(ai_result.image_analysis_experience)
                    if tech_evidence:
                        category_data['technical_experience_evidence'] = tech_evidence
                    
                    # Evaluation reasoning (sensitive opinions)
                    evaluation_reasoning = {}
                    if ai_result.scientific_excellence_reason:
                        evaluation_reasoning['scientific_excellence_reason'] = ai_result.scientific_excellence_reason
                    if ai_result.research_fit_reason:
                        evaluation_reasoning['research_fit_reason'] = ai_result.research_fit_reason
                    if ai_result.recommendation_reason:
                        evaluation_reasoning['overall_recommendation_reason'] = ai_result.recommendation_reason
                    if evaluation_reasoning:
                        category_data['evaluation_reasoning'] = evaluation_reasoning
                    
                    # Additional information request flags (non-PII, queryable)
                    if ai_result.should_request_additional_info is not None:
                        category_metadata['should_request_additional_info'] = ai_result.should_request_additional_info
                    if ai_result.missing_information_items:
                        category_metadata['missing_information_items'] = ai_result.missing_information_items
                    if ai_result.potential_recommendation_score is not None:
                        category_metadata['potential_recommendation_score'] = ai_result.potential_recommendation_score
                    
                    # Key strengths, concerns, notes (sensitive evaluation details)
                    if ai_result.key_strengths:
                        category_data['key_strengths'] = ai_result.key_strengths
                    if ai_result.concerns:
                        category_data['concerns'] = ai_result.concerns
                    if ai_result.next_steps:
                        category_data['next_steps'] = ai_result.next_steps
                    if ai_result.additional_notes:
                        category_data['additional_notes'] = ai_result.additional_notes
                
                # Receipt-specific fields
                if ai_result.vendor:
                    category_data['vendor'] = ai_result.vendor
                if ai_result.amount:
                    category_data['amount'] = ai_result.amount
                if ai_result.currency:
                    category_data['currency'] = ai_result.currency
                
                # Draft responses
                if ai_result.answer_options:
                    category_data['answer_options'] = ai_result.answer_options
                
                # Note: application_source is stored outside this block to ensure
                # it's saved even when AI classification fails
                
                # Store category_metadata (non-PII, queryable)
                # MERGE with existing metadata to preserve fields like application_source
                if category_metadata:
                    try:
                        json.dumps(category_metadata)
                        # Preserve existing metadata and merge new data
                        # IMPORTANT: Create a NEW dict to ensure SQLAlchemy detects the change
                        existing_metadata = dict(metadata.category_metadata or {})  # Create new dict copy
                        existing_metadata.update(category_metadata)
                        metadata.category_metadata = existing_metadata
                        # Flag the column as modified to ensure SQLAlchemy persists the change
                        from sqlalchemy.orm.attributes import flag_modified
                        flag_modified(metadata, 'category_metadata')
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Category metadata not JSON-serializable, skipping: {e}")
                        # Don't overwrite with empty dict - preserve existing
                        if not metadata.category_metadata:
                            metadata.category_metadata = {}
                
                # Store category_specific_data (PII/sensitive, encrypted)
                # MERGE with existing data to preserve previously stored fields
                if category_data:
                    # Validate that data is JSON-serializable before storing
                    try:
                        json.dumps(category_data)
                        # Preserve existing data and merge new data
                        # IMPORTANT: We must create a NEW dict to ensure SQLAlchemy detects the change
                        # for encrypted columns (TypeDecorator doesn't track in-place mutations)
                        existing_data = dict(metadata.category_specific_data or {})  # Create new dict copy
                        logger.info(f"📝 store_metadata: category_data keys to add: {list(category_data.keys())}")
                        logger.info(f"📝 store_metadata: existing_data keys before merge: {list(existing_data.keys())}")
                        existing_data.update(category_data)
                        logger.info(f"📝 store_metadata: merged data keys: {list(existing_data.keys())}")
                        metadata.category_specific_data = existing_data
                        # Flag the column as modified to ensure SQLAlchemy persists the change
                        from sqlalchemy.orm.attributes import flag_modified
                        flag_modified(metadata, 'category_specific_data')
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Category data not JSON-serializable, skipping: {e}")
                        # Don't overwrite with empty dict - preserve existing
                        if not metadata.category_specific_data:
                            metadata.category_specific_data = {}
                else:
                    logger.warning(f"📝 store_metadata: category_data is EMPTY - nothing to add to category_specific_data")
                
                if ai_result.reply_deadline:
                    try:
                        metadata.reply_deadline = datetime.fromisoformat(ai_result.reply_deadline)
                    except:
                        pass
                
                if ai_result.followup_to_date:
                    try:
                        metadata.followup_to_date = datetime.fromisoformat(ai_result.followup_to_date)
                    except:
                        pass
            
            # Store application_source OUTSIDE the if ai_result block
            # This ensures it's stored even when AI classification fails/is skipped
            # (application_source comes from rule-based classification, not AI)
            if application_source:
                existing_metadata = metadata.category_metadata or {}
                existing_metadata['application_source'] = application_source
                metadata.category_metadata = existing_metadata
        
            metadata.updated_at = datetime.utcnow()
            self.db.flush()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to store metadata for email {email.id}: {e}")
            self.db.rollback()
            return None
    
    def store_classification(self, email: Email, classifier_type: str,
                           category: str, confidence: float,
                           reasoning: str, model_used: Optional[str] = None) -> Optional[Classification]:
        """
        Store classification in audit log.
        
        Args:
            email: Email model
            classifier_type: "rule", "ai", "vip", "manual"
            category: Classification category
            confidence: 0-1
            reasoning: Why classified this way
            model_used: AI model name (if AI classification)
            
        Returns:
            Classification model or None if error
        """
        try:
            classification = Classification(
                email_id=email.id,
                classifier_type=classifier_type,
                category=category,
                confidence=confidence,
                reasoning=reasoning,
                model_used=model_used
            )
            
            self.db.add(classification)
            self.db.flush()
            
            return classification
            
        except Exception as e:
            logger.error(f"Failed to store classification for email {email.id}: {e}")
            self.db.rollback()
            return None
    
    def update_sender_history(self, email: Email) -> Optional[SenderHistory]:
        """
        Update sender history statistics.
        
        Args:
            email: Email model
            
        Returns:
            SenderHistory model or None if error
        """
        try:
            sender = self.db.query(SenderHistory).filter(
                SenderHistory.email_address == email.from_address
            ).first()
            
            if not sender:
                sender = SenderHistory(
                    email_address=email.from_address,
                    sender_name=email.from_name,
                    domain=email.from_address.split('@')[1] if '@' in email.from_address else '',
                    email_count=0,
                    first_seen=email.date,
                    last_seen=email.date
                )
                self.db.add(sender)
            
            # Update statistics
            sender.email_count += 1
            sender.last_seen = email.date
            sender.is_frequent = sender.email_count > 10
            
            # Update sender name if we have a better one
            if email.from_name and not sender.sender_name:
                sender.sender_name = email.from_name
            
            sender.updated_at = datetime.utcnow()
            self.db.flush()
            
            return sender
            
        except Exception as e:
            logger.error(f"Failed to update sender history for {email.from_address}: {e}")
            self.db.rollback()
            return None
    
    def get_sender_history(self, email_address: str) -> Optional[Dict]:
        """
        Get sender history as dict for AI classifier.
        
        Args:
            email_address: Sender email address
            
        Returns:
            Dict with sender statistics or None
        """
        sender = self.db.query(SenderHistory).filter(
            SenderHistory.email_address == email_address
        ).first()
        
        if not sender:
            return None
        
        return {
            'email_count': sender.email_count,
            'sender_type': sender.sender_type,
            'typical_category': sender.typical_category,
            'is_frequent': sender.is_frequent,
            'is_cold_sender': sender.is_cold_sender,
            'avg_reply_time_hours': sender.avg_reply_time_hours,
            'last_seen': sender.last_seen.isoformat() if sender.last_seen else None,
            'first_seen': sender.first_seen.isoformat() if sender.first_seen else None,
        }
    
    def commit(self):
        """Commit transaction with error handling"""
        try:
            self.db.commit()
        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            self.db.rollback()
            raise
    
    def rollback(self):
        """Rollback transaction"""
        try:
            self.db.rollback()
        except Exception as e:
            logger.error(f"Failed to rollback transaction: {e}")
    
    def bulk_store_emails(self, processed_emails: List[ProcessedEmail]) -> List[tuple[Optional[Email], bool]]:
        """
        Store multiple emails efficiently in batch.
        
        Args:
            processed_emails: List of ProcessedEmail objects
            
        Returns:
            List of tuples (Email model or None, is_new: bool)
        """
        results = []
        
        try:
            for proc_email in processed_emails:
                email, is_new = self.get_or_create_email(proc_email)
                results.append((email, is_new))
            
            # Single flush for all
            self.db.flush()
            
            new_count = sum(1 for _, is_new in results if is_new)
            logger.info(f"Bulk stored {len(results)} emails ({new_count} new, {len(results)-new_count} updated)")
            
        except Exception as e:
            logger.error(f"Bulk store failed: {e}")
            self.db.rollback()
        
        return results
    
    def bulk_check_processing_status(self, message_ids: List[str], 
                                     check_embeddings: bool = False,
                                     check_ai_classification: bool = False,
                                     account_id: str = "work") -> Dict[str, Dict[str, Any]]:
        """
        Efficiently check processing status for multiple emails in a single query.
        This is 100x faster than checking each email individually.
        
        Args:
            message_ids: List of email Message-IDs to check (with or without angle brackets)
            check_embeddings: If True, check if embeddings exist
            check_ai_classification: If True, check if AI classification exists
            account_id: Account ID to filter emails (required for multi-account support)
            
        Returns:
            Dict mapping message_id to status:
            {
                'msg123': {
                    'in_db': True,
                    'has_embedding': False,
                    'has_ai_classification': True,
                    'email_id': 'uuid-...'
                },
                ...
            }
        """
        if not message_ids:
            return {}
        
        try:
            # Import here to avoid circular imports
            from .models import EmailEmbedding
            
            # Normalize message IDs: add angle brackets if missing (database stores them with brackets)
            normalized_ids = []
            original_to_normalized = {}
            for msg_id in message_ids:
                if not msg_id.startswith('<'):
                    normalized = f'<{msg_id}>'
                else:
                    normalized = msg_id
                normalized_ids.append(normalized)
                original_to_normalized[msg_id] = normalized
            
            # Query emails by message_id in batches to avoid timeout on large datasets
            # OPTIMIZATION: Only select id and message_id columns (not full email records)
            email_map = {}
            batch_size = 250  # Conservative batch size to avoid timeout even on slow connections
            total_batches = (len(normalized_ids) + batch_size - 1) // batch_size
            
            if total_batches > 1:
                logger.info(f"   Checking {len(normalized_ids)} emails in {total_batches} batches...")
            
            for batch_num, i in enumerate(range(0, len(normalized_ids), batch_size), 1):
                batch = normalized_ids[i:i + batch_size]
                # Only fetch id and message_id columns (not body, headers, etc)
                # Filter by account_id to ensure we check the correct account's emails
                # Add timeout handling for very large inboxes
                try:
                    batch_emails = self.db.query(Email.id, Email.message_id).filter(
                        Email.message_id.in_(batch),
                        Email.account_id == account_id
                    ).all()
                except Exception as e:
                    logger.warning(f"Query timeout/error for batch {batch_num}: {e}")
                    # Continue with next batch rather than failing completely
                    logger.info(f"   Skipping batch {batch_num} ({len(batch)} message IDs)")
                    continue
                for email_id, message_id in batch_emails:
                    # Store as simple object with just id and message_id
                    class EmailStub:
                        pass
                    stub = EmailStub()
                    stub.id = email_id
                    stub.message_id = message_id
                    email_map[message_id] = stub
                
                if total_batches > 1 and batch_num % 10 == 0:
                    logger.info(f"   Checked emails batch {batch_num}/{total_batches} ({len(email_map)} found so far)")
            
            # Get embeddings if requested (batch queries to avoid timeout)
            # Note: Querying email_embeddings is slower due to Vector(3072) column
            embeddings_map = {}
            if check_embeddings and email_map:
                email_ids = [email.id for email in email_map.values()]
                
                # Use small batches for embeddings (Vector columns are very expensive)
                # 100 is very conservative but safe for remote databases
                batch_size = 100
                total_batches = (len(email_ids) + batch_size - 1) // batch_size
                
                if total_batches > 1:
                    logger.info(f"   Checking embeddings for {len(email_ids)} emails in {total_batches} batches...")
                
                for batch_num, i in enumerate(range(0, len(email_ids), batch_size), 1):
                    batch = email_ids[i:i + batch_size]
                    # Only select the email_id column (not the full embedding vector)
                    # Add timeout handling
                    try:
                        embeddings = self.db.query(EmailEmbedding.email_id).filter(
                            EmailEmbedding.email_id.in_(batch)
                        ).all()
                    except Exception as e:
                        logger.warning(f"Embedding query timeout/error for batch {batch_num}: {e}")
                        continue
                    # Just mark as having embedding (we don't need the actual vector)
                    for emb in embeddings:
                        embeddings_map[emb.email_id] = True  # Just a marker
                    
                    if total_batches > 1 and batch_num % 20 == 0:
                        logger.info(f"   Checked embeddings batch {batch_num}/{total_batches} ({len(embeddings_map)} found)")
            
            # Get AI classifications if requested (batch queries to avoid timeout)
            ai_class_map = {}
            if check_ai_classification and email_map:
                email_ids = [email.id for email in email_map.values()]
                
                # Batch in chunks of 250 (match email batch size for consistency)
                batch_size = 250
                total_batches = (len(email_ids) + batch_size - 1) // batch_size
                
                if total_batches > 1:
                    logger.info(f"   Checking AI classifications for {len(email_ids)} emails in {total_batches} batches...")
                
                for batch_num, i in enumerate(range(0, len(email_ids), batch_size), 1):
                    batch = email_ids[i:i + batch_size]
                    # Add timeout handling
                    try:
                        ai_classes = self.db.query(Classification.email_id).filter(
                            Classification.email_id.in_(batch),
                            Classification.classifier_type == 'ai'
                        ).all()
                    except Exception as e:
                        logger.warning(f"AI classification query timeout/error for batch {batch_num}: {e}")
                        continue
                    # Just mark as having AI classification
                    for cls in ai_classes:
                        ai_class_map[cls.email_id] = True  # Just a marker
                    
                    if total_batches > 1 and batch_num % 20 == 0:
                        logger.info(f"   Checked AI classifications batch {batch_num}/{total_batches} ({len(ai_class_map)} found)")
            
            # Build result (use original message IDs as keys)
            result = {}
            for message_id in message_ids:
                normalized_id = original_to_normalized[message_id]
                email = email_map.get(normalized_id)
                
                if email:
                    result[message_id] = {
                        'in_db': True,
                        'email_id': str(email.id),
                        'has_embedding': email.id in embeddings_map if check_embeddings else None,
                        'has_ai_classification': email.id in ai_class_map if check_ai_classification else None
                    }
                else:
                    result[message_id] = {
                        'in_db': False,
                        'email_id': None,
                        'has_embedding': None,
                        'has_ai_classification': None
                    }
            
            in_db_count = sum(1 for r in result.values() if r['in_db'])
            has_emb_count = sum(1 for r in result.values() if r.get('has_embedding')) if check_embeddings else 0
            
            logger.info(f"Bulk checked {len(message_ids)} emails: "
                       f"{in_db_count} in DB, "
                       f"{has_emb_count} have embeddings")
            
            # Debug: Show sample of message IDs if none found
            if in_db_count == 0 and len(message_ids) > 0:
                logger.debug(f"Sample message_ids from IMAP: {message_ids[:3]}")
                sample_db_ids = self.db.query(Email.message_id).limit(3).all()
                logger.debug(f"Sample message_ids from DB: {[e.message_id for e in sample_db_ids]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to bulk check processing status: {e}")
            return {}
    
    def get_folder_sync_state(self, folder: str, account_id: str = "work") -> Optional[FolderSyncState]:
        """
        Get sync state for a folder in a specific account (last processed UID).
        
        Args:
            folder: IMAP folder name
            account_id: Account nickname (e.g., 'work', 'personal', 'work_archive')
            
        Returns:
            FolderSyncState or None if folder never synced
        """
        try:
            return self.db.query(FolderSyncState).filter(
                FolderSyncState.account_id == account_id,
                FolderSyncState.folder == folder
            ).first()
        except Exception as e:
            logger.error(f"Failed to get sync state for {account_id}:{folder}: {e}")
            return None
    
    def update_folder_sync_state(self, folder: str, last_uid: int, emails_processed: int = 0, 
                                 uid_validity: Optional[int] = None, account_id: str = "work"):
        """
        Update sync state for a folder in a specific account after processing emails.
        
        Args:
            folder: IMAP folder name
            last_uid: Highest UID processed in this sync
            emails_processed: Number of emails processed in this sync
            uid_validity: UIDVALIDITY from IMAP server (to detect folder resets)
            account_id: Account nickname (e.g., 'work', 'personal', 'work_archive')
        """
        try:
            sync_state = self.get_folder_sync_state(folder, account_id)
            
            if sync_state:
                # Update existing
                if last_uid > sync_state.last_processed_uid:
                    sync_state.last_processed_uid = last_uid
                sync_state.total_processed += emails_processed
                sync_state.last_sync_at = datetime.utcnow()
                if uid_validity:
                    # Check for UIDVALIDITY change (folder was reset)
                    if sync_state.uid_validity and sync_state.uid_validity != uid_validity:
                        logger.warning(f"UIDVALIDITY changed for {account_id}:{folder} - folder was reset! Resetting sync state.")
                        sync_state.last_processed_uid = last_uid
                        sync_state.total_processed = emails_processed
                    sync_state.uid_validity = uid_validity
            else:
                # Create new
                sync_state = FolderSyncState(
                    account_id=account_id,
                    folder=folder,
                    last_processed_uid=last_uid,
                    total_processed=emails_processed,
                    uid_validity=uid_validity
                )
                self.db.add(sync_state)
            
            self.db.flush()
            logger.debug(f"Updated sync state for {account_id}:{folder}: last_uid={last_uid}, total={sync_state.total_processed}")
            
        except Exception as e:
            logger.error(f"Failed to update sync state for {account_id}:{folder}: {e}")
    
    def track_location_change(self, email: Email, new_folder: str, new_account: str,
                             moved_by: str = "unknown", 
                             move_reason: Optional[str] = None) -> Optional[EmailLocationHistory]:
        """
        Track when an email moves to a different folder or account.
        Records movement history and updates email lifecycle metadata.
        
        Args:
            email: Email database model
            new_folder: New folder name
            new_account: New account ID  
            moved_by: Who moved it ('rule', 'ai', 'user', 'system', 'cross_account_rule', 'ui')
            move_reason: Why it was moved (rule name, AI category, etc)
        
        Returns:
            EmailLocationHistory record or None if no change
        """
        try:
            old_folder = email.folder
            old_account = email.account_id
            
            # No change, nothing to track
            if old_folder == new_folder and old_account == new_account:
                return None
            
            # Determine if cross-account
            is_cross_account = (old_account != new_account)
            
            # Get last history entry to calculate time in previous location
            last_history = self.db.query(EmailLocationHistory).filter(
                EmailLocationHistory.email_id == email.id
            ).order_by(EmailLocationHistory.moved_at.desc()).first()
            
            time_in_previous = None
            if last_history:
                time_delta = datetime.utcnow() - last_history.moved_at
                time_in_previous = int(time_delta.total_seconds())
            
            # Create history record
            history = EmailLocationHistory(
                email_id=email.id,
                from_account_id=old_account if is_cross_account else None,
                from_folder=old_folder,
                to_account_id=new_account,
                to_folder=new_folder,
                moved_at=datetime.utcnow(),
                moved_by=moved_by,
                move_reason=move_reason,
                is_cross_account=is_cross_account,
                time_in_previous_location_seconds=time_in_previous
            )
            self.db.add(history)
            
            # Update email metadata lifecycle fields
            metadata = self.db.query(EmailMetadata).filter(
                EmailMetadata.email_id == email.id
            ).first()
            
            if not metadata:
                metadata = EmailMetadata(email_id=email.id)
                self.db.add(metadata)
            
            # Set first seen info if this is the first movement
            if not metadata.first_seen_folder:
                metadata.first_seen_folder = old_folder or new_folder
                metadata.first_seen_at = email.created_at or datetime.utcnow()
            
            # Update current folder tracking
            metadata.current_folder_since = datetime.utcnow()
            metadata.folder_move_count = (metadata.folder_move_count or 0) + 1
            
            # Detect user actions and update status
            if new_folder and 'archive' in new_folder.lower():
                metadata.user_archived = True
                metadata.email_status = 'archived'
                metadata.status_updated_at = datetime.utcnow()
                
                # Calculate time to action (if moving from INBOX)
                if old_folder == 'INBOX' and metadata.first_seen_at:
                    time_delta = datetime.utcnow() - metadata.first_seen_at
                    metadata.time_to_action_seconds = int(time_delta.total_seconds())
                
                # Archive suggests handled/medium importance
                metadata.importance_score = 5
                
            elif new_folder and ('trash' in new_folder.lower() or 'deleted' in new_folder.lower()):
                metadata.user_deleted = True
                metadata.email_status = 'deleted'
                metadata.status_updated_at = datetime.utcnow()
                
                # Trash suggests low importance
                metadata.importance_score = 0
                
            elif moved_by in ['rule', 'ai']:
                metadata.auto_filed = True
                # Auto-filed to specific folder suggests higher importance
                if new_folder and new_folder not in ['INBOX', 'Sent Items', 'Sent']:
                    metadata.importance_score = 7
                    metadata.email_status = 'handled'
                    metadata.status_updated_at = datetime.utcnow()

            # Update email's account_id and folder for cross-account moves
            # This ensures the email won't be reprocessed when the target account is processed
            if is_cross_account:
                email.account_id = new_account
                logger.info(f"Updated email account_id: {old_account} → {new_account}")
            email.folder = new_folder

            self.db.flush()
            logger.info(f"Tracked folder change for {email.message_id}: {old_folder} → {new_folder} (by {moved_by})")
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to track folder change: {e}")
            self.db.rollback()
            return None
    
    def get_folder_history(self, email: Email) -> List[EmailLocationHistory]:
        """
        Get complete folder movement history for an email.
        
        Args:
            email: Email database model
            
        Returns:
            List of EmailLocationHistory records, ordered by time (oldest first)
        """
        try:
            return self.db.query(EmailLocationHistory).filter(
                EmailLocationHistory.email_id == email.id
            ).order_by(EmailLocationHistory.moved_at.asc()).all()
        except Exception as e:
            logger.error(f"Failed to get folder history: {e}")
            return []
    
    def should_generate_embedding(self, email: Email, folder: str) -> bool:
        """
        Determine if we should generate embeddings for this email based on folder and status.
        
        Rules:
        - INBOX, Sent Items: Always generate (active emails)
        - Organized folders (Applications, etc): Always generate (important)
        - Archive: Use existing (already generated when in INBOX)
        - Trash/Deleted: Never generate (waste of resources)
        
        Args:
            email: Email database model
            folder: Current folder name
            
        Returns:
            True if embeddings should be generated
        """
        folder_lower = folder.lower()
        
        # Never generate for trash
        if 'trash' in folder_lower or 'deleted' in folder_lower:
            logger.debug(f"Skipping embeddings for {email.message_id} (in trash)")
            return False
        
        # Check if email was deleted (status tracking)
        metadata = self.db.query(EmailMetadata).filter(
            EmailMetadata.email_id == email.id
        ).first()
        
        if metadata and metadata.user_deleted:
            logger.debug(f"Skipping embeddings for {email.message_id} (user deleted)")
            return False
        
        # For archive, only generate if we don't have one yet
        # (should already exist from when in INBOX)
        if 'archive' in folder_lower:
            from .models import EmailEmbedding
            existing = self.db.query(EmailEmbedding).filter(
                EmailEmbedding.email_id == email.id
            ).first()
            
            if existing:
                logger.debug(f"Skipping embeddings for {email.message_id} (already exists, in archive)")
                return False
            else:
                # Archive email without embedding - generate it
                logger.info(f"Generating embeddings for archived {email.message_id} (missing from inbox processing)")
                return True
        
        # All other folders: generate embeddings
        return True

    def get_emails_for_attachment_backfill(
        self,
        since_date: Optional[datetime] = None,
        until_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        include_retries: bool = True,
        account_id: Optional[str] = None,
        max_attempts: int = 5,
    ) -> List[Email]:
        """
        Get emails that need attachment indexing (for backfill).

        Finds emails with attachments that haven't been indexed yet, or
        that failed previously and are due for retry (with exponential backoff).

        Backoff schedule:
        - Attempt 2: after 1 hour
        - Attempt 3: after 4 hours
        - Attempt 4: after 1 day
        - Attempt 5: after 1 week
        - After 5: marked as 'failed', no more retries

        Args:
            since_date: Only include emails after this date
            until_date: Only include emails before this date
            limit: Maximum number of emails to return
            include_retries: Include previously failed emails due for retry
            account_id: Filter to specific account (optional)
            max_attempts: Maximum retry attempts before permanent failure

        Returns:
            List of Email objects needing attachment indexing
        """
        now = datetime.utcnow()

        # Build base conditions
        conditions = [
            Email.has_attachments == True,  # noqa: E712
        ]

        if since_date:
            conditions.append(Email.date >= since_date)
        if until_date:
            conditions.append(Email.date <= until_date)
        if account_id:
            conditions.append(Email.account_id == account_id)

        # Status conditions: never attempted OR pending OR due for retry
        status_conditions = [
            Email.attachment_index_status.is_(None),  # Never attempted
            Email.attachment_index_status == 'pending',  # Queued
        ]

        if include_retries:
            # Calculate backoff intervals for each attempt level
            backoff_intervals = {
                1: timedelta(hours=1),    # 2nd attempt after 1 hour
                2: timedelta(hours=4),    # 3rd attempt after 4 hours
                3: timedelta(days=1),     # 4th attempt after 1 day
                4: timedelta(weeks=1),    # 5th attempt after 1 week
            }

            # Build retry condition: failed but under max attempts and backoff elapsed
            retry_conditions = []
            for attempts, interval in backoff_intervals.items():
                if attempts < max_attempts:
                    retry_conditions.append(
                        and_(
                            Email.attachment_index_attempts == attempts,
                            Email.attachment_index_last_attempt < now - interval
                        )
                    )

            if retry_conditions:
                status_conditions.append(
                    and_(
                        Email.attachment_index_status.in_(['failed', 'partial']),
                        Email.attachment_index_attempts < max_attempts,
                        or_(*retry_conditions)
                    )
                )

        # Combine all conditions
        conditions.append(or_(*status_conditions))

        # Query with ordering (oldest first to process chronologically)
        query = (
            self.db.query(Email)
            .filter(and_(*conditions))
            .order_by(Email.date.asc())
        )

        if limit:
            query = query.limit(limit)

        return query.all()

    def update_attachment_index_status(
        self,
        email: Email,
        status: str,
        error: Optional[str] = None,
        increment_attempts: bool = False,
    ) -> None:
        """
        Update the attachment indexing status for an email.

        Args:
            email: Email to update
            status: New status ('pending', 'success', 'partial', 'failed')
            error: Error message if failed
            increment_attempts: Whether to increment the attempt counter
        """
        email.attachment_index_status = status
        email.attachment_index_last_attempt = datetime.utcnow()

        if error:
            email.attachment_index_error = error[:2000]  # Truncate long errors

        if increment_attempts:
            email.attachment_index_attempts = (email.attachment_index_attempts or 0) + 1

        email.updated_at = datetime.utcnow()

    def get_attachment_backfill_stats(
        self,
        since_date: Optional[datetime] = None,
        account_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Get statistics about attachment indexing status.

        Args:
            since_date: Only count emails after this date
            account_id: Filter to specific account (optional)

        Returns:
            Dict with counts: total_with_attachments, indexed, pending, failed, never_attempted
        """
        from sqlalchemy import func

        conditions = [Email.has_attachments == True]  # noqa: E712
        if since_date:
            conditions.append(Email.date >= since_date)
        if account_id:
            conditions.append(Email.account_id == account_id)

        # Base query
        base_query = self.db.query(func.count(Email.id)).filter(and_(*conditions))

        # Total with attachments
        total = base_query.scalar() or 0

        # By status
        success = base_query.filter(Email.attachment_index_status == 'success').scalar() or 0
        partial = base_query.filter(Email.attachment_index_status == 'partial').scalar() or 0
        failed = base_query.filter(Email.attachment_index_status == 'failed').scalar() or 0
        pending = base_query.filter(Email.attachment_index_status == 'pending').scalar() or 0
        never_attempted = base_query.filter(Email.attachment_index_status.is_(None)).scalar() or 0

        return {
            'total_with_attachments': total,
            'indexed_success': success,
            'indexed_partial': partial,
            'failed': failed,
            'pending': pending,
            'never_attempted': never_attempted,
        }

