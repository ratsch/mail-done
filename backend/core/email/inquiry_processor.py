"""
Inquiry Processor - Handles #application #info emails

This module orchestrates the entire inquiry processing pipeline:
1. Detects #info emails (Stage 0)
2. Runs AI classification for validation + type + name extraction
3. Generates draft responses
4. Creates IMAP drafts
5. Moves emails to inquiry folder
6. Updates database
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timedelta

from .inquiry_handler import (
    is_inquiry_email,
    extract_inquiry_type_from_tags,
    build_inquiry_classification_prompt,
    parse_inquiry_classification_response,
    validate_extracted_name,
    get_inquiry_category,
    InquiryClassificationResult,
    InquiryTemplateLoader,
    InquiryDraftGenerator,
)
from .models import ProcessedEmail

logger = logging.getLogger(__name__)


class InquiryProcessor:
    """
    Orchestrates the inquiry email processing pipeline.
    """
    
    # Folder where inquiry emails are moved after processing
    INQUIRY_FOLDER = "MD/Applications/Inquiries"
    
    # Days within which we consider a duplicate inquiry (same sender + overlapping types)
    DUPLICATE_WINDOW_DAYS = 30
    
    def __init__(
        self,
        ai_classifier=None,  # For making AI calls
        db_session=None,
        dry_run: bool = False
    ):
        """
        Initialize inquiry processor.
        
        Args:
            ai_classifier: AI classifier for validation/type/name extraction
            db_session: SQLAlchemy database session
            dry_run: If True, don't make actual changes
        """
        self.ai_classifier = ai_classifier
        self.db_session = db_session
        self.dry_run = dry_run
        self.template_loader = InquiryTemplateLoader()
        self.draft_generator = InquiryDraftGenerator(self.template_loader)
    
    def check_recent_inquiry_draft(
        self,
        from_address: str,
        inquiry_types: List[str],
        db_session=None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if we've already created a draft for this sender with overlapping inquiry types
        within the duplicate window.
        
        Args:
            from_address: Sender's email address
            inquiry_types: List of inquiry types for this email
            db_session: Database session to use
            
        Returns:
            Dict with info about recent duplicate, or None if no duplicate found
        """
        session = db_session or self.db_session
        if not session:
            return None
        
        try:
            from backend.core.database.models import EmailMetadata, Email
            
            # Look for recent inquiries from same sender
            # Need to join Email table to filter by from_address
            cutoff_date = datetime.utcnow() - timedelta(days=self.DUPLICATE_WINDOW_DAYS)
            
            recent_inquiries = session.query(EmailMetadata).join(
                Email, EmailMetadata.email_id == Email.id
            ).filter(
                Email.from_address == from_address,
                EmailMetadata.ai_category.like('inquiry-%'),
                EmailMetadata.draft_status.in_(['created', 'sent']),
                EmailMetadata.draft_created_at >= cutoff_date
            ).order_by(EmailMetadata.draft_created_at.desc()).all()
            
            if not recent_inquiries:
                return None
            
            # Check for overlapping inquiry types
            current_types = set(inquiry_types)
            
            for inquiry in recent_inquiries:
                if inquiry.inquiry_types:
                    past_types = set(inquiry.inquiry_types) if isinstance(inquiry.inquiry_types, list) else set()
                    overlap = current_types & past_types
                    
                    if overlap:
                        # Get the email's message_id through the relationship
                        email = session.query(Email).filter(Email.id == inquiry.email_id).first()
                        return {
                            'message_id': email.message_id if email else None,
                            'draft_created_at': inquiry.draft_created_at,
                            'draft_status': inquiry.draft_status,
                            'overlapping_types': list(overlap),
                            'past_types': list(past_types),
                            'days_ago': (datetime.utcnow() - inquiry.draft_created_at).days
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking for duplicate inquiries: {e}")
            return None
    
    def should_process_as_inquiry(self, email: ProcessedEmail) -> bool:
        """
        Check if email should be processed through inquiry pipeline.
        
        Args:
            email: Email to check
            
        Returns:
            True if email should be handled as inquiry
        """
        return is_inquiry_email(email.subject)
    
    async def classify_inquiry(
        self, 
        email: ProcessedEmail
    ) -> InquiryClassificationResult:
        """
        Classify an inquiry email using AI.
        
        Args:
            email: Email to classify
            
        Returns:
            InquiryClassificationResult
        """
        # First try to extract type from subject tags
        tag_types = extract_inquiry_type_from_tags(email.subject)
        
        # Build prompt
        system_prompt, user_prompt = build_inquiry_classification_prompt(
            subject=email.subject,
            body=email.body_text or email.body_markdown or "",
            from_display_name=email.from_name
        )
        
        # Run AI classification using model routing infrastructure
        try:
            import os
            from backend.core.ai.llm_config import get_model_config
            
            model_name = os.getenv('INQUIRY_MODEL', 'gpt-5-mini')
            provider, api_key, endpoint, api_version = get_model_config(model_name)
            
            if provider == "azure":
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=endpoint,
                    api_version=api_version or "2024-08-01-preview"
                )
            else:
                from openai import OpenAI
                client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            result = parse_inquiry_classification_response(response_text)
            
            # Use tag types if AI didn't determine types but returned valid
            if result.is_valid_inquiry and tag_types and not result.inquiry_types:
                result.inquiry_types = tag_types
            
            # If AI determined types, prefer them unless tags strongly indicate otherwise
            if result.is_valid_inquiry and not result.inquiry_types and tag_types:
                result.inquiry_types = tag_types
            
            logger.info(f"Inquiry classified: valid={result.is_valid_inquiry}, types={result.inquiry_types}, name={result.full_name}")
            return result
            
        except Exception as e:
            logger.error(f"AI inquiry classification failed: {e}")
            # Fallback: assume valid inquiry if tags are present
            return InquiryClassificationResult(
                is_valid_inquiry=True,
                full_name=validate_extracted_name(email.from_name),
                inquiry_types=tag_types or ["unknown"],
                confidence=0.5,
                reasoning=f"AI classification failed ({e}), using fallback"
            )
    
    async def process_inquiry(
        self,
        email: ProcessedEmail,
        imap_drafts_manager=None,
        repo=None,
        db_email=None
    ) -> Dict[str, Any]:
        """
        Process an inquiry email through the full pipeline.
        
        Args:
            email: Email to process
            imap_drafts_manager: IMAP drafts manager for creating drafts
            repo: EmailRepository for database updates
            db_email: Database email record (if exists)
            
        Returns:
            Processing result dictionary
        """
        result = {
            'is_inquiry': True,
            'inquiry_valid': False,
            'inquiry_types': [],
            'extracted_name': None,
            'draft_created': False,
            'draft_message_id': None,
            'moved_to_folder': None,
            'ai_category': None,
            'error': None
        }
        
        try:
            # Step 1: Classify the inquiry
            classification = await self.classify_inquiry(email)
            result['inquiry_valid'] = classification.is_valid_inquiry
            result['inquiry_types'] = classification.inquiry_types
            result['inquiry_confidence'] = classification.confidence
            result['inquiry_reasoning'] = classification.reasoning
            
            if not classification.is_valid_inquiry:
                logger.info(f"Email {email.uid} is NOT a valid inquiry: {classification.reasoning}")
                result['skip_normal_pipeline'] = False  # Process through normal pipeline
                return result
            
            # Step 2: Extract and validate name
            extracted_name = validate_extracted_name(classification.full_name)
            if not extracted_name:
                extracted_name = "prospective applicant"
            result['extracted_name'] = extracted_name
            
            # Step 3: Determine category
            ai_category = get_inquiry_category(classification.inquiry_types)
            result['ai_category'] = ai_category
            
            # Step 3.5: Check for duplicate inquiry (same sender, overlapping types, within 30 days)
            db_session_for_check = repo.db if repo else self.db_session
            duplicate = self.check_recent_inquiry_draft(
                from_address=email.from_address,
                inquiry_types=classification.inquiry_types,
                db_session=db_session_for_check
            )
            
            if duplicate:
                result['is_duplicate'] = True
                result['duplicate_info'] = duplicate
                logger.info(
                    f"Skipping draft for {email.from_address}: duplicate inquiry "
                    f"({duplicate['overlapping_types']}) from {duplicate['days_ago']} days ago"
                )
                # Still mark as processed inquiry, but no draft
                result['skip_normal_pipeline'] = True
                result['moved_to_folder'] = self.INQUIRY_FOLDER
                # Note: We'll still move the email but not create a draft
            else:
                result['is_duplicate'] = False
            
            if self.dry_run:
                if result.get('is_duplicate'):
                    logger.info(f"[DRY RUN] Would skip draft (duplicate): {ai_category}, name={extracted_name}")
                    result['draft_created'] = False
                else:
                    logger.info(f"[DRY RUN] Would process inquiry: {ai_category}, name={extracted_name}")
                    result['draft_created'] = True
                result['moved_to_folder'] = self.INQUIRY_FOLDER
                result['skip_normal_pipeline'] = True
                return result
            
            # Step 4: Generate and create draft (skip if duplicate)
            if imap_drafts_manager and not result.get('is_duplicate'):
                draft = self.draft_generator.generate_draft(
                    to_address=email.from_address,
                    original_subject=email.subject,
                    original_message_id=email.message_id,
                    full_name=extracted_name,
                    inquiry_types=classification.inquiry_types
                )
                
                # Create draft in IMAP
                draft_result = imap_drafts_manager.save_draft_with_message_id(
                    to_address=draft.to_address,
                    subject=draft.subject,
                    body=draft.body,
                    in_reply_to=draft.in_reply_to,
                    references=draft.references,
                    cc_addresses=draft.cc_addresses
                )
                
                if draft_result:
                    draft_uid, draft_message_id = draft_result
                    result['draft_created'] = True
                    result['draft_message_id'] = draft_message_id
                    result['draft_uid'] = draft_uid
                    logger.info(f"Created inquiry draft for {email.uid}: Message-ID={draft_message_id}")
            
            # Step 5: Move email to inquiries folder
            if imap_drafts_manager:
                # Ensure folder exists
                imap_drafts_manager.ensure_folder_exists(self.INQUIRY_FOLDER)
                
                # Move email (the email might be in INBOX or another folder)
                source_folder = email.folder if hasattr(email, 'folder') and email.folder else "INBOX"
                moved = imap_drafts_manager.move_email(
                    source_folder=source_folder,
                    uid=email.uid,
                    dest_folder=self.INQUIRY_FOLDER
                )
                if moved:
                    result['moved_to_folder'] = self.INQUIRY_FOLDER
                    logger.info(f"Moved inquiry email {email.uid} to {self.INQUIRY_FOLDER}")
            
            # Step 6: Update database
            if repo and db_email:
                try:
                    self._update_database(
                        repo=repo,
                        db_email=db_email,
                        classification=classification,
                        extracted_name=extracted_name,
                        ai_category=ai_category,
                        draft_message_id=result.get('draft_message_id'),
                        is_duplicate=result.get('is_duplicate', False),
                        duplicate_info=result.get('duplicate_info')
                    )
                except Exception as e:
                    logger.error(f"Failed to update database for inquiry: {e}")
                    result['db_error'] = str(e)
            
            result['skip_normal_pipeline'] = True
            
        except Exception as e:
            logger.error(f"Error processing inquiry {email.uid}: {e}", exc_info=True)
            result['error'] = str(e)
            result['skip_normal_pipeline'] = False  # Fall back to normal pipeline
        
        return result
    
    def _update_database(
        self,
        repo,
        db_email,
        classification: InquiryClassificationResult,
        extracted_name: str,
        ai_category: str,
        draft_message_id: Optional[str],
        is_duplicate: bool = False,
        duplicate_info: Optional[Dict[str, Any]] = None
    ):
        """Update database with inquiry information."""
        from backend.core.database.models import EmailMetadata
        
        # Get or create metadata
        metadata = self.db_session.query(EmailMetadata).filter(
            EmailMetadata.email_id == db_email.id
        ).first()
        
        if not metadata:
            metadata = EmailMetadata(email_id=db_email.id)
            self.db_session.add(metadata)
        
        # Update inquiry fields
        metadata.ai_category = ai_category
        metadata.ai_confidence = classification.confidence
        
        # Add duplicate info to reasoning if applicable
        if is_duplicate and duplicate_info:
            reasoning = classification.reasoning or ""
            duplicate_note = (
                f"[DUPLICATE: Skipped draft - same sender asked about "
                f"{duplicate_info.get('overlapping_types', [])} "
                f"{duplicate_info.get('days_ago', '?')} days ago]"
            )
            metadata.ai_reasoning = f"{duplicate_note} {reasoning}".strip()
        else:
            metadata.ai_reasoning = classification.reasoning
            
        metadata.inquiry_types = classification.inquiry_types
        metadata.extracted_name = extracted_name
        metadata.name_extraction_source = "ai"
        metadata.inquiry_classification_source = "tag+ai"
        
        # Update draft lifecycle fields
        if draft_message_id:
            metadata.draft_status = "created"
            metadata.draft_created_at = datetime.utcnow()
            metadata.draft_message_id = draft_message_id
            metadata.inquiry_message_id = db_email.message_id
        elif is_duplicate:
            # Mark as skipped due to duplicate
            metadata.draft_status = "skipped_duplicate"
        
        self.db_session.commit()
        logger.debug(f"Updated database for inquiry {db_email.id}")


async def process_inquiry_email(
    email: ProcessedEmail,
    ai_classifier=None,
    imap_drafts_manager=None,
    db_session=None,
    repo=None,
    db_email=None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to process an inquiry email.
    
    Args:
        email: Email to process
        ai_classifier: AI classifier instance
        imap_drafts_manager: IMAP drafts manager
        db_session: Database session
        repo: Email repository
        db_email: Database email record
        dry_run: If True, don't make changes
        
    Returns:
        Processing result dictionary
    """
    processor = InquiryProcessor(
        ai_classifier=ai_classifier,
        db_session=db_session,
        dry_run=dry_run
    )
    
    return await processor.process_inquiry(
        email=email,
        imap_drafts_manager=imap_drafts_manager,
        repo=repo,
        db_email=db_email
    )
