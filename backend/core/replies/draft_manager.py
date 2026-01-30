"""
Draft Manager

Coordinates draft reply generation using both templates and AI.
Manages draft lifecycle: create, store, retrieve, edit, send.
"""
import logging
from typing import List, Optional, Dict
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session

from backend.core.database.models import Email, EmailMetadata, SenderHistory, ReplyDraft as DBReplyDraft
from backend.core.replies.templates import ReplyTemplates
from backend.core.replies.ai_generator import AIReplyGenerator, ReplyDraft
from backend.core.email.smtp_sender import SMTPSender
from backend.core.email.imap_drafts import IMAPDraftsManager

logger = logging.getLogger(__name__)


class DraftManager:
    """Manage reply drafts: generation, storage, and lifecycle."""
    
    def __init__(
        self,
        db: Session,
        use_ai: bool = True,
        ai_model: str = "gpt-4o-mini",
        smtp_sender: Optional[SMTPSender] = None,
        imap_drafts: Optional[IMAPDraftsManager] = None
    ):
        """
        Initialize draft manager.
        
        Args:
            db: Database session
            use_ai: Whether to use AI generation (or templates only)
            ai_model: AI model to use for generation
            smtp_sender: SMTP sender for sending emails (optional)
            imap_drafts: IMAP drafts manager for saving to IMAP (optional)
        """
        self.db = db
        self.templates = ReplyTemplates()
        self.use_ai = use_ai
        self.smtp_sender = smtp_sender
        self.imap_drafts = imap_drafts
        
        if use_ai:
            self.ai_generator = AIReplyGenerator(model=ai_model)
        else:
            self.ai_generator = None
    
    async def generate_drafts(
        self,
        email_id: UUID,
        decision: str,  # accept/decline/maybe/acknowledge
        tone: str = "professional",
        num_variations: int = 2,
        use_templates: bool = True
    ) -> List[DBReplyDraft]:
        """
        Generate draft replies for an email.
        
        Combines template-based and AI-based generation to provide options.
        
        Args:
            email_id: Email to generate replies for
            decision: Response decision (accept/decline/maybe/acknowledge)
            tone: Desired tone
            num_variations: Number of variations to generate
            use_templates: Whether to include template-based drafts
        
        Returns:
            List of ReplyDraft database objects
        """
        # Get email and metadata
        email = self.db.query(Email).filter(Email.id == email_id).first()
        if not email:
            raise ValueError(f"Email {email_id} not found")
        
        metadata = email.email_metadata
        if not metadata:
            raise ValueError(f"Email {email_id} has no metadata")
        
        # Get sender history
        sender_history = self.db.query(SenderHistory).filter(
            SenderHistory.email_address == email.from_address
        ).first()
        
        drafts = []
        
        # 1. Generate template-based draft (fast, reliable)
        if use_templates:
            template_draft = self._generate_template_draft(
                email=email,
                metadata=metadata,
                decision=decision,
                tone=tone
            )
            if template_draft:
                drafts.append(template_draft)
        
        # 2. Generate AI-based drafts (personalized)
        if self.use_ai and self.ai_generator:
            ai_drafts = await self._generate_ai_drafts(
                email=email,
                metadata=metadata,
                decision=decision,
                tone=tone,
                sender_history=sender_history,
                num_variations=num_variations
            )
            drafts.extend(ai_drafts)
        
        # Save all drafts to database
        saved_drafts = []
        for i, draft in enumerate(drafts, 1):
            db_draft = DBReplyDraft(
                email_id=email.id,
                subject=draft.subject,
                body=draft.body,
                tone=draft.tone,
                option_number=i,
                generated_by=draft.generated_by,
                template_used=draft.template_used,
                model_used=draft.model_used,
                confidence=draft.confidence,
                reasoning=draft.reasoning,
                decision=decision,
                category=metadata.ai_category,
                status='draft'
            )
            self.db.add(db_draft)
            saved_drafts.append(db_draft)
        
        self.db.commit()
        
        # Refresh to get IDs
        for draft in saved_drafts:
            self.db.refresh(draft)
        
        return saved_drafts
    
    def _generate_template_draft(
        self,
        email: Email,
        metadata: EmailMetadata,
        decision: str,
        tone: str
    ) -> Optional[ReplyDraft]:
        """Generate draft using template system."""
        category = metadata.ai_category or "general"
        
        # Build context for template variables
        context = self._build_template_context(email, metadata)
        
        # Get template
        body = self.templates.generate_from_template(
            category=category,
            decision=decision,
            tone=tone,
            context=context
        )
        
        if not body:
            # Try generic template
            body = self.templates.generate_from_template(
                category="general",
                decision="acknowledge",
                tone="confirm",
                context=context
            )
        
        if not body:
            return None
        
        # Get subject
        subject = self.templates.get_subject(
            category=category,
            decision=decision,
            original_subject=email.subject
        )
        
        template_key = f"{category}-{decision}"
        
        return ReplyDraft(
            subject=subject,
            body=body,
            tone=tone,
            confidence=0.9,  # Templates are reliable
            reasoning=f"Generated from template: {template_key}",
            generated_by="template",
            template_used=template_key,
            model_used=None
        )
    
    async def _generate_ai_drafts(
        self,
        email: Email,
        metadata: EmailMetadata,
        decision: str,
        tone: str,
        sender_history: Optional[SenderHistory],
        num_variations: int
    ) -> List[ReplyDraft]:
        """Generate drafts using AI."""
        try:
            ai_drafts = await self.ai_generator.generate_reply(
                email=email,
                metadata=metadata,
                decision=decision,
                tone=tone,
                sender_history=sender_history,
                num_variations=num_variations
            )
            
            # Add metadata
            for draft in ai_drafts:
                draft.generated_by = "ai"
                draft.model_used = self.ai_generator.model
                # Fix subject line (AI doesn't include original subject)
                draft.subject = f"Re: {email.subject}"
            
            return ai_drafts
            
        except Exception as e:
            logger.error(f"Error generating AI drafts: {e}", exc_info=True)
            return []
    
    def _build_template_context(
        self,
        email: Email,
        metadata: EmailMetadata
    ) -> Dict[str, str]:
        """Build context dictionary for template variable substitution."""
        # Extract sender name
        sender_name = email.from_name or email.from_address.split('@')[0]
        
        # Get category-specific data
        category_data = metadata.category_specific_data or {}
        
        context = {
            'original_subject': email.subject,
            'sender_name': sender_name,
            'applicant_name': sender_name,  # Alias
            'colleague_name': sender_name,  # Alias
            'student_name': sender_name,  # Alias
            'organizer_name': sender_name,  # Alias
            'editor_name': sender_name,  # Alias
            'contact_name': sender_name,  # Alias
            
            # Generic placeholders
            'research_area': 'your research area',
            'specific_detail': 'your background',
            'event_name': 'the event',
            'event_date': 'the scheduled date',
            'topic': 'the topic',
            'response': 'I acknowledge your email',
            'timeline': 'soon',
            'timeframe': 'in the coming days',
        }
        
        # Add category-specific data if available
        if category_data:
            for key, value in category_data.items():
                if isinstance(value, str):
                    context[key] = value
        
        # Extract from AI summary if available
        if metadata.ai_summary:
            # Try to extract key info from summary
            summary = metadata.ai_summary.lower()
            
            if 'machine learning' in summary or 'ml' in summary or 'ai' in summary:
                context['research_area'] = 'machine learning'
            elif 'genomics' in summary or 'biology' in summary:
                context['research_area'] = 'computational biology'
        
        return context
    
    def get_drafts_for_email(self, email_id: UUID) -> List[DBReplyDraft]:
        """Get all drafts for an email."""
        return self.db.query(DBReplyDraft).filter(
            DBReplyDraft.email_id == email_id
        ).order_by(DBReplyDraft.option_number).all()
    
    def get_draft(self, draft_id: UUID) -> Optional[DBReplyDraft]:
        """Get a specific draft."""
        return self.db.query(DBReplyDraft).filter(
            DBReplyDraft.id == draft_id
        ).first()
    
    def update_draft(
        self,
        draft_id: UUID,
        subject: Optional[str] = None,
        body: Optional[str] = None
    ) -> Optional[DBReplyDraft]:
        """
        Edit a draft.
        
        Args:
            draft_id: Draft to edit
            subject: New subject (optional)
            body: New body (optional)
        
        Returns:
            Updated draft or None
        """
        draft = self.get_draft(draft_id)
        if not draft:
            return None
        
        if subject is not None:
            draft.subject = subject
            draft.edited_by_user = True
        
        if body is not None:
            draft.body = body
            draft.edited_by_user = True
        
        draft.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(draft)
        
        return draft
    
    def send_draft(
        self,
        draft_id: UUID,
        use_smtp: bool = True,
        delete_imap_draft: bool = True
    ) -> Optional[str]:
        """
        Send a draft via SMTP and update tracking.
        
        Args:
            draft_id: Draft to send
            use_smtp: Actually send via SMTP (if False, just marks as sent)
            delete_imap_draft: Delete from IMAP Drafts folder after sending
        
        Returns:
            Message-ID of sent email, or None if failed
        """
        draft = self.get_draft(draft_id)
        if not draft:
            logger.error(f"Draft {draft_id} not found")
            return None
        
        # Get original email for threading
        email = self.db.query(Email).filter(Email.id == draft.email_id).first()
        if not email:
            logger.error(f"Original email {draft.email_id} not found")
            return None
        
        message_id = None
        
        if use_smtp and self.smtp_sender:
            try:
                # Send via SMTP
                message_id = self.smtp_sender.send_reply(
                    original_message_id=email.message_id or "",
                    original_subject=email.subject,
                    original_references=email.references,
                    to_address=email.from_address,
                    subject=draft.subject,
                    body=draft.body
                )
                
                if not message_id:
                    logger.error(f"Failed to send draft {draft_id} via SMTP")
                    return None
                
                logger.info(f"Draft {draft_id} sent successfully via SMTP: {message_id}")
                
            except Exception as e:
                logger.error(f"Error sending draft {draft_id}: {e}", exc_info=True)
                return None
        
        # Update draft status
        draft.status = 'sent'
        draft.sent_at = datetime.utcnow()
        if message_id:
            draft.imap_message_id = message_id
        
        # Delete from IMAP Drafts if it was saved there
        if delete_imap_draft and draft.imap_draft_uid and self.imap_drafts:
            try:
                self.imap_drafts.delete_draft(draft.imap_draft_uid)
                logger.info(f"Deleted draft UID {draft.imap_draft_uid} from IMAP")
            except Exception as e:
                logger.warning(f"Could not delete IMAP draft: {e}")
        
        # Mark original email as replied (via response tracker)
        try:
            from backend.core.tracking.response_tracker import ResponseTracker
            tracker = ResponseTracker(self.db)
            tracker.mark_replied(email.id, message_id)
            logger.info(f"Marked original email {email.id} as replied")
        except Exception as e:
            logger.warning(f"Could not mark email as replied: {e}")
        
        self.db.commit()
        self.db.refresh(draft)
        
        return message_id
    
    def save_draft_to_imap(
        self,
        draft_id: UUID
    ) -> Optional[str]:
        """
        Save a draft to IMAP Drafts folder.
        
        Args:
            draft_id: Draft to save
        
        Returns:
            IMAP UID of saved draft, or None if failed
        """
        if not self.imap_drafts:
            logger.warning("IMAP drafts manager not configured")
            return None
        
        draft = self.get_draft(draft_id)
        if not draft:
            logger.error(f"Draft {draft_id} not found")
            return None
        
        # Get original email for threading
        email = self.db.query(Email).filter(Email.id == draft.email_id).first()
        if not email:
            logger.error(f"Original email {draft.email_id} not found")
            return None
        
        try:
            # Save to IMAP
            uid = self.imap_drafts.save_draft(
                to_address=email.from_address,
                subject=draft.subject,
                body=draft.body,
                in_reply_to=email.message_id,
                references=email.references
            )
            
            if uid:
                # Update draft with IMAP UID
                draft.imap_draft_uid = uid
                self.db.commit()
                logger.info(f"Draft {draft_id} saved to IMAP with UID {uid}")
            
            return uid
            
        except Exception as e:
            logger.error(f"Error saving draft to IMAP: {e}", exc_info=True)
            return None
    
    def mark_draft_sent(
        self,
        draft_id: UUID,
        message_id: Optional[str] = None
    ) -> Optional[DBReplyDraft]:
        """
        Mark draft as sent (for manual sending workflow).
        
        Args:
            draft_id: Draft that was sent
            message_id: Message-ID of sent email
        
        Returns:
            Updated draft or None
        """
        draft = self.get_draft(draft_id)
        if not draft:
            return None
        
        draft.status = 'sent'
        draft.sent_at = datetime.utcnow()
        
        if message_id:
            draft.imap_message_id = message_id
        
        self.db.commit()
        self.db.refresh(draft)
        
        return draft
    
    def discard_draft(self, draft_id: UUID) -> bool:
        """
        Mark draft as discarded.
        
        Args:
            draft_id: Draft to discard
        
        Returns:
            True if successful
        """
        draft = self.get_draft(draft_id)
        if not draft:
            return False
        
        draft.status = 'discarded'
        draft.updated_at = datetime.utcnow()
        
        self.db.commit()
        
        return True
    
    def delete_draft(self, draft_id: UUID) -> bool:
        """
        Delete a draft permanently.
        
        Args:
            draft_id: Draft to delete
        
        Returns:
            True if successful
        """
        draft = self.get_draft(draft_id)
        if not draft:
            return False
        
        self.db.delete(draft)
        self.db.commit()
        
        return True
    
    def get_drafts_by_status(
        self,
        status: str = 'draft',
        limit: int = 100
    ) -> List[DBReplyDraft]:
        """Get drafts by status."""
        return self.db.query(DBReplyDraft).filter(
            DBReplyDraft.status == status
        ).order_by(DBReplyDraft.created_at.desc()).limit(limit).all()

