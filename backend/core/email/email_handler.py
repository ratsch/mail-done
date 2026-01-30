"""
Unified email handler that combines IMAP (reading) and SMTP (sending).
Handles replies and drafts with proper context from original emails.
"""
from typing import Optional, List
from datetime import datetime
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr, make_msgid, formatdate

from .models import ProcessedEmail, Email
from .imap_monitor import IMAPMonitor, IMAPConfig
from .smtp_sender import SMTPSender
from .annotations import AnnotationManager, EmailMetadata, EmailNote, IMAPKeywords

logger = logging.getLogger(__name__)


class EmailHandler:
    """
    Unified handler for email operations requiring both IMAP and SMTP context.
    Keeps IMAP (reading) and SMTP (sending) separate but coordinates them.
    """
    
    def __init__(self, 
                 imap_config: IMAPConfig,
                 smtp_config: SMTPSender,
                 imap_timeout: int = 30,
                 database=None):
        """
        Initialize email handler with both IMAP and SMTP.
        
        Args:
            imap_config: IMAP configuration for reading emails
            smtp_config: SMTP configuration for sending emails
            imap_timeout: IMAP connection timeout
            database: Optional database connection for metadata storage (Phase 2)
        """
        self.imap = IMAPMonitor(imap_config, timeout=imap_timeout)
        self.smtp = SMTPSender(smtp_config)
        self.annotations = AnnotationManager(imap_monitor=self.imap, database=database)
        
    def create_reply_draft(self,
                          original_email: ProcessedEmail,
                          reply_body: str,
                          drafts_folder: str = 'Drafts') -> bool:
        """
        Create a draft reply with full context from original email.
        Saves to IMAP Drafts folder (visible in Apple Mail).
        
        Args:
            original_email: The email being replied to (with all context)
            reply_body: Body of the reply (can be AI-generated)
            drafts_folder: Folder to save draft in
            
        Returns:
            True if draft created successfully
        """
        logger.info(f"Creating reply draft for email: {original_email.subject}")
        
        # Build reply subject (add Re: if not present)
        reply_subject = original_email.subject
        if not reply_subject.startswith('Re:'):
            reply_subject = f'Re: {reply_subject}'
        
        # Use save_draft from IMAP (drafts are stored via IMAP, not sent via SMTP)
        return self.imap.save_draft(
            to_address=original_email.from_address,
            subject=reply_subject,
            body=reply_body,
            original_message_id=original_email.message_id,
            original_references=original_email.references,  # Pass full References chain
            drafts_folder=drafts_folder
        )
    
    def send_reply(self,
                   original_email: ProcessedEmail,
                   reply_body: str,
                   cc_addresses: Optional[List[str]] = None,
                   from_name: Optional[str] = None) -> bool:
        """
        Send a reply email with proper threading and context.
        
        Args:
            original_email: The email being replied to
            reply_body: Body of the reply
            cc_addresses: Additional CC recipients
            from_name: Display name for sender
            
        Returns:
            True if sent successfully
        """
        logger.info(f"Sending reply to: {original_email.from_address}")
        
        # Build reply subject
        reply_subject = original_email.subject
        if not reply_subject.startswith('Re:'):
            reply_subject = f'Re: {reply_subject}'
        
        # Send via SMTP with threading context
        return self.smtp.send_reply(
            to_address=original_email.from_address,
            subject=reply_subject,
            body=reply_body,
            original_message_id=original_email.message_id,
            cc_addresses=cc_addresses,
            from_name=from_name
        )
    
    def attach_note_to_email(self,
                            email_uid: str,
                            note: str,
                            visible_in_mail: bool = False,
                            note_type: str = "user_note") -> bool:
        """
        Attach a note/annotation to an email.
        
        Args:
            email_uid: UID of email to annotate
            note: Note text (can be long, detailed)
            visible_in_mail: If True, also add simplified version as IMAP keyword
            note_type: Type of note (user_note, ai_summary, reminder, etc.)
            
        Returns:
            True if note attached successfully
        """
        # Always store full note in database/memory (private)
        self.annotations.add_user_note(
            email_uid=email_uid,
            note=note,
            visible_in_mail=visible_in_mail,
            tags=[note_type]
        )
        
        # Optionally add simple keyword to IMAP (visible in Mail.app)
        if visible_in_mail:
            # Create short label from note type
            simple_label = note_type.replace('_', '').capitalize()
            return self.annotations.add_simple_note(email_uid, simple_label)
        
        return True
    
    def add_ai_metadata(self,
                       email_uid: str,
                       summary: Optional[str] = None,
                       sentiment: Optional[str] = None,
                       action_items: Optional[List[str]] = None,
                       classification_reason: Optional[str] = None,
                       entities: Optional[dict] = None,
                       confidence: Optional[float] = None) -> bool:
        """
        Add AI-generated metadata to an email (PRIVATE - never in IMAP).
        
        Args:
            email_uid: Email UID
            summary: AI-generated summary
            sentiment: Detected sentiment
            action_items: Extracted action items
            classification_reason: Why classified this way
            entities: Extracted entities (people, orgs, dates)
            confidence: Classification confidence score
            
        Returns:
            True if metadata stored
        """
        metadata = EmailMetadata(
            email_uid=email_uid,
            ai_summary=summary,
            ai_sentiment=sentiment,
            ai_action_items=action_items or [],
            classification_reasoning=classification_reason,
            ai_entities=entities,
            classification_confidence=confidence,
            processed_at=datetime.now()
        )
        
        return self.annotations.add_detailed_metadata(metadata)
    
    def mark_sensitive_content(self,
                              email_uid: str,
                              contains_pii: bool = False,
                              pii_types: Optional[List[str]] = None) -> bool:
        """
        Mark email as containing sensitive/private information.
        This is PRIVATE metadata for security tracking.
        
        Args:
            email_uid: Email UID
            contains_pii: Whether email contains PII
            pii_types: Types of PII (ssn, credit_card, health_info, etc.)
            
        Returns:
            True if marked
        """
        if contains_pii:
            return self.annotations.mark_contains_pii(email_uid, pii_types)
        
        metadata = self.annotations.pending_metadata.get(
            email_uid, EmailMetadata(email_uid=email_uid)
        )
        metadata.contains_sensitive = True
        return self.annotations.add_detailed_metadata(metadata)
    
    def apply_color_to_email(self,
                            email_uid: str,
                            color: int,
                            reason: Optional[str] = None) -> bool:
        """
        Apply Apple Mail color label to an email.
        
        Args:
            email_uid: UID of email
            color: Color code (1-7, see AppleMailColor enum)
            reason: Optional reason for coloring (logged)
            
        Returns:
            True if color applied successfully
        """
        if reason:
            logger.info(f"Applying color {color} to UID {email_uid}: {reason}")
        
        return self.imap.apply_color_label(email_uid, color)
    
    def create_forward_draft(self,
                            original_email: ProcessedEmail,
                            forward_to: str,
                            forward_note: Optional[str] = None,
                            drafts_folder: str = 'Drafts') -> bool:
        """
        Create a draft for forwarding an email.
        
        Args:
            original_email: Email to forward
            forward_to: Recipient email address
            forward_note: Optional note to add before forwarded content
            drafts_folder: Folder to save draft in
            
        Returns:
            True if draft created successfully
        """
        logger.info(f"Creating forward draft for: {original_email.subject}")
        
        # Build forward subject
        fwd_subject = original_email.subject
        if not fwd_subject.startswith('Fwd:'):
            fwd_subject = f'Fwd: {fwd_subject}'
        
        # Build forward body with original email context
        forward_body = []
        
        if forward_note:
            forward_body.append(forward_note)
            forward_body.append("\n" + "-" * 50 + "\n")
        
        forward_body.append(f"From: {original_email.from_address}")
        forward_body.append(f"Date: {original_email.date}")
        forward_body.append(f"Subject: {original_email.subject}")
        forward_body.append(f"To: {', '.join(original_email.to_addresses)}")
        forward_body.append("\n")
        forward_body.append(original_email.body_markdown)
        
        if original_email.attachment_texts:
            forward_body.append("\n" + "-" * 50)
            forward_body.append("Attachments:")
            for att_text in original_email.attachment_texts:
                forward_body.append(att_text)
        
        # Save as draft via IMAP
        return self.imap.save_draft(
            to_address=forward_to,
            subject=fwd_subject,
            body='\n'.join(forward_body),
            original_message_id=None,  # Forward doesn't thread
            drafts_folder=drafts_folder
        )
    
    def connect_both(self):
        """Connect both IMAP and SMTP"""
        self.imap.connect()
        self.smtp.connect()
    
    def disconnect_both(self):
        """Disconnect both IMAP and SMTP"""
        self.imap.disconnect()
        self.smtp.disconnect()
    
    def __enter__(self):
        """Context manager support"""
        self.imap.connect()
        # SMTP connects on-demand
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.imap.disconnect()
        self.smtp.disconnect()

