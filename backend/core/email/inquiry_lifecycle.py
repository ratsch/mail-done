"""
Inquiry Lifecycle Tracking

Tracks the lifecycle of inquiry draft responses:
- created: Draft exists in Drafts folder
- sent: User sent the draft (found in Sent Items)
- skipped: Draft was deleted/moved (user didn't want to send)
"""

import logging
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def update_draft_lifecycle(
    db_session: Session,
    imap_drafts_manager,
    sent_folder: str = "Sent Items"
) -> Dict[str, Any]:
    """
    Update the lifecycle status of all pending inquiry drafts.
    
    Checks each draft with status='created' to see if it was:
    - Sent (reply found in Sent Items)
    - Skipped (draft no longer exists in Drafts)
    - Still pending (draft still in Drafts)
    
    Args:
        db_session: SQLAlchemy database session
        imap_drafts_manager: IMAP drafts manager
        sent_folder: Name of Sent folder
        
    Returns:
        Dict with counts of updated statuses
    """
    from backend.core.database.models import EmailMetadata, Email
    
    result = {
        'checked': 0,
        'sent': 0,
        'skipped': 0,
        'still_pending': 0,
        'errors': 0
    }
    
    try:
        # Get all pending inquiry drafts
        pending_drafts = db_session.query(EmailMetadata, Email).join(
            Email, EmailMetadata.email_id == Email.id
        ).filter(
            EmailMetadata.draft_status == 'created',
            EmailMetadata.draft_message_id.isnot(None)
        ).all()
        
        logger.info(f"Checking lifecycle for {len(pending_drafts)} pending inquiry drafts...")
        
        for metadata, email in pending_drafts:
            result['checked'] += 1
            
            try:
                # Check if reply was sent (search Sent Items)
                sent_reply = imap_drafts_manager.search_sent_items_robust(
                    original_message_id=metadata.inquiry_message_id or email.message_id,
                    to_address=email.from_address,
                    subject_fragment=email.subject.replace("#info", "").replace("#application", "").strip(),
                    sent_folder=sent_folder
                )
                
                if sent_reply:
                    # Draft was sent
                    metadata.draft_status = 'sent'
                    result['sent'] += 1
                    logger.info(f"Draft for {email.from_address} was SENT (found by {sent_reply.get('found_by')})")
                    continue
                
                # Check if draft still exists
                draft_exists = imap_drafts_manager.draft_exists(metadata.draft_message_id)
                
                if not draft_exists:
                    # Draft was deleted/moved - user skipped it
                    metadata.draft_status = 'skipped'
                    result['skipped'] += 1
                    logger.info(f"Draft for {email.from_address} was SKIPPED (deleted from Drafts)")
                else:
                    # Draft still pending
                    result['still_pending'] += 1
                    logger.debug(f"Draft for {email.from_address} still PENDING")
                    
            except Exception as e:
                logger.error(f"Error checking draft lifecycle for {email.id}: {e}")
                result['errors'] += 1
        
        db_session.commit()
        
    except Exception as e:
        logger.error(f"Error updating draft lifecycle: {e}", exc_info=True)
        result['error'] = str(e)
    
    return result


def get_pending_drafts(
    db_session: Session
) -> List[Dict[str, Any]]:
    """
    Get all pending inquiry drafts for display.
    
    Returns:
        List of pending draft info dicts
    """
    from backend.core.database.models import EmailMetadata, Email
    
    pending = db_session.query(EmailMetadata, Email).join(
        Email, EmailMetadata.email_id == Email.id
    ).filter(
        EmailMetadata.draft_status == 'created',
        EmailMetadata.draft_message_id.isnot(None)
    ).order_by(EmailMetadata.draft_created_at.desc()).all()
    
    results = []
    for metadata, email in pending:
        # Generate Apple Mail link using the ORIGINAL email's Message-ID
        # This lets user see the original inquiry, then find the draft in Drafts folder
        # Format: message://<urlencoded-message-id>
        message_link = None
        if metadata.inquiry_message_id:
            msg_id = metadata.inquiry_message_id
            if not msg_id.startswith('<'):
                msg_id = f"<{msg_id}>"
            encoded_id = urllib.parse.quote(msg_id, safe='')
            message_link = f"message://{encoded_id}"
        
        results.append({
            'email_id': str(email.id),
            'from_address': email.from_address,
            'from_name': email.from_name,
            'subject': email.subject,
            'received_at': email.date,
            'draft_created_at': metadata.draft_created_at,
            'inquiry_types': metadata.inquiry_types,
            'extracted_name': metadata.extracted_name,
            'apple_mail_link': message_link,
            'draft_message_id': metadata.draft_message_id
        })
    
    return results


def format_pending_drafts_report(drafts: List[Dict[str, Any]]) -> str:
    """
    Format pending drafts as a CLI report.
    
    Args:
        drafts: List of pending draft dicts
        
    Returns:
        Formatted string for CLI output
    """
    if not drafts:
        return "✅ No pending inquiry drafts."
    
    lines = [
        "",
        "=" * 60,
        f"📝 PENDING INQUIRY DRAFTS ({len(drafts)})",
        "=" * 60,
        ""
    ]
    
    for i, draft in enumerate(drafts, 1):
        # Format date
        created = draft.get('draft_created_at')
        if isinstance(created, datetime):
            created_str = created.strftime('%Y-%m-%d %H:%M')
        else:
            created_str = str(created) if created else 'Unknown'
        
        # Format inquiry types
        types = draft.get('inquiry_types', [])
        types_str = ', '.join(types) if types else 'Unknown'
        
        lines.extend([
            f"{i}. {draft.get('from_name', 'Unknown')} <{draft.get('from_address', 'Unknown')}>",
            f"   Subject: {draft.get('subject', 'Unknown')[:60]}...",
            f"   Inquiry: {types_str}",
            f"   Draft created: {created_str}",
        ])
        
        # Add clickable link if available (option+click in Terminal)
        if draft.get('apple_mail_link'):
            lines.append(f"   📧 Open in Mail: {draft['apple_mail_link']}")
        
        lines.append("")
    
    lines.extend([
        "=" * 60,
        "💡 Option+click the link to view the original inquiry in Apple Mail",
        "   Then open Drafts folder to review/edit/send the response",
        "=" * 60
    ])
    
    return '\n'.join(lines)
