"""
Reply Generation API Endpoints

Provides API access to draft reply generation and management.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from uuid import UUID
from pydantic import BaseModel

import logging

from backend.api.auth import verify_api_key

logger = logging.getLogger(__name__)
from backend.core.database import get_db
from backend.core.database.models import ReplyDraft
from backend.core.replies import DraftManager


router = APIRouter(prefix="/api/replies", tags=["replies"])


class GenerateRepliesRequest(BaseModel):
    """Request to generate reply drafts"""
    decision: str  # accept/decline/maybe/acknowledge
    tone: str = "professional"  # professional/friendly/formal/enthusiastic/cautious
    num_variations: int = 2
    use_templates: bool = True
    use_ai: bool = True


class UpdateDraftRequest(BaseModel):
    """Request to update a draft"""
    subject: Optional[str] = None
    body: Optional[str] = None


class DraftResponse(BaseModel):
    """Draft response model"""
    id: str
    email_id: str
    subject: str
    body: str
    tone: str
    option_number: int
    generated_by: str
    template_used: Optional[str]
    model_used: Optional[str]
    confidence: float
    reasoning: str
    decision: str
    category: str
    status: str
    edited_by_user: bool
    created_at: str
    
    class Config:
        from_attributes = True


@router.post("/{email_id}/generate", dependencies=[Depends(verify_api_key)])
async def generate_replies(
    email_id: UUID,
    request: GenerateRepliesRequest,
    db: Session = Depends(get_db)
):
    """
    Generate reply drafts for an email.
    
    **Request Body:**
    ```json
    {
        "decision": "accept",  // accept/decline/maybe/acknowledge
        "tone": "professional",  // professional/friendly/formal
        "num_variations": 2,  // Number of AI variations (1-3)
        "use_templates": true,  // Include template-based draft
        "use_ai": true  // Include AI-generated drafts
    }
    ```
    
    **Returns:**
    - 2-3 draft variations
    - Template-based (fast, reliable)
    - AI-generated (personalized)
    - Each with confidence score
    
    **Example:**
    ```bash
    POST /api/replies/{email-id}/generate
    {
      "decision": "accept",
      "tone": "enthusiastic"
    }
    ```
    """
    # Validate decision
    valid_decisions = ['accept', 'decline', 'maybe', 'acknowledge']
    if request.decision not in valid_decisions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision. Must be one of: {', '.join(valid_decisions)}"
        )
    
    # Validate tone
    valid_tones = ['professional', 'friendly', 'formal', 'enthusiastic', 'cautious']
    if request.tone not in valid_tones:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tone. Must be one of: {', '.join(valid_tones)}"
        )
    
    # Create draft manager
    manager = DraftManager(
        db=db,
        use_ai=request.use_ai,
        ai_model="gpt-4o-mini"
    )
    
    try:
        # Generate drafts
        drafts = await manager.generate_drafts(
            email_id=email_id,
            decision=request.decision,
            tone=request.tone,
            num_variations=request.num_variations,
            use_templates=request.use_templates
        )
        
        # Convert to response format
        draft_responses = [
            DraftResponse(
                id=str(draft.id),
                email_id=str(draft.email_id),
                subject=draft.subject,
                body=draft.body,
                tone=draft.tone or request.tone,
                option_number=draft.option_number,
                generated_by=draft.generated_by or "unknown",
                template_used=draft.template_used,
                model_used=draft.model_used,
                confidence=draft.confidence or 0.5,
                reasoning=draft.reasoning or "",
                decision=draft.decision or request.decision,
                category=draft.category or "",
                status=draft.status,
                edited_by_user=draft.edited_by_user,
                created_at=draft.created_at.isoformat()
            )
            for draft in drafts
        ]
        
        return {
            "count": len(draft_responses),
            "drafts": draft_responses,
            "email_id": str(email_id),
            "decision": request.decision,
            "tone": request.tone
        }
        
    except ValueError as e:
        logger.error(f"ValueError in generate_replies: {str(e)}", exc_info=True)
        raise HTTPException(status_code=404, detail="Email not found or invalid request.")
    except Exception as e:
        logger.error(f"Error generating drafts: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error generating drafts. Please try again."
        )


@router.get("/{email_id}", dependencies=[Depends(verify_api_key)])
async def get_drafts_for_email(
    email_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get all drafts for an email.
    
    **Returns:**
    - All drafts (active and discarded)
    - Sorted by option_number
    - Includes edit history
    """
    manager = DraftManager(db=db, use_ai=False)  # No AI needed for retrieval
    
    drafts = manager.get_drafts_for_email(email_id)
    
    if not drafts:
        return {
            "count": 0,
            "drafts": [],
            "email_id": str(email_id)
        }
    
    draft_responses = [
        DraftResponse(
            id=str(draft.id),
            email_id=str(draft.email_id),
            subject=draft.subject,
            body=draft.body,
            tone=draft.tone or "professional",
            option_number=draft.option_number,
            generated_by=draft.generated_by or "unknown",
            template_used=draft.template_used,
            model_used=draft.model_used,
            confidence=draft.confidence or 0.5,
            reasoning=draft.reasoning or "",
            decision=draft.decision or "",
            category=draft.category or "",
            status=draft.status,
            edited_by_user=draft.edited_by_user,
            created_at=draft.created_at.isoformat()
        )
        for draft in drafts
    ]
    
    return {
        "count": len(draft_responses),
        "drafts": draft_responses,
        "email_id": str(email_id)
    }


@router.get("/drafts/{draft_id}", dependencies=[Depends(verify_api_key)])
async def get_draft(
    draft_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific draft by ID."""
    manager = DraftManager(db=db, use_ai=False)
    
    draft = manager.get_draft(draft_id)
    
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    return DraftResponse(
        id=str(draft.id),
        email_id=str(draft.email_id),
        subject=draft.subject,
        body=draft.body,
        tone=draft.tone or "professional",
        option_number=draft.option_number,
        generated_by=draft.generated_by or "unknown",
        template_used=draft.template_used,
        model_used=draft.model_used,
        confidence=draft.confidence or 0.5,
        reasoning=draft.reasoning or "",
        decision=draft.decision or "",
        category=draft.category or "",
        status=draft.status,
        edited_by_user=draft.edited_by_user,
        created_at=draft.created_at.isoformat()
    )


@router.put("/drafts/{draft_id}", dependencies=[Depends(verify_api_key)])
async def update_draft(
    draft_id: UUID,
    request: UpdateDraftRequest,
    db: Session = Depends(get_db)
):
    """
    Edit a draft before sending.
    
    **Request Body:**
    ```json
    {
        "subject": "Updated subject",  // optional
        "body": "Updated body text"  // optional
    }
    ```
    
    **Marks draft as edited_by_user=true**
    """
    manager = DraftManager(db=db, use_ai=False)
    
    draft = manager.update_draft(
        draft_id=draft_id,
        subject=request.subject,
        body=request.body
    )
    
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    return {
        "status": "updated",
        "draft_id": str(draft_id),
        "edited_by_user": draft.edited_by_user
    }


@router.post("/drafts/{draft_id}/send", dependencies=[Depends(verify_api_key)])
async def send_draft(
    draft_id: UUID,
    use_smtp: bool = Body(True, description="Actually send via SMTP"),
    save_to_imap_first: bool = Body(False, description="Save to IMAP Drafts before sending"),
    db: Session = Depends(get_db)
):
    """
    Send a draft via SMTP.
    
    **Request Body:**
    ```json
    {
        "use_smtp": true,  // Actually send via SMTP (requires SMTP config)
        "save_to_imap_first": false  // Save to IMAP Drafts before sending
    }
    ```
    
    **Two Modes:**
    1. **Full Auto** (`use_smtp=true`): Sends via SMTP, marks as sent, updates tracking
    2. **Manual** (`use_smtp=false`): Just marks as sent (you send externally)
    
    **What Happens:**
    - Sends email via SMTP with proper threading headers
    - Marks draft as sent in database
    - Marks original email as replied
    - Updates sender history
    - Optionally deletes from IMAP Drafts folder
    
    **Requires:** SMTP configuration (SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD)
    """
    # Import SMTP sender components
    from backend.core.email.smtp_sender import SMTPSender
    from backend.core.email.imap_drafts import IMAPDraftsManager
    
    # Initialize components if needed
    smtp_sender = SMTPSender() if use_smtp else None
    imap_drafts = IMAPDraftsManager() if save_to_imap_first else None
    
    manager = DraftManager(
        db=db,
        use_ai=False,
        smtp_sender=smtp_sender,
        imap_drafts=imap_drafts
    )
    
    # Save to IMAP first if requested
    if save_to_imap_first and imap_drafts:
        uid = manager.save_draft_to_imap(draft_id)
        if not uid:
            raise HTTPException(
                status_code=500,
                detail="Failed to save draft to IMAP Drafts folder"
            )
    
    # Send the draft
    if use_smtp:
        message_id = manager.send_draft(
            draft_id=draft_id,
            use_smtp=True,
            delete_imap_draft=save_to_imap_first  # Delete if we saved it first
        )
        
        if not message_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to send email via SMTP. Check SMTP configuration."
            )
        
        return {
            "status": "sent",
            "method": "smtp",
            "draft_id": str(draft_id),
            "message_id": message_id,
            "sent_at": datetime.utcnow().isoformat()
        }
    else:
        # Manual mode - just mark as sent
        draft = manager.mark_draft_sent(draft_id=draft_id)
        
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
        
        return {
            "status": "marked_sent",
            "method": "manual",
            "draft_id": str(draft_id),
            "sent_at": draft.sent_at.isoformat() if draft.sent_at else None,
            "note": "Draft marked as sent. You must send it manually."
        }


@router.post("/drafts/{draft_id}/save-to-imap", dependencies=[Depends(verify_api_key)])
async def save_draft_to_imap(
    draft_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Save a draft to IMAP Drafts folder.
    
    **Use Case:**
    - Generate draft in system
    - Save to IMAP Drafts
    - Edit/send from Mail.app
    
    **Returns:**
    - IMAP UID of saved draft
    
    **Requires:** IMAP configuration
    """
    from backend.core.email.imap_drafts import IMAPDraftsManager
    
    imap_drafts = IMAPDraftsManager()
    manager = DraftManager(db=db, use_ai=False, imap_drafts=imap_drafts)
    
    uid = manager.save_draft_to_imap(draft_id)
    
    if not uid:
        raise HTTPException(
            status_code=500,
            detail="Failed to save to IMAP. Check IMAP configuration."
        )
    
    return {
        "status": "saved",
        "draft_id": str(draft_id),
        "imap_uid": uid,
        "folder": "Drafts"
    }


@router.delete("/drafts/{draft_id}", dependencies=[Depends(verify_api_key)])
async def delete_draft(
    draft_id: UUID,
    discard_only: bool = Query(False, description="Just mark as discarded, don't delete"),
    db: Session = Depends(get_db)
):
    """
    Delete or discard a draft.
    
    **Query Parameters:**
    - discard_only: If true, marks as discarded but keeps in database
    - If false (default), permanently deletes
    
    **Use discard_only=true to keep history**
    """
    manager = DraftManager(db=db, use_ai=False)
    
    if discard_only:
        success = manager.discard_draft(draft_id)
        action = "discarded"
    else:
        success = manager.delete_draft(draft_id)
        action = "deleted"
    
    if not success:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    return {
        "status": action,
        "draft_id": str(draft_id)
    }


@router.get("/drafts", dependencies=[Depends(verify_api_key)])
async def list_drafts(
    status: str = Query("draft", description="Filter by status"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    List all drafts by status.
    
    **Query Parameters:**
    - status: draft/sent/discarded
    - limit: Maximum results
    
    **Use cases:**
    - Get all pending drafts: status=draft
    - See sent drafts: status=sent
    - Review discarded: status=discarded
    """
    manager = DraftManager(db=db, use_ai=False)
    
    drafts = manager.get_drafts_by_status(status=status, limit=limit)
    
    draft_responses = [
        DraftResponse(
            id=str(draft.id),
            email_id=str(draft.email_id),
            subject=draft.subject,
            body=draft.body[:200] + "..." if len(draft.body) > 200 else draft.body,  # Truncate for list
            tone=draft.tone or "professional",
            option_number=draft.option_number,
            generated_by=draft.generated_by or "unknown",
            template_used=draft.template_used,
            model_used=draft.model_used,
            confidence=draft.confidence or 0.5,
            reasoning=draft.reasoning or "",
            decision=draft.decision or "",
            category=draft.category or "",
            status=draft.status,
            edited_by_user=draft.edited_by_user,
            created_at=draft.created_at.isoformat()
        )
        for draft in drafts
    ]
    
    return {
        "count": len(draft_responses),
        "drafts": draft_responses,
        "status": status,
        "limit": limit
    }

