"""
Email management API endpoints
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from sqlalchemy.types import JSON
from uuid import UUID
from datetime import datetime, timedelta
import logging

from backend.api.auth import verify_api_key

logger = logging.getLogger(__name__)
from backend.api.schemas import (
    EmailResponse,
    EmailListResponse,
    EmailDetailResponse,
    UpdateMetadataRequest,
    StatsResponse,
    SenderStatsResponse
)
from backend.core.database import get_db
from backend.core.database.models import Email, EmailMetadata, Classification, SenderHistory, EmailLocationHistory
from backend.core.email.imap_actions import IMAPActionService
from backend.core.database.repository import EmailRepository
from sqlalchemy import Integer

router = APIRouter(prefix="/api/emails", tags=["emails"])


@router.get("", response_model=EmailListResponse, dependencies=[Depends(verify_api_key)])
async def list_emails(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    folder: Optional[str] = Query(None, description="Filter by folder"),
    vip_level: Optional[str] = Query(None, description="Filter by VIP level (urgent/high/medium)"),
    category: Optional[str] = Query(None, description="Filter by category"),
    needs_reply: Optional[bool] = Query(None, description="Filter emails needing reply"),
    is_flagged: Optional[bool] = Query(None, description="Filter flagged emails"),
    is_seen: Optional[bool] = Query(None, description="Filter seen/unseen emails"),
    search: Optional[str] = Query(None, description="Search in subject and body"),
    exclude_handled: bool = Query(False, description="Exclude emails marked as handled"),
    exclude_spam: bool = Query(True, description="Exclude spam-marked emails"),
    db: Session = Depends(get_db)
):
    """
    List emails with pagination and filters.
    
    **Filters:**
    - folder: INBOX, Sent, etc.
    - vip_level: urgent, high, medium
    - category: work, personal, invitation, review, etc.
    - needs_reply: true/false
    - is_flagged: true/false
    - is_seen: true/false (unread)
    - search: Search in subject/body
    
    **Returns paginated list of emails with metadata**
    """
    query = db.query(Email)
    
    # Join EmailMetadata once if we need any metadata filters
    needs_metadata_join = vip_level or category or (needs_reply is not None)
    if needs_metadata_join:
        query = query.join(EmailMetadata, Email.id == EmailMetadata.email_id, isouter=True)
    
    # Apply filters
    if folder:
        query = query.filter(Email.folder == folder)
    
    if is_flagged is not None:
        query = query.filter(Email.is_flagged == is_flagged)
    
    if is_seen is not None:
        query = query.filter(Email.is_seen == is_seen)
    
    if vip_level:
        query = query.filter(EmailMetadata.vip_level == vip_level)
    
    if category:
        query = query.filter(EmailMetadata.ai_category == category)
    
    if needs_reply is not None:
        query = query.filter(EmailMetadata.needs_reply == needs_reply)
    
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (Email.subject.ilike(search_pattern)) |
            (Email.body_markdown.ilike(search_pattern))
        )
    
    # Exclude handled emails if requested
    if exclude_handled:
        # Need to ensure EmailMetadata is joined
        if not needs_metadata_join:
            query = query.join(EmailMetadata, Email.id == EmailMetadata.email_id, isouter=True)
        # Exclude emails where 'handled' is in project_tags or user_tags
        # Cast JSON columns to JSONB for comparison
        from sqlalchemy import text
        query = query.filter(
            text("NOT (CAST(email_metadata.project_tags AS jsonb) @> '[\"handled\"]'::jsonb OR CAST(email_metadata.user_tags AS jsonb) @> '[\"handled\"]'::jsonb)")
        )
    
    # Exclude spam-marked emails if requested (default: true)
    if exclude_spam:
        # Need to ensure EmailMetadata is joined
        if not needs_metadata_join and not exclude_handled:
            query = query.join(EmailMetadata, Email.id == EmailMetadata.email_id, isouter=True)
        # Exclude emails where 'user-spam' is in user_tags
        from sqlalchemy import text
        query = query.filter(
            text("NOT CAST(email_metadata.user_tags AS jsonb) @> '[\"user-spam\"]'::jsonb")
        )
    
    # Get total count
    total = query.count()
    
    # Apply pagination and sorting (most recent first)
    offset = (page - 1) * page_size
    emails = query.order_by(desc(Email.date)).offset(offset).limit(page_size).all()
    
    # Calculate total pages
    pages = (total + page_size - 1) // page_size
    
    return EmailListResponse(
        emails=emails,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages
    )


@router.get("/{email_id}", response_model=EmailDetailResponse, dependencies=[Depends(verify_api_key)])
async def get_email(
    email_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get detailed email information including sender history.
    
    **Returns:**
    - Full email content
    - All metadata (VIP, AI classification, etc.)
    - Sender history and statistics
    - Classifications
    """
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Get sender history
    sender_history = db.query(SenderHistory).filter(
        SenderHistory.email_address == email.from_address
    ).first()
    
    # Build response
    response = EmailDetailResponse.model_validate(email)
    if sender_history:
        response.sender_history = sender_history
    
    return response


@router.put("/{email_id}/metadata", dependencies=[Depends(verify_api_key)])
async def update_email_metadata(
    email_id: UUID,
    updates: UpdateMetadataRequest,
    db: Session = Depends(get_db)
):
    """
    Update user-editable email metadata.
    
    **Editable fields:**
    - user_notes: Your personal notes
    - project_tags: Tags for organization
    - awaiting_reply: Mark as waiting for reply
    - needs_reply: Mark as needs response
    """
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Get or create metadata
    metadata = email.email_metadata
    if not metadata:
        metadata = EmailMetadata(email_id=email.id)
        db.add(metadata)
    
    # Update fields
    update_data = updates.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(metadata, field, value)
    
    metadata.updated_at = datetime.utcnow()
    
    try:
        db.commit()
        db.refresh(metadata)
        return {"status": "success", "metadata": metadata}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update metadata for {email_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update metadata. Please try again.")


@router.get("/folders/list", dependencies=[Depends(verify_api_key)])
async def list_folders(db: Session = Depends(get_db)):
    """
    List all folders with email counts.
    
    **Returns:**
    - List of folders
    - Email count per folder
    - Unread count per folder
    """
    folders = db.query(
        Email.folder,
        func.count(Email.id).label('total'),
        func.sum(func.cast(~Email.is_seen, Integer)).label('unread')
    ).group_by(Email.folder).all()
    
    return {
        "folders": [
            {
                "name": folder,
                "total": total,
                "unread": unread or 0
            }
            for folder, total, unread in folders
        ]
    }


@router.delete("/{email_id}", dependencies=[Depends(verify_api_key)])
async def delete_email(
    email_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete email from database (not from IMAP server).
    
    **Warning:** This only removes from local database, not from IMAP.
    """
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    try:
        db.delete(email)
        db.commit()
        return {"status": "deleted", "email_id": str(email_id)}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete email {email_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete email. Please try again.")


@router.get("/{email_id}/folder-history", dependencies=[Depends(verify_api_key)])
async def get_folder_history(
    email_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get the folder movement history for an email.
    
    **Returns:**
    - Complete timeline of folder movements
    - Who moved it (user/rule/ai/system)
    - When it was moved
    - How long it spent in each folder
    - Why it was moved (rule name, category, etc.)
    
    **Example Response:**
    ```json
    {
        "email_id": "123e4567-e89b-12d3-a456-426614174000",
        "current_folder": "INBOX",
        "total_moves": 3,
        "history": [
            {
                "from_folder": null,
                "to_folder": "INBOX",
                "moved_at": "2024-11-10T10:00:00",
                "moved_by": "system",
                "move_reason": "First discovered",
                "time_in_previous_folder_seconds": null
            },
            {
                "from_folder": "INBOX",
                "to_folder": "Applications/PhD",
                "moved_at": "2024-11-10T10:05:00",
                "moved_by": "rule",
                "move_reason": "Rule: PhD Application Filter",
                "time_in_previous_folder_seconds": 300
            }
        ]
    }
    ```
    """
    # Check if email exists
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Get folder history ordered by moved_at (newest first)
    history = db.query(EmailLocationHistory).filter(
        EmailLocationHistory.email_id == email_id
    ).order_by(desc(EmailLocationHistory.moved_at)).all()
    
    # Format the response
    return {
        "email_id": str(email_id),
        "current_folder": email.folder,
        "total_moves": len(history),
        "history": [
            {
                "from_folder": h.from_folder,
                "to_folder": h.to_folder,
                "moved_at": h.moved_at.isoformat() if h.moved_at else None,
                "moved_by": h.moved_by,
                "move_reason": h.move_reason,
                "time_in_previous_folder_seconds": h.time_in_previous_folder_seconds
            }
            for h in history
        ]
    }


@router.post("/{email_id}/mark-spam", dependencies=[Depends(verify_api_key)])
async def mark_email_as_spam(
    email_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Mark email as spam: move to spam folder, mark as read, add spam tag.
    
    **IMAP Actions:**
    - Moves email to configured spam folder (default: MD/Spam)
    - Marks as read (\\Seen flag)
    - Auto-creates folder if missing
    
    **Database Updates:**
    - Adds 'user-spam' tag to user_tags
    - Updates folder field
    - Tracks folder change in history
    
    **Returns:**
    - success: true/false
    - message: Status message
    - error: Error details (if failed)
    """
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Create service and repository
    service = IMAPActionService()
    repo = EmailRepository(db)
    
    try:
        success, message = await service.mark_as_spam(email, repo)
        
        if success:
            return {
                "success": True,
                "message": message,
                "email_id": str(email_id),
                "new_folder": email.folder
            }
        else:
            return {
                "success": False,
                "message": "IMAP operation failed",
                "error": message,
                "email_id": str(email_id)
            }
            
    except Exception as e:
        logger.error(f"Mark spam failed for {email_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to mark as spam: {str(e)}"
        )


@router.post("/{email_id}/delete", dependencies=[Depends(verify_api_key)])
async def delete_email_to_trash(
    email_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete email: move to trash folder, mark as read, mark as handled.
    
    **IMAP Actions:**
    - Moves email to trash folder (default: Trash)
    - Marks as read (\\Seen flag)
    - Auto-creates folder if missing
    
    **Database Updates:**
    - Updates folder field
    - Adds 'handled' tag to user_tags
    - Tracks folder change in history
    
    **Note:** This does NOT delete from database, only moves to Trash folder.
    Use DELETE /api/emails/{email_id} to remove from database.
    
    **Returns:**
    - success: true/false
    - message: Status message
    - error: Error details (if failed)
    """
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Create service and repository
    service = IMAPActionService()
    repo = EmailRepository(db)
    
    try:
        success, message = await service.delete_email(email, repo)
        
        if success:
            return {
                "success": True,
                "message": message,
                "email_id": str(email_id),
                "new_folder": email.folder
            }
        else:
            return {
                "success": False,
                "message": "IMAP operation failed",
                "error": message,
                "email_id": str(email_id)
            }
            
    except Exception as e:
        logger.error(f"Delete failed for {email_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete email: {str(e)}"
        )


@router.post("/{email_id}/archive", dependencies=[Depends(verify_api_key)])
async def archive_email_action(
    email_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Archive email: move to archive folder, mark as read.
    
    **IMAP Actions:**
    - Moves email to archive folder (default: Archive)
    - Marks as read (\\Seen flag)
    - Auto-creates folder if missing
    
    **Database Updates:**
    - Sets user_archived = True
    - Updates folder field
    - Tracks folder change in history
    
    **Returns:**
    - success: true/false
    - message: Status message
    - error: Error details (if failed)
    """
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Create service and repository
    service = IMAPActionService()
    repo = EmailRepository(db)
    
    try:
        success, message = await service.archive_email(email, repo)
        
        if success:
            return {
                "success": True,
                "message": message,
                "email_id": str(email_id),
                "new_folder": email.folder
            }
        else:
            return {
                "success": False,
                "message": "IMAP operation failed",
                "error": message,
                "email_id": str(email_id)
            }
            
    except Exception as e:
        logger.error(f"Archive failed for {email_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to archive email: {str(e)}"
        )


@router.post("/{email_id}/mark-handled", dependencies=[Depends(verify_api_key)])
async def mark_email_as_handled(
    email_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Mark email as handled: mark as read, clear flags, stay in current folder.
    
    **IMAP Actions:**
    - Marks as read (\\Seen flag)
    - Clears \\Flagged flag
    - Stays in current folder (no move)
    
    **Database Updates:**
    - Adds 'handled' tag to user_tags
    - Sets is_seen = True
    - Sets is_flagged = False
    - Keeps labels/metadata tags intact
    
    **Returns:**
    - success: true/false
    - message: Status message
    - error: Error details (if failed)
    """
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Create service and repository
    service = IMAPActionService()
    repo = EmailRepository(db)
    
    try:
        success, message = await service.mark_as_handled(email, repo)
        
        if success:
            return {
                "success": True,
                "message": message,
                "email_id": str(email_id),
                "folder": email.folder
            }
        else:
            return {
                "success": False,
                "message": "IMAP operation failed",
                "error": message,
                "email_id": str(email_id)
            }
            
    except Exception as e:
        logger.error(f"Mark handled failed for {email_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to mark as handled: {str(e)}"
        )

