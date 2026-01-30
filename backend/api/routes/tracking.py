"""
Response Tracking API Endpoints

Endpoints for managing reply tracking and unanswered emails.
Based on Phase 3 Response Tracking implementation.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime

import logging

from backend.api.auth import verify_api_key

logger = logging.getLogger(__name__)
from backend.core.database import get_db
from backend.core.tracking import ResponseTracker

router = APIRouter(prefix="/api/tracking", tags=["tracking"])


@router.get("/unanswered", dependencies=[Depends(verify_api_key)])
async def get_unanswered_emails(
    vip_only: bool = Query(False, description="Only VIP emails"),
    min_priority: int = Query(1, ge=1, le=10, description="Minimum priority level"),
    category: Optional[str] = Query(None, description="Filter by AI category"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    db: Session = Depends(get_db)
):
    """
    Get all unanswered important emails.
    
    **Query Parameters:**
    - vip_only: Only VIP emails
    - min_priority: Minimum priority level (1-10)
    - category: Filter by AI category
    - limit: Max results
    
    **Returns:**
    - List of emails with priority, time waiting, deadline
    - Sorted by: deadline → VIP → priority → date
    """
    tracker = ResponseTracker(db)
    
    unanswered = await tracker.get_unanswered_emails(
        vip_only=vip_only,
        min_priority=min_priority,
        category=category,
        limit=limit
    )
    
    return {
        "count": len(unanswered),
        "emails": unanswered,
        "filters": {
            "vip_only": vip_only,
            "min_priority": min_priority,
            "category": category
        }
    }


@router.get("/overdue", dependencies=[Depends(verify_api_key)])
async def get_overdue_replies(
    max_age_hours: int = Query(72, ge=1, description="Max hours without reply"),
    db: Session = Depends(get_db)
):
    """
    Get emails overdue for reply.
    
    **Overdue Criteria:**
    - Explicit deadline passed OR
    - No deadline but > max_age_hours old
    
    **Returns:**
    - List of overdue emails sorted by urgency
    """
    tracker = ResponseTracker(db)
    
    overdue = await tracker.get_overdue_replies(max_age_hours=max_age_hours)
    
    return {
        "count": len(overdue),
        "emails": overdue,
        "max_age_hours": max_age_hours
    }


@router.post("/{tracking_id}/mark-replied", dependencies=[Depends(verify_api_key)])
async def mark_email_replied(
    tracking_id: UUID,
    reply_email_id: Optional[UUID] = None,
    db: Session = Depends(get_db)
):
    """
    Mark an email as replied.
    
    **Updates:**
    - Sets replied_at timestamp
    - Sets needs_reply=False
    - Links to reply email if provided
    - Updates sender history (reply count, avg response time)
    """
    tracker = ResponseTracker(db)
    
    try:
        await tracker.mark_replied(tracking_id, reply_email_id)
        return {
            "status": "success",
            "tracking_id": str(tracking_id),
            "replied_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to mark as replied: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to mark as replied. Please try again."
        )


@router.post("/{tracking_id}/snooze", dependencies=[Depends(verify_api_key)])
async def snooze_reply_reminder(
    tracking_id: UUID,
    hours: int = Query(24, ge=1, le=168, description="Hours to snooze (max 1 week)"),
    db: Session = Depends(get_db)
):
    """
    Snooze reply reminder for specified hours.
    
    **Updates:**
    - Sets reminded_at to future time
    - Email won't appear in overdue list until snooze expires
    """
    tracker = ResponseTracker(db)
    
    try:
        new_deadline = await tracker.snooze_reply(tracking_id, hours)
        return {
            "status": "snoozed",
            "tracking_id": str(tracking_id),
            "snooze_hours": hours,
            "new_reminder_time": new_deadline.isoformat() if new_deadline else None
        }
    except Exception as e:
        logger.error(f"Failed to snooze: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to snooze. Please try again."
        )


@router.delete("/{tracking_id}/tracking", dependencies=[Depends(verify_api_key)])
async def remove_reply_tracking(
    tracking_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Remove reply tracking (mark as doesn't need reply).
    
    **Use case:** False positives - email doesn't actually need reply
    """
    tracker = ResponseTracker(db)
    
    try:
        await tracker.remove_tracking(tracking_id)
        return {
            "status": "removed",
            "tracking_id": str(tracking_id)
        }
    except Exception as e:
        logger.error(f"Failed to remove tracking: {str(e)}", exc_info=True)
        raise HTTPException(status_code=404, detail="Tracking not found.")


@router.get("/stats", dependencies=[Depends(verify_api_key)])
async def get_reply_stats(
    db: Session = Depends(get_db)
):
    """
    Get reply tracking statistics.
    
    **Returns:**
    - Total emails needing reply
    - Overdue count
    - Breakdown by category
    - Breakdown by VIP level
    - Average response time
    """
    tracker = ResponseTracker(db)
    
    stats = await tracker.get_reply_stats()
    
    return stats


