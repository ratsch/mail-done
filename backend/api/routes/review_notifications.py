"""
Notifications Routes for Lab Application Review System
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from sqlalchemy import and_, or_
from datetime import datetime, timedelta
import logging

from backend.core.database import get_db
from backend.core.database.models import (
    Email, EmailMetadata, NotificationLog, LabMember
)
from backend.api.review_auth import get_current_user, get_current_user_hybrid
from backend.api.review_schemas import NotificationResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=List[NotificationResponse])
async def get_notifications(
    notification_type: Optional[str] = Query(None, description="Filter by type"),
    unread_only: bool = Query(True, description="Only unread notifications"),
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get notifications for current user.
    
    Types: new_application, deadline_approaching, deadline_passed
    """
    notifications = []
    
    # Get user's last login or last notification check
    last_check = current_user.last_login_at or datetime.utcnow() - timedelta(days=30)
    
    # New applications (since last login, not yet reviewed by user)
    if not notification_type or notification_type == "new_application":
        new_apps = db.query(Email, EmailMetadata).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).outerjoin(
            NotificationLog,
            and_(
                NotificationLog.email_id == Email.id,
                NotificationLog.lab_member_id == current_user.id,
                NotificationLog.notification_type == "new_application"
            )
        ).filter(
            EmailMetadata.ai_category.like('application-%'),
            Email.date > last_check,
            NotificationLog.id.is_(None)  # Not yet notified
        ).limit(20).all()
        
        for email, metadata in new_apps:
            # Check if user has reviewed
            from backend.core.database.models import ApplicationReview
            reviewed = db.query(ApplicationReview).filter(
                ApplicationReview.email_id == email.id,
                ApplicationReview.lab_member_id == current_user.id
            ).first()
            
            if not reviewed:
                notifications.append(NotificationResponse(
                    id=email.id,  # Using email_id as notification ID
                    email_id=email.id,
                    applicant_name=metadata.applicant_name,
                    category=metadata.ai_category,
                    notification_type="new_application",
                    deadline=None,
                    created_at=email.date
                ))
    
    # Deadline approaching (within 3 days)
    if not notification_type or notification_type == "deadline_approaching":
        three_days_from_now = datetime.utcnow() + timedelta(days=3)
        
        approaching = db.query(Email, EmailMetadata).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).outerjoin(
            NotificationLog,
            and_(
                NotificationLog.email_id == Email.id,
                NotificationLog.lab_member_id == current_user.id,
                NotificationLog.notification_type == "deadline_approaching"
            )
        ).filter(
            EmailMetadata.ai_category.like('application-%'),
            EmailMetadata.review_deadline.isnot(None),
            EmailMetadata.review_deadline <= three_days_from_now,
            EmailMetadata.review_deadline > datetime.utcnow(),
            NotificationLog.id.is_(None)  # Not yet notified
        ).limit(20).all()
        
        for email, metadata in approaching:
            # Check if user has reviewed
            from backend.core.database.models import ApplicationReview
            reviewed = db.query(ApplicationReview).filter(
                ApplicationReview.email_id == email.id,
                ApplicationReview.lab_member_id == current_user.id
            ).first()
            
            if not reviewed:
                notifications.append(NotificationResponse(
                    id=email.id,
                    email_id=email.id,
                    applicant_name=metadata.applicant_name,
                    category=metadata.ai_category,
                    notification_type="deadline_approaching",
                    deadline=metadata.review_deadline,
                    created_at=datetime.utcnow()
                ))
    
    # Deadline passed
    if not notification_type or notification_type == "deadline_passed":
        passed = db.query(Email, EmailMetadata).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).outerjoin(
            NotificationLog,
            and_(
                NotificationLog.email_id == Email.id,
                NotificationLog.lab_member_id == current_user.id,
                NotificationLog.notification_type == "deadline_passed"
            )
        ).filter(
            EmailMetadata.ai_category.like('application-%'),
            EmailMetadata.review_deadline.isnot(None),
            EmailMetadata.review_deadline < datetime.utcnow(),
            NotificationLog.id.is_(None)  # Not yet notified
        ).limit(20).all()
        
        for email, metadata in passed:
            # Check if user has reviewed
            from backend.core.database.models import ApplicationReview
            reviewed = db.query(ApplicationReview).filter(
                ApplicationReview.email_id == email.id,
                ApplicationReview.lab_member_id == current_user.id
            ).first()
            
            if not reviewed:
                notifications.append(NotificationResponse(
                    id=email.id,
                    email_id=email.id,
                    applicant_name=metadata.applicant_name,
                    category=metadata.ai_category,
                    notification_type="deadline_passed",
                    deadline=metadata.review_deadline,
                    created_at=datetime.utcnow()
                ))
    
    # Sort by created_at desc
    notifications.sort(key=lambda x: x.created_at, reverse=True)
    
    return notifications[:50]  # Limit to 50 notifications


@router.post("/mark-read")
async def mark_notifications_read(
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db)
):
    """
    Mark all unread notifications for the current user as read.
    """
    from datetime import datetime
    
    # Update all unread notifications for this user
    updated_count = db.query(NotificationLog).filter(
        NotificationLog.lab_member_id == current_user.id,
        NotificationLog.read_at.is_(None)
    ).update(
        {"read_at": datetime.utcnow()},
        synchronize_session=False
    )
    
    db.commit()
    
    return {
        "message": "Notifications marked as read",
        "marked_count": updated_count
    }


@router.put("/{email_id}/read")
async def mark_single_notification_read(
    email_id: UUID,
    notification_type: Optional[str] = Query(None, description="Notification type to mark as read"),
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db)
):
    """
    Mark a specific notification as read by email_id.
    If notification_type is provided, marks only that type; otherwise marks all types for that email.
    """
    from datetime import datetime
    
    query = db.query(NotificationLog).filter(
        NotificationLog.email_id == email_id,
        NotificationLog.lab_member_id == current_user.id
    )
    
    if notification_type:
        query = query.filter(NotificationLog.notification_type == notification_type)
    
    notifications = query.all()
    
    if not notifications:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    marked_count = 0
    for notification in notifications:
        if notification.read_at is None:
            notification.read_at = datetime.utcnow()
            marked_count += 1
    
    db.commit()
    
    return {
        "message": "Notification marked as read",
        "email_id": str(email_id),
        "marked_count": marked_count
    }

