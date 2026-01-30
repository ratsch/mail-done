"""
Admin Routes for Lab Application Review System

User management, decisions, audit log, settings
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status, Response
from sqlalchemy.orm import Session
import json
from sqlalchemy import func, and_, or_, desc
from uuid import UUID
from datetime import datetime, timedelta
import logging

from backend.core.database import get_db
from backend.core.database.models import (
    LabMember, ApplicationDecision, Email, EmailMetadata, AuditLog, SystemSettings
)
from backend.api.review_auth import get_current_admin, get_current_admin_hybrid
from backend.api.review_schemas import (
    UserResponse, CreateUserRequest, UpdateUserRequest,
    DecisionRequest, DecisionResponse, BatchDecisionRequest,
    BatchUserRequest, BatchUserResponse,
    AuditLogResponse, UpdateIsNotApplicationRequest
)
from backend.api.review_middleware import log_audit_event, log_security_event
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================================
# User Management
# ============================================================================

@router.get("/users", response_model=dict)
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """List all lab members (admin only, paginated)."""
    try:
        total = db.query(LabMember).count()
        
        users = db.query(LabMember).offset((page - 1) * limit).limit(limit).all()
        
        return {
            "items": [
                UserResponse(
                    id=u.id,
                    email=u.email,
                    full_name=u.full_name,
                    role=u.role,
                    can_review=u.can_review,
                    is_active=u.is_active,
                    avatar_url=u.avatar_url,
                    created_at=u.created_at
                )
                for u in users
            ],
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit
        }
    except Exception as e:
        logger.error(f"Error listing users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve users: {str(e)}"
        )


@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: CreateUserRequest,
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """Add new lab member (admin only)."""
    try:
        # Check if user already exists by email
        existing = db.query(LabMember).filter(LabMember.email == user_data.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="User with this email already exists")
        
        # Create user (gsuite_id will be set on first OAuth login)
        # Use None instead of empty string to avoid unique constraint violations
        # PostgreSQL allows multiple NULLs in unique constraints
        new_user = LabMember(
            email=user_data.email,
            full_name=user_data.full_name,
            role=user_data.role,
            can_review=user_data.can_review,
            is_active=True,
            avatar_url=user_data.avatar_url,
            gsuite_id=None  # Will be set on first login, use None instead of "" to avoid unique constraint
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Log admin action
        try:
            log_security_event(
                db=db,
                user_id=str(current_admin.id),
                event_type="admin_action",
                endpoint="/admin/users",
                details={"action": "create_user", "user_email": user_data.email}
            )
        except Exception as log_error:
            logger.warning(f"Failed to log security event for user creation: {log_error}")
        
        return UserResponse(
            id=new_user.id,
            email=new_user.email,
            full_name=new_user.full_name,
            role=new_user.role,
            can_review=new_user.can_review,
            is_active=new_user.is_active,
            avatar_url=new_user.avatar_url,
            created_at=new_user.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user {user_data.email}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    user_data: UpdateUserRequest,
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """Update user permissions (admin only)."""
    user = db.query(LabMember).filter(LabMember.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update fields
    if user_data.can_review is not None:
        user.can_review = user_data.can_review
    if user_data.role is not None:
        user.role = user_data.role
    if user_data.is_active is not None:
        user.is_active = user_data.is_active
    if user_data.avatar_url is not None:
        user.avatar_url = user_data.avatar_url
    
    db.commit()
    db.refresh(user)
    
    # Log admin action
    log_security_event(
        db=db,
        user_id=str(current_admin.id),
        event_type="admin_action",
        endpoint=f"/admin/users/{user_id}",
        details={"action": "update_user", "changes": user_data.dict(exclude_unset=True)}
    )
    
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        can_review=user.can_review,
        is_active=user.is_active,
        avatar_url=user.avatar_url,
        created_at=user.created_at
    )


@router.post("/users/batch", response_model=BatchUserResponse)
async def batch_create_users(
    batch_data: BatchUserRequest,
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """
    Create multiple users in a single API call.
    Optimizes "Sync with Google" feature by preventing multiple individual API calls.
    """
    created_count = 0
    failed_count = 0
    results = []
    
    for user_data in batch_data.users:
        try:
            # Check if user already exists
            existing = db.query(LabMember).filter(LabMember.email == user_data.email).first()
            if existing:
                results.append({
                    "email": user_data.email,
                    "status": "skipped",
                    "message": "User already exists"
                })
                continue
            
            # Create user
            # Use None instead of empty string to avoid unique constraint violations
            # PostgreSQL allows multiple NULLs in unique constraints
            new_user = LabMember(
                email=user_data.email,
                full_name=user_data.full_name,
                role=user_data.role,
                can_review=user_data.can_review,
                is_active=True,
                avatar_url=user_data.avatar_url,
                gsuite_id=None  # Will be set on first login, use None instead of "" to avoid unique constraint
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            created_count += 1
            results.append({
                "email": user_data.email,
                "status": "success",
                "user_id": str(new_user.id)
            })
            
        except Exception as e:
            db.rollback()
            failed_count += 1
            logger.error(f"Failed to create user {user_data.email}: {e}")
            results.append({
                "email": user_data.email,
                "status": "failed",
                "error": str(e)
            })
    
    # Log admin action
    log_security_event(
        db=db,
        user_id=str(current_admin.id),
        event_type="admin_action",
        endpoint="/admin/users/batch",
        details={
            "action": "batch_create_users",
            "created": created_count,
            "failed": failed_count
        }
    )
    
    return BatchUserResponse(
        created=created_count,
        failed=failed_count,
        results=results
    )


# ============================================================================
# Decisions
# ============================================================================

@router.post("/applications/{email_id}/decision", response_model=DecisionResponse)
async def make_decision(
    email_id: UUID,
    decision_data: DecisionRequest,
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """
    Make final decision on an application (admin only).
    
    Triggers Google Drive archiving.
    """
    # Verify application exists
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Application not found")
    
    metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == email_id).first()
    if not metadata or not metadata.ai_category.startswith('application-'):
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Get or create decision
    existing_decision = db.query(ApplicationDecision).filter(
        ApplicationDecision.email_id == email_id
    ).first()
    
    action_type = "decision_make"
    if existing_decision:
        existing_decision.decision = decision_data.decision
        existing_decision.notes = decision_data.notes
        existing_decision.admin_id = current_admin.id
        action_type = "decision_update"
        db.commit()
        db.refresh(existing_decision)
        decision_obj = existing_decision
    else:
        decision_obj = ApplicationDecision(
            email_id=email_id,
            admin_id=current_admin.id,
            decision=decision_data.decision,
            notes=decision_data.notes
        )
        db.add(decision_obj)
        db.commit()
        db.refresh(decision_obj)
    
    # Update application status
    metadata.application_status = "decided"
    metadata.application_status_updated_at = datetime.utcnow()
    
    # Note: Google Drive archiving is currently disabled
    # To re-enable, uncomment the archiving code below and ensure archive_application_to_gdrive function is available
    
    db.commit()
    
    # Log audit event
    try:
        log_audit_event(
            db=db,
            user_id=str(current_admin.id),
            email_id=str(email_id),
            action_type=action_type,
            action_details={
                "decision": decision_data.decision,
                "notes_length": len(decision_data.notes) if decision_data.notes else 0
            }
        )
    except Exception as e:
        logger.error(f"Failed to log decision: {e}")
    
    return DecisionResponse(
        id=decision_obj.id,
        email_id=decision_obj.email_id,
        admin_id=decision_obj.admin_id,
        admin_name=current_admin.full_name or current_admin.email,
        decision=decision_obj.decision,
        notes=decision_obj.notes,
        decided_at=decision_obj.decided_at
    )


@router.post("/applications/batch/decide", response_model=dict)
async def batch_decide(
    batch_data: BatchDecisionRequest,
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """
    Make batch decisions for multiple applications (admin only).
    
    Processes each application individually - partial success allowed.
    Each successful decision is committed immediately.
    """
    results = []
    errors = []
    
    for email_id in batch_data.email_ids:
        try:
            # Verify application exists
            email = db.query(Email).filter(Email.id == email_id).first()
            if not email:
                errors.append({"email_id": str(email_id), "error": "Not found"})
                continue
            
            metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == email_id).first()
            if not metadata or not metadata.ai_category.startswith('application-'):
                errors.append({"email_id": str(email_id), "error": "Not an application"})
                continue
            
            # Create or update decision
            decision = db.query(ApplicationDecision).filter(
                ApplicationDecision.email_id == email_id
            ).first()
            
            if decision:
                decision.decision = batch_data.decision
                decision.notes = batch_data.notes
                decision.admin_id = current_admin.id
            else:
                decision = ApplicationDecision(
                    email_id=email_id,
                    admin_id=current_admin.id,
                    decision=batch_data.decision,
                    notes=batch_data.notes
                )
                db.add(decision)
            
            # Update status
            metadata.application_status = "decided"
            metadata.application_status_updated_at = datetime.utcnow()
            
            # Note: Google Drive archiving is currently disabled
            # To re-enable, uncomment the archiving code below
            # try:
            #     archive_path = archive_application_to_gdrive(email, metadata, batch_data.decision, db)
            #     metadata.gdrive_archive_folder_id = archive_path.get('folder_id')
            #     metadata.gdrive_archive_path = archive_path.get('path')
            #     metadata.archived_at = datetime.utcnow()
            # except Exception as e:
            #     logger.error(f"Failed to archive {email_id}: {e}")
            
            # Commit this item's changes
            try:
                db.commit()
                results.append({"email_id": str(email_id), "status": "success"})
                
                # Log audit event
                log_audit_event(
                    db=db,
                    user_id=str(current_admin.id),
                    email_id=str(email_id),
                    action_type="decision_make",
                    action_details={"decision": batch_data.decision, "batch": True}
                )
            except Exception as commit_error:
                logger.error(f"Failed to commit decision for {email_id}: {commit_error}")
                db.rollback()
                errors.append({"email_id": str(email_id), "error": f"Commit failed: {str(commit_error)}"})
        except Exception as e:
            logger.error(f"Error processing {email_id}: {e}")
            errors.append({"email_id": str(email_id), "error": str(e)})
            # Rollback this specific item's changes
            db.rollback()
    
    return {
        "processed": len(results),
        "errors": len(errors),
        "results": results,
        "errors_detail": errors
    }


@router.patch("/applications/{email_id}/is-not-application", response_model=dict)
async def update_is_not_application(
    email_id: UUID,
    request_data: UpdateIsNotApplicationRequest,
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """
    Update the is_not_application flag for an application (admin only).
    
    This allows admins to correct misclassifications where an actual application
    was marked as is_not_application=true, or vice versa.
    """
    # Verify application exists
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Application not found")
    
    metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == email_id).first()
    if not metadata or not metadata.ai_category.startswith('application-'):
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Get or initialize category_metadata
    if not metadata.category_metadata:
        metadata.category_metadata = {}
    
    # Get or initialize red_flags
    if 'red_flags' not in metadata.category_metadata:
        metadata.category_metadata['red_flags'] = {}
    
    # Update the flag
    old_value = metadata.category_metadata['red_flags'].get('is_not_application', False)
    metadata.category_metadata['red_flags']['is_not_application'] = request_data.is_not_application
    
    # Update reason if provided
    if request_data.reason:
        metadata.category_metadata['is_not_application_reason'] = request_data.reason
    elif request_data.is_not_application:
        # If setting to True but no reason provided, keep existing reason or set default
        if 'is_not_application_reason' not in metadata.category_metadata:
            metadata.category_metadata['is_not_application_reason'] = "Manually updated by admin"
    else:
        # If setting to False, clear the reason
        metadata.category_metadata.pop('is_not_application_reason', None)
    
    # Update correct_category if setting to True
    if request_data.is_not_application:
        # If not already set, don't override existing correct_category
        if 'correct_category' not in metadata.category_metadata:
            metadata.category_metadata['correct_category'] = None  # Admin can set this separately if needed
    else:
        # If setting to False, clear correct_category
        metadata.category_metadata.pop('correct_category', None)
    
    db.commit()
    db.refresh(metadata)
    
    # Log audit event
    try:
        log_audit_event(
            db=db,
            user_id=str(current_admin.id),
            email_id=str(email_id),
            action_type="admin_update_is_not_application",
            action_details={
                "old_value": old_value,
                "new_value": request_data.is_not_application,
                "reason": request_data.reason
            }
        )
    except Exception as e:
        logger.error(f"Failed to log is_not_application update: {e}")
    
    return {
        "success": True,
        "email_id": str(email_id),
        "is_not_application": request_data.is_not_application,
        "message": f"is_not_application flag updated from {old_value} to {request_data.is_not_application}"
    }


def archive_application_to_gdrive(email: Email, metadata: EmailMetadata, decision: str, db: Session) -> dict:
    """
    Archive application files to Google Drive.
    
    Structure: archive/{application-type}/{YYYY-MM}/{applicant-name}/
    
    Moves the original GDrive folder to the archive location.
    """
    # Extract application type from category (e.g., "application-phd" -> "phd")
    app_type = metadata.ai_category.replace('application-', '') if metadata.ai_category else 'other'
    
    # Get applicant name (sanitize for folder name)
    applicant_name = metadata.applicant_name or "unknown"
    applicant_name = "".join(c for c in applicant_name if c.isalnum() or c in (' ', '-', '_')).strip()
    applicant_name = applicant_name.replace(' ', '_')[:50]  # Limit length
    
    # Create folder path: archive/{type}/{YYYY-MM}/{name}
    now = datetime.utcnow()
    folder_path_parts = ["archive", app_type, now.strftime('%Y-%m'), applicant_name]
    
    # Get archive root folder ID from settings
    archive_root_id = None
    setting = db.query(SystemSettings).filter(SystemSettings.key == "gdrive_archive_root_id").first()
    if setting and isinstance(setting.value, dict):
        archive_root_id = setting.value.get("value")
    
    if not archive_root_id:
        logger.warning("GDrive archive root folder not configured. Skipping archiving.")
        return {
            "folder_id": None,
            "path": "/".join(folder_path_parts)
        }
    
    # Get DriveClient configuration
    service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
    if not service_account_path or not os.path.exists(service_account_path):
        logger.warning("Google service account not configured. Skipping archiving.")
        return {
            "folder_id": None,
            "path": "/".join(folder_path_parts)
        }
    
    try:
        from backend.core.google.drive_client import GoogleDriveClient
        
        # Initialize DriveClient with archive root
        drive_client = GoogleDriveClient(service_account_path, archive_root_id)
        
        # Get or create folder structure
        archive_folder_id = drive_client.get_or_create_folder_structure(folder_path_parts)
        
        # Move original folder to archive location
        original_folder_id = metadata.google_drive_folder_id
        if original_folder_id:
            try:
                # Get current parents
                file = drive_client.drive_service.files().get(
                    fileId=original_folder_id,
                    fields='parents',
                    supportsAllDrives=True
                ).execute()
                
                previous_parents = ",".join(file.get('parents', []))
                
                # Move file to archive folder
                drive_client.drive_service.files().update(
                    fileId=original_folder_id,
                    addParents=archive_folder_id,
                    removeParents=previous_parents,
                    fields='id, parents',
                    supportsAllDrives=True
                ).execute()
                
                logger.info(f"Archived application {email.id} to {archive_folder_id}")
                
                return {
                    "folder_id": archive_folder_id,
                    "path": "/".join(folder_path_parts)
                }
            except Exception as e:
                logger.error(f"Failed to move folder {original_folder_id} to archive: {e}")
                # Still return the archive folder ID even if move failed
                return {
                    "folder_id": archive_folder_id,
                    "path": "/".join(folder_path_parts)
                }
        else:
            # No original folder to move, just create archive structure
            return {
                "folder_id": archive_folder_id,
                "path": "/".join(folder_path_parts)
            }
            
    except Exception as e:
        logger.error(f"GDrive archiving failed: {e}")
        return {
            "folder_id": None,
            "path": "/".join(folder_path_parts)
        }


# ============================================================================
# Old Applications
# ============================================================================

@router.get("/old-applications", response_model=dict)
async def list_old_applications(
    days: int = Query(30, ge=1, description="Applications older than X days"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """List applications older than X days without decision (admin only)."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(Email, EmailMetadata).join(
        EmailMetadata, Email.id == EmailMetadata.email_id
    ).outerjoin(
        ApplicationDecision, Email.id == ApplicationDecision.email_id
    ).filter(
        EmailMetadata.ai_category.like('application-%'),
        Email.date < cutoff_date,
        ApplicationDecision.id.is_(None)  # No decision
    )
    
    total = query.count()
    results = query.order_by(Email.date.asc()).offset((page - 1) * limit).limit(limit).all()
    
    items = []
    for email, metadata in results:
        days_old = (datetime.utcnow() - email.date).days
        items.append({
            "email_id": str(email.id),
            "applicant_name": metadata.applicant_name,
            "applicant_institution": metadata.applicant_institution,
            "received_date": email.date.isoformat(),
            "days_old": days_old,
            "category": metadata.ai_category,
            "scientific_excellence_score": metadata.category_metadata.get('scientific_excellence_score') if metadata.category_metadata else None,
            "research_fit_score": metadata.research_fit_score,
            "overall_recommendation_score": metadata.overall_recommendation_score
        })
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit,
        "cutoff_days": days
    }


# ============================================================================
# Audit Log
# ============================================================================

@router.get("/audit-log", response_model=dict)
async def get_audit_log(
    user_id: Optional[UUID] = Query(None),
    email_id: Optional[UUID] = Query(None),
    action_type: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """Get audit log with filters (admin only)."""
    query = db.query(AuditLog, LabMember, EmailMetadata).join(
        LabMember, AuditLog.user_id == LabMember.id
    ).join(
        EmailMetadata, AuditLog.email_id == EmailMetadata.email_id
    )
    
    # Apply filters
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if email_id:
        query = query.filter(AuditLog.email_id == email_id)
    if action_type:
        query = query.filter(AuditLog.action_type == action_type)
    if start_date:
        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            query = query.filter(AuditLog.created_at >= start)
        except ValueError:
            pass
    if end_date:
        try:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            query = query.filter(AuditLog.created_at <= end)
        except ValueError:
            pass
    
    total = query.count()
    results = query.order_by(desc(AuditLog.created_at)).offset((page - 1) * limit).limit(limit).all()
    
    items = []
    for audit, user, metadata in results:
        items.append(AuditLogResponse(
            id=audit.id,
            user_id=audit.user_id,
            user_name=user.full_name or user.email,
            email_id=audit.email_id,
            applicant_name=metadata.applicant_name,
            action_type=audit.action_type,
            action_details=audit.action_details,
            ip_address=audit.ip_address,
            created_at=audit.created_at
        ))
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit
    }


# ============================================================================
# Settings
# ============================================================================

@router.get("/audit-log/export")
async def export_audit_log(
    format: str = Query("csv", description="Export format: excel or csv"),
    user_id: Optional[UUID] = Query(None),
    email_id: Optional[UUID] = Query(None),
    action_type: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """Export audit log to CSV or Excel (admin only)."""
    # Build query (same as get_audit_log)
    query = db.query(AuditLog, LabMember, EmailMetadata).join(
        LabMember, AuditLog.user_id == LabMember.id
    ).join(
        EmailMetadata, AuditLog.email_id == EmailMetadata.email_id
    )
    
    # Apply filters
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if email_id:
        query = query.filter(AuditLog.email_id == email_id)
    if action_type:
        query = query.filter(AuditLog.action_type == action_type)
    if start_date:
        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            query = query.filter(AuditLog.created_at >= start)
        except ValueError:
            pass
    if end_date:
        try:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            query = query.filter(AuditLog.created_at <= end)
        except ValueError:
            pass
    
    # Limit to 10,000 rows
    results = query.order_by(desc(AuditLog.created_at)).limit(10000).all()
    
    if len(results) >= 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Export limit exceeded. Maximum 10,000 rows. Please use filters to reduce the dataset."
        )
    
    # Prepare export data
    export_data = []
    for audit, user, metadata in results:
        row = {
            "timestamp": audit.created_at.isoformat(),
            "user_email": user.email,
            "user_name": user.full_name,
            "applicant_name": metadata.applicant_name,
            "email_id": str(audit.email_id),
            "action_type": audit.action_type,
            "ip_address": audit.ip_address,
            "user_agent": audit.user_agent,
            "action_details": json.dumps(audit.action_details) if audit.action_details else None
        }
        export_data.append(row)
    
    # Generate filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_log_export_{timestamp}_{current_admin.email.split('@')[0]}"
    
    if format.lower() == "csv":
        import csv
        import io
        
        output = io.StringIO()
        if export_data:
            writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
            writer.writeheader()
            writer.writerows(export_data)
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}.csv"',
                "X-Export-Metadata": json.dumps({
                    "exported_by": current_admin.email,
                    "exported_at": datetime.utcnow().isoformat(),
                    "row_count": len(export_data)
                })
            }
        )
    else:  # Excel
        try:
            import pandas as pd
            from io import BytesIO
            
            df = pd.DataFrame(export_data)
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Audit Log')
            
            output.seek(0)
            
            return Response(
                content=output.read(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}.xlsx"',
                    "X-Export-Metadata": json.dumps({
                        "exported_by": current_admin.email,
                        "exported_at": datetime.utcnow().isoformat(),
                        "row_count": len(export_data)
                    })
                }
            )
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Excel export requires pandas and openpyxl. Use CSV format or install dependencies."
            )


@router.get("/settings", response_model=dict)
async def get_settings(
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """Get all system settings (admin only)."""
    settings = db.query(SystemSettings).all()
    
    # Ensure category_thresholds setting exists with defaults
    category_thresholds_setting = db.query(SystemSettings).filter(
        SystemSettings.key == "category_thresholds"
    ).first()
    
    if not category_thresholds_setting:
        # Create default category_thresholds setting
        default_thresholds = {
            "application-phd": 7,
            "application-postdoc": 7,
            "application-internship": 7,
            "application-other": 6,
            "application-msc-thesis": 1,
        }
        category_thresholds_setting = SystemSettings(
            key="category_thresholds",
            value={"value": default_thresholds, "type": "object"},
            description="Minimum recommendation score thresholds for each application category",
            updated_by=current_admin.id
        )
        db.add(category_thresholds_setting)
        db.commit()
        db.refresh(category_thresholds_setting)
        # Add to settings list so it's included in response
        settings.append(category_thresholds_setting)
    
    return {
        "settings": {
            s.key: {
                "value": s.value.get("value") if isinstance(s.value, dict) else s.value,
                "type": s.value.get("type") if isinstance(s.value, dict) else "unknown",
                "description": s.description
            }
            for s in settings
        }
    }


@router.put("/settings/{key}", response_model=dict)
async def update_setting(
    key: str,
    value: dict,  # {"value": <actual_value>, "type": "integer|string|boolean|float|object"}
    current_admin: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db)
):
    """Update a system setting (admin only)."""
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    
    # If setting doesn't exist and it's category_thresholds, create it
    if not setting and key == "category_thresholds":
        setting = SystemSettings(
            key=key,
            value=value,
            description="Minimum recommendation score thresholds for each application category",
            updated_by=current_admin.id
        )
        db.add(setting)
    elif not setting:
        raise HTTPException(status_code=404, detail="Setting not found")
    else:
        setting.value = value
        setting.updated_at = datetime.utcnow()
        setting.updated_by = current_admin.id
    
    db.commit()
    db.refresh(setting)
    
    # Log admin action
    log_security_event(
        db=db,
        user_id=str(current_admin.id),
        event_type="admin_action",
        endpoint=f"/admin/settings/{key}",
        details={"action": "update_setting", "key": key, "value": value}
    )
    
    return {
        "key": setting.key,
        "value": setting.value,
        "description": setting.description,
        "updated_at": setting.updated_at.isoformat()
    }

