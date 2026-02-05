"""
Review Assignment Routes

Endpoints for assigning applications to reviewers, tracking progress,
and managing assignment batches.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, case, and_, or_
from uuid import UUID
from datetime import datetime, timezone
import uuid as uuid_mod
import logging
import math


def _is_past(dt: Optional[datetime]) -> bool:
    """Check if a datetime is in the past, handling both naive and aware datetimes."""
    if dt is None:
        return False
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        now = now.replace(tzinfo=None)
    return dt < now

from backend.core.database import get_db
from backend.core.database.models import (
    Email, EmailMetadata, LabMember,
    AssignmentBatch, AssignmentBatchShare, ApplicationReviewAssignment,
)
from backend.api.review_auth import get_current_user_hybrid
from backend.api.review_schemas import (
    CreateAssignmentRequest, CreateAssignmentResponse, DuplicateAssignmentInfo,
    AddToBatchRequest,
    AssignmentResponse, AssignmentListResponse, PaginationInfo,
    BatchResponse, BatchListResponse, BatchDetailResponse, BatchStatsInfo,
    SharedWithInfo, UpdateBatchRequest,
    DeclineAssignmentRequest, ApplicationAssignmentResponse,
    PreviewBulkAssignmentRequest, PreviewBulkAssignmentResponse,
    PreviewApplicationInfo, PreviewDuplicateInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assignments", tags=["assignments"])


# ============================================================================
# Helper Functions
# ============================================================================

def _check_batch_permission(db: Session, batch: AssignmentBatch, user: LabMember, require_edit: bool = False):
    """
    Check if user has access to a batch.

    Creator and shared users have full access.
    Assignees have read access only (unless require_edit=True).

    Returns (has_access, can_edit).
    Raises 403 if require_edit and user cannot edit.
    Raises 404 if no access at all.
    """
    is_creator = batch.created_by == user.id
    is_shared = db.query(AssignmentBatchShare).filter(
        AssignmentBatchShare.batch_id == batch.id,
        AssignmentBatchShare.shared_with == user.id,
    ).first() is not None

    can_edit = is_creator or is_shared

    if can_edit:
        return True, True

    # Check if assignee
    is_assignee = db.query(ApplicationReviewAssignment).filter(
        ApplicationReviewAssignment.batch_id == batch.id,
        ApplicationReviewAssignment.assigned_to == user.id,
    ).first() is not None

    if is_assignee:
        if require_edit:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "permission_denied", "message": "You do not have permission to edit this batch"}
            )
        return True, False

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": "Batch not found"}
    )


def _get_batch_stats_map(db: Session, batch_ids: list) -> dict:
    """
    Get stats for multiple batches in a single query.
    Returns dict mapping batch_id -> BatchStatsInfo.
    """
    if not batch_ids:
        return {}

    rows = db.query(
        ApplicationReviewAssignment.batch_id,
        func.count().label('total'),
        func.sum(case((ApplicationReviewAssignment.status == 'pending', 1), else_=0)).label('pending'),
        func.sum(case((ApplicationReviewAssignment.status == 'completed', 1), else_=0)).label('completed'),
        func.sum(case((ApplicationReviewAssignment.status == 'declined', 1), else_=0)).label('declined'),
    ).filter(
        ApplicationReviewAssignment.batch_id.in_(batch_ids)
    ).group_by(ApplicationReviewAssignment.batch_id).all()

    result = {}
    for row in rows:
        result[row.batch_id] = BatchStatsInfo(
            total=row.total or 0,
            pending=row.pending or 0,
            completed=row.completed or 0,
            declined=row.declined or 0,
        )
    return result


def _build_batch_response(batch: AssignmentBatch, user: LabMember, stats: BatchStatsInfo, can_edit: bool) -> dict:
    """Build a batch response dict."""
    shared_with = []
    for share in batch.shares:
        shared_with.append(SharedWithInfo(
            id=share.shared_with,
            name=share.user.full_name if share.user else None,
        ))

    return BatchResponse(
        id=batch.id,
        notes=batch.notes,
        deadline=batch.deadline,
        created_by=batch.created_by,
        created_by_name=batch.creator.full_name if batch.creator else None,
        is_owner=batch.created_by == user.id,
        can_edit=can_edit,
        shared_with=shared_with,
        created_at=batch.created_at,
        updated_at=batch.updated_at,
        stats=stats,
    )


def _create_assignments_for_batch(
    db: Session, batch: AssignmentBatch, email_ids: list, assigned_to_ids: list
) -> tuple:
    """
    Create cross-product assignments for a batch.
    Returns (created_count, duplicates_list).
    """
    created = 0
    duplicates = []

    for eid in email_ids:
        # Look up applicant name for this email
        metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == eid).first()
        applicant_name = metadata.applicant_name if metadata else None

        for uid in assigned_to_ids:
            # Check for existing assignment (unique constraint)
            existing = db.query(ApplicationReviewAssignment).filter(
                ApplicationReviewAssignment.email_id == eid,
                ApplicationReviewAssignment.assigned_to == uid,
            ).first()

            if existing:
                # Get info about existing assignment for error response
                existing_batch = db.query(AssignmentBatch).filter(
                    AssignmentBatch.id == existing.batch_id
                ).first()
                assigner_name = None
                assignee_name = None
                if existing_batch and existing_batch.creator:
                    assigner_name = existing_batch.creator.full_name
                assignee_obj = db.query(LabMember).filter(LabMember.id == uid).first()
                if assignee_obj:
                    assignee_name = assignee_obj.full_name

                duplicates.append(DuplicateAssignmentInfo(
                    email_id=eid,
                    assigned_to=uid,
                    applicant_name=applicant_name,
                    assignee_name=assignee_name,
                    existing_batch_id=existing.batch_id,
                    existing_assigner=assigner_name,
                ))
                continue

            assignment = ApplicationReviewAssignment(
                email_id=eid,
                assigned_to=uid,
                batch_id=batch.id,
            )
            db.add(assignment)
            created += 1

    return created, duplicates


def complete_assignments_on_review(db: Session, email_id, reviewer_id):
    """
    Auto-complete pending assignments when a review is submitted.

    Does not commit — caller is responsible for transaction management.
    Returns list of completed assignments (for future notification use).
    """
    assignments = db.query(ApplicationReviewAssignment).filter(
        ApplicationReviewAssignment.email_id == email_id,
        ApplicationReviewAssignment.assigned_to == reviewer_id,
        ApplicationReviewAssignment.status == "pending"
    ).all()

    now = datetime.now(timezone.utc)
    for assignment in assignments:
        assignment.status = "completed"
        assignment.completed_at = now
        assignment.updated_at = now

    return assignments


# ============================================================================
# Endpoints
# ============================================================================

@router.post("", response_model=CreateAssignmentResponse)
async def create_assignments(
    request: CreateAssignmentRequest,
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """
    Create a batch of assignments. Creates cross-product of email_ids x assigned_to.
    Skips existing (email_id, assigned_to) pairs with detailed info.
    """
    # Validate total count
    total = len(request.email_ids) * len(request.assigned_to)
    if total > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "too_many_assignments", "message": f"Maximum 100 assignments per request, got {total}"}
        )

    # Validate all assigned_to users have can_review=true
    reviewers = db.query(LabMember).filter(
        LabMember.id.in_(request.assigned_to),
        LabMember.is_active == True,
    ).all()
    reviewer_map = {r.id: r for r in reviewers}

    for uid in request.assigned_to:
        if uid not in reviewer_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "user_not_found", "message": f"User {uid} not found or inactive"}
            )
        if not reviewer_map[uid].can_review:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "cannot_review", "message": f"User {reviewer_map[uid].full_name or uid} does not have review permission"}
            )

    # Validate email_ids exist and are applications
    emails_exist = db.query(Email.id).filter(Email.id.in_(request.email_ids)).all()
    existing_ids = {e.id for e in emails_exist}
    for eid in request.email_ids:
        if eid not in existing_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "application_not_found", "message": f"Application {eid} not found"}
            )

    # Validate share_with users exist
    if request.share_with:
        share_users = db.query(LabMember).filter(
            LabMember.id.in_(request.share_with),
            LabMember.is_active == True,
        ).all()
        share_user_ids = {u.id for u in share_users}
        for uid in request.share_with:
            if uid not in share_user_ids:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "share_user_not_found", "message": f"Share user {uid} not found or inactive"}
                )

    # Create batch
    batch = AssignmentBatch(
        created_by=current_user.id,
        notes=request.notes,
        deadline=request.deadline,
    )
    db.add(batch)
    db.flush()  # Get batch.id

    # Create shares
    if request.share_with:
        for uid in request.share_with:
            if uid != current_user.id:  # Don't share with self
                share = AssignmentBatchShare(
                    batch_id=batch.id,
                    shared_with=uid,
                )
                db.add(share)

    # Create assignments
    created, duplicates = _create_assignments_for_batch(
        db, batch, request.email_ids, request.assigned_to
    )

    if created == 0:
        # All duplicates — roll back batch creation
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "all_duplicates",
                "message": "All requested assignments already exist",
                "duplicates": [d.model_dump(mode='json') for d in duplicates],
            }
        )

    db.commit()

    return CreateAssignmentResponse(
        batch_id=batch.id,
        created=created,
        skipped_duplicates=len(duplicates),
        duplicates=duplicates,
    )


@router.get("", response_model=AssignmentListResponse)
async def list_my_assignments(
    status_filter: Optional[str] = Query(None, alias="status", description="pending|completed|declined|all"),
    batch_id: Optional[UUID] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """List current user's assignments (as assignee)."""
    # Base filter: assigned to me
    query = db.query(ApplicationReviewAssignment).filter(
        ApplicationReviewAssignment.assigned_to == current_user.id,
    )

    # Status filter (default: pending)
    if status_filter and status_filter != "all":
        query = query.filter(ApplicationReviewAssignment.status == status_filter)
    elif not status_filter:
        query = query.filter(ApplicationReviewAssignment.status == "pending")

    # Batch filter
    if batch_id:
        query = query.filter(ApplicationReviewAssignment.batch_id == batch_id)

    # Count total
    total_items = query.count()
    total_pages = max(1, math.ceil(total_items / page_size))

    # Paginate
    assignments = query.order_by(
        ApplicationReviewAssignment.created_at.desc()
    ).offset((page - 1) * page_size).limit(page_size).all()

    # Batch stats subquery for relevant batches
    batch_ids = list({a.batch_id for a in assignments})
    stats_map = _get_batch_stats_map(db, batch_ids)

    # Summary: total pending and overdue for this user (across all pages)
    now = datetime.now(timezone.utc)
    total_pending = db.query(func.count()).filter(
        ApplicationReviewAssignment.assigned_to == current_user.id,
        ApplicationReviewAssignment.status == "pending",
    ).scalar() or 0

    # Overdue: pending assignments where batch deadline has passed
    total_overdue = db.query(func.count(ApplicationReviewAssignment.id)).join(
        AssignmentBatch, ApplicationReviewAssignment.batch_id == AssignmentBatch.id,
    ).filter(
        ApplicationReviewAssignment.assigned_to == current_user.id,
        ApplicationReviewAssignment.status == "pending",
        AssignmentBatch.deadline.isnot(None),
        AssignmentBatch.deadline < now,
    ).scalar() or 0

    # Build response
    items = []
    for a in assignments:
        batch = a.batch
        b_stats = stats_map.get(a.batch_id, BatchStatsInfo())
        metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == a.email_id).first()

        is_overdue = a.status == "pending" and _is_past(batch.deadline)

        items.append(AssignmentResponse(
            id=a.id,
            email_id=a.email_id,
            batch_id=a.batch_id,
            applicant_name=metadata.applicant_name if metadata else None,
            category=metadata.ai_category if metadata else None,
            assigner_name=batch.creator.full_name if batch.creator else None,
            deadline=batch.deadline,
            notes=batch.notes,
            status=a.status,
            declined_reason=a.declined_reason,
            completed_at=a.completed_at,
            declined_at=a.declined_at,
            created_at=a.created_at,
            is_overdue=is_overdue,
            batch_total=b_stats.total,
            batch_completed=b_stats.completed,
        ))

    return AssignmentListResponse(
        assignments=items,
        pagination=PaginationInfo(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
        ),
        summary={"total_pending": total_pending, "total_overdue": total_overdue},
    )


@router.get("/batches", response_model=BatchListResponse)
async def list_my_batches(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """List batches the current user can manage (created or shared with)."""
    # Get batch IDs: owned or shared
    owned_ids = db.query(AssignmentBatch.id).filter(AssignmentBatch.created_by == current_user.id)
    shared_ids = db.query(AssignmentBatchShare.batch_id).filter(AssignmentBatchShare.shared_with == current_user.id)
    managed_ids_subq = owned_ids.union(shared_ids).subquery()

    # Query batches
    query = db.query(AssignmentBatch).filter(
        AssignmentBatch.id.in_(db.query(managed_ids_subq))
    )

    total_items = query.count()
    total_pages = max(1, math.ceil(total_items / page_size))

    batches = query.order_by(
        AssignmentBatch.updated_at.desc()
    ).offset((page - 1) * page_size).limit(page_size).all()

    # Stats for all batches
    batch_ids = [b.id for b in batches]
    stats_map = _get_batch_stats_map(db, batch_ids)

    items = []
    for b in batches:
        b_stats = stats_map.get(b.id, BatchStatsInfo())
        can_edit = b.created_by == current_user.id or any(
            s.shared_with == current_user.id for s in b.shares
        )
        items.append(_build_batch_response(b, current_user, b_stats, can_edit))

    return BatchListResponse(
        batches=items,
        pagination=PaginationInfo(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
        ),
    )


@router.get("/batch/{batch_id}", response_model=BatchDetailResponse)
async def get_batch_detail(
    batch_id: UUID,
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """Get batch details with assignments."""
    batch = db.query(AssignmentBatch).filter(AssignmentBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Batch not found"})

    has_access, can_edit = _check_batch_permission(db, batch, current_user)

    # Stats
    stats_map = _get_batch_stats_map(db, [batch.id])
    b_stats = stats_map.get(batch.id, BatchStatsInfo())

    # Get assignments
    assignment_query = db.query(ApplicationReviewAssignment).filter(
        ApplicationReviewAssignment.batch_id == batch.id,
    )

    # Assignees only see their own assignments
    if not can_edit:
        assignment_query = assignment_query.filter(
            ApplicationReviewAssignment.assigned_to == current_user.id,
        )

    assignments_db = assignment_query.order_by(ApplicationReviewAssignment.created_at.desc()).all()

    now = datetime.now(timezone.utc)
    assignment_items = []
    for a in assignments_db:
        metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == a.email_id).first()
        is_overdue = a.status == "pending" and _is_past(batch.deadline)

        assignment_items.append(AssignmentResponse(
            id=a.id,
            email_id=a.email_id,
            batch_id=a.batch_id,
            applicant_name=metadata.applicant_name if metadata else None,
            category=metadata.ai_category if metadata else None,
            assignee_name=a.assignee.full_name if a.assignee else None,
            assigner_name=batch.creator.full_name if batch.creator else None,
            deadline=batch.deadline,
            notes=batch.notes,
            status=a.status,
            declined_reason=a.declined_reason,
            completed_at=a.completed_at,
            declined_at=a.declined_at,
            created_at=a.created_at,
            is_overdue=is_overdue,
        ))

    # Build response
    base = _build_batch_response(batch, current_user, b_stats, can_edit)
    return BatchDetailResponse(
        **base.model_dump(),
        assignments=assignment_items,
    )


@router.patch("/batch/{batch_id}")
async def update_batch(
    batch_id: UUID,
    request: UpdateBatchRequest,
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """Update batch metadata (notes, deadline, shares)."""
    batch = db.query(AssignmentBatch).filter(AssignmentBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Batch not found"})

    _check_batch_permission(db, batch, current_user, require_edit=True)

    if request.notes is not None:
        batch.notes = request.notes
    if request.deadline is not None:
        batch.deadline = request.deadline

    # Replace shares if provided
    if request.share_with is not None:
        # Delete existing shares
        db.query(AssignmentBatchShare).filter(
            AssignmentBatchShare.batch_id == batch.id,
        ).delete()

        # Create new shares
        for uid in request.share_with:
            if uid != batch.created_by:  # Don't share with creator
                user = db.query(LabMember).filter(LabMember.id == uid, LabMember.is_active == True).first()
                if user:
                    share = AssignmentBatchShare(
                        batch_id=batch.id,
                        shared_with=uid,
                    )
                    db.add(share)

    batch.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {"message": "Batch updated successfully"}


@router.delete("/batch/{batch_id}")
async def delete_batch(
    batch_id: UUID,
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a batch and all its assignments (CASCADE)."""
    batch = db.query(AssignmentBatch).filter(AssignmentBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Batch not found"})

    _check_batch_permission(db, batch, current_user, require_edit=True)

    db.delete(batch)
    db.commit()

    return {"message": "Batch deleted successfully"}


@router.post("/batch/{batch_id}/assignments", response_model=CreateAssignmentResponse)
async def add_to_batch(
    batch_id: UUID,
    request: AddToBatchRequest,
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """Add more assignments to an existing batch."""
    batch = db.query(AssignmentBatch).filter(AssignmentBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Batch not found"})

    _check_batch_permission(db, batch, current_user, require_edit=True)

    # Validate assignees
    reviewers = db.query(LabMember).filter(
        LabMember.id.in_(request.assigned_to),
        LabMember.is_active == True,
    ).all()
    reviewer_map = {r.id: r for r in reviewers}

    for uid in request.assigned_to:
        if uid not in reviewer_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "user_not_found", "message": f"User {uid} not found or inactive"}
            )
        if not reviewer_map[uid].can_review:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "cannot_review", "message": f"User {reviewer_map[uid].full_name or uid} does not have review permission"}
            )

    # Validate email_ids
    emails_exist = db.query(Email.id).filter(Email.id.in_(request.email_ids)).all()
    existing_ids = {e.id for e in emails_exist}
    for eid in request.email_ids:
        if eid not in existing_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "application_not_found", "message": f"Application {eid} not found"}
            )

    created, duplicates = _create_assignments_for_batch(
        db, batch, request.email_ids, request.assigned_to
    )

    if created == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "all_duplicates",
                "message": "All requested assignments already exist",
                "duplicates": [d.model_dump(mode='json') for d in duplicates],
            }
        )

    batch.updated_at = datetime.now(timezone.utc)
    db.commit()

    return CreateAssignmentResponse(
        batch_id=batch.id,
        created=created,
        skipped_duplicates=len(duplicates),
        duplicates=duplicates,
    )


@router.get("/application/{email_id}")
async def get_application_assignments(
    email_id: UUID,
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """List all assignments for a specific application. Any authenticated user can see this."""
    assignments = db.query(ApplicationReviewAssignment).filter(
        ApplicationReviewAssignment.email_id == email_id,
    ).order_by(ApplicationReviewAssignment.created_at.desc()).all()

    items = []
    for a in assignments:
        batch = a.batch
        is_overdue = a.status == "pending" and _is_past(batch.deadline)

        items.append(ApplicationAssignmentResponse(
            id=a.id,
            batch_id=a.batch_id,
            assignee_name=a.assignee.full_name if a.assignee else None,
            assigner_name=batch.creator.full_name if batch.creator else None,
            deadline=batch.deadline,
            status=a.status,
            completed_at=a.completed_at,
            declined_at=a.declined_at,
        ))

    return {"assignments": [item.model_dump(mode='json') for item in items]}


@router.patch("/{assignment_id}")
async def decline_assignment(
    assignment_id: UUID,
    request: DeclineAssignmentRequest,
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """Decline an assignment. Only the assignee can decline."""
    assignment = db.query(ApplicationReviewAssignment).filter(
        ApplicationReviewAssignment.id == assignment_id,
    ).first()

    if not assignment:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Assignment not found"})

    if assignment.assigned_to != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "permission_denied", "message": "Only the assignee can decline an assignment"}
        )

    if assignment.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_status", "message": f"Cannot decline assignment with status '{assignment.status}'"}
        )

    now = datetime.now(timezone.utc)
    assignment.status = "declined"
    assignment.declined_reason = request.declined_reason
    assignment.declined_at = now
    assignment.updated_at = now
    db.commit()

    return {"message": "Assignment declined"}


@router.delete("/{assignment_id}")
async def delete_assignment(
    assignment_id: UUID,
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """Remove a single assignment. Only batch creator/shared users can delete."""
    assignment = db.query(ApplicationReviewAssignment).filter(
        ApplicationReviewAssignment.id == assignment_id,
    ).first()

    if not assignment:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Assignment not found"})

    batch = db.query(AssignmentBatch).filter(AssignmentBatch.id == assignment.batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Parent batch not found"})

    _check_batch_permission(db, batch, current_user, require_edit=True)

    db.delete(assignment)
    batch.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {"message": "Assignment removed"}


@router.post("/preview", response_model=PreviewBulkAssignmentResponse)
async def preview_bulk_assignment(
    request: PreviewBulkAssignmentRequest,
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db),
):
    """Preview bulk assignment by date range. Shows what would be created."""
    try:
        date_from = datetime.strptime(request.date_from, "%Y-%m-%d")
        date_to = datetime.strptime(request.date_to, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_date", "message": "Dates must be YYYY-MM-DD format"}
        )

    # Find applications in date range
    query = db.query(Email, EmailMetadata).join(
        EmailMetadata, Email.id == EmailMetadata.email_id,
    ).filter(
        EmailMetadata.ai_category.like('application-%'),
        Email.date >= date_from,
        Email.date <= date_to,
    )

    if request.category:
        query = query.filter(EmailMetadata.ai_category == request.category)

    results = query.all()

    applications = []
    duplicates = []

    for email, metadata in results:
        # Check if already assigned to any of the requested reviewers
        existing = db.query(ApplicationReviewAssignment).filter(
            ApplicationReviewAssignment.email_id == email.id,
            ApplicationReviewAssignment.assigned_to.in_(request.assigned_to),
        ).first()

        if existing:
            existing_batch = db.query(AssignmentBatch).filter(
                AssignmentBatch.id == existing.batch_id,
            ).first()
            duplicates.append(PreviewDuplicateInfo(
                email_id=email.id,
                applicant_name=metadata.applicant_name,
                existing_batch_id=existing.batch_id,
                existing_assigner=existing_batch.creator.full_name if existing_batch and existing_batch.creator else None,
            ))
        else:
            applications.append(PreviewApplicationInfo(
                email_id=email.id,
                applicant_name=metadata.applicant_name,
                category=metadata.ai_category,
                date=email.date,
            ))

    return PreviewBulkAssignmentResponse(
        total_applications=len(results),
        already_assigned=len(duplicates),
        new_to_assign=len(applications),
        applications=applications,
        duplicates=duplicates,
    )
