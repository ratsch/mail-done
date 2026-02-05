"""
Statistics Routes for Lab Application Review System
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, cast, Boolean, case, extract
from sqlalchemy import text
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import logging

from backend.core.database import get_db
from backend.core.database.models import (
    Email, EmailMetadata, ApplicationReview, ApplicationDecision, LabMember, SystemSettings
)
from backend.api.review_auth import get_current_user, get_current_user_hybrid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["review-stats"])


def get_category_thresholds(db: Session) -> dict:
    """Get category thresholds from SystemSettings."""
    default_thresholds = {
        "application-phd": 7,
        "application-postdoc": 7,
        "application-internship": 7,
        "application-visiting": 7,
        "application-other": 6,
        "application-msc-thesis": 1,
    }
    
    setting = db.query(SystemSettings).filter(SystemSettings.key == "category_thresholds").first()
    if setting and isinstance(setting.value, dict):
        # Handle nested structure: {"value": {...}, "type": "object"}
        # or direct structure: {"application-phd": 7, ...}
        if "value" in setting.value and isinstance(setting.value["value"], dict):
            thresholds = setting.value["value"]
        else:
            # Direct dict structure
            thresholds = setting.value
        
        # Only update if thresholds is a dict with category keys (not metadata keys)
        if isinstance(thresholds, dict):
            # Filter out metadata keys like 'value', 'type', etc.
            category_keys = [k for k in thresholds.keys() if k.startswith("application-")]
            if category_keys:
                result = default_thresholds.copy()
                # Only update with actual category keys
                for key in category_keys:
                    if isinstance(thresholds[key], (int, float)):
                        result[key] = int(thresholds[key])
                return result
    
    return default_thresholds


def get_base_application_query(db: Session):
    """
    Get base query for valid applications.
    
    Returns a query filtered to only include:
    - Applications (ai_category like 'application-%')
    - That went through reprocessing (has overall_recommendation_score or other scores)
    - That are actually applications (is_not_application = false or NULL)
    
    This ensures consistency across all stats calculations.
    An application is considered processed if it has any of the scores set by reprocess_applications.py.
    """
    return db.query(EmailMetadata).join(
        Email, EmailMetadata.email_id == Email.id
    ).filter(
        EmailMetadata.ai_category.like('application-%'),
        # Check if application has been processed by reprocess_applications.py
        # (has overall_recommendation_score OR scientific_excellence_score OR research_fit_score)
        or_(
            EmailMetadata.overall_recommendation_score.isnot(None),
            EmailMetadata.research_fit_score.isnot(None),
            text("(email_metadata.category_metadata->>'scientific_excellence_score') IS NOT NULL")
        ),
        # Exclude applications marked as not applications
        or_(
            text("(email_metadata.category_metadata->'red_flags'->>'is_not_application')::boolean = false"),
            text("email_metadata.category_metadata->'red_flags'->>'is_not_application' IS NULL"),
            text("email_metadata.category_metadata->'red_flags' IS NULL")
        )
    )


def apply_relevant_threshold_filter(query, thresholds: dict):
    """
    Apply category threshold filters to a query to get only relevant applications.
    
    Args:
        query: SQLAlchemy query to filter
        thresholds: Dictionary mapping category names to threshold scores
        
    Returns:
        Filtered query with threshold conditions applied
    """
    threshold_conditions = []
    for category, threshold in thresholds.items():
        threshold_conditions.append(
            and_(
                EmailMetadata.ai_category == category,
                EmailMetadata.overall_recommendation_score >= threshold
            )
        )
    
    if threshold_conditions:
        return query.filter(or_(*threshold_conditions))
    return query


@router.get("/overview")
async def get_overview_stats(
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get dashboard statistics.
    
    Returns overview stats for applications, reviews, decisions, etc.
    """
    try:
        # Total applications (all that went through reprocess_applications.py)
        total_applications = get_base_application_query(db).count()
        
        # Relevant applications (meeting category thresholds)
        thresholds = get_category_thresholds(db)
        relevant_query = apply_relevant_threshold_filter(
            get_base_application_query(db),
            thresholds
        )
        relevant_applications = relevant_query.count()
        
        # Applications by status (only valid applications)
        status_counts = get_base_application_query(db).with_entities(
            EmailMetadata.application_status,
            func.count(EmailMetadata.id)
        ).group_by(EmailMetadata.application_status).all()
        
        applications_by_status = {status or "pending": count for status, count in status_counts}
        
        # Pending review (no decision) - only for valid applications
        pending_review_query = get_base_application_query(db).outerjoin(
            ApplicationDecision, EmailMetadata.email_id == ApplicationDecision.email_id
        ).filter(
            ApplicationDecision.id.is_(None)
        )
        pending_review = pending_review_query.count()
        
        # Relevant pending review (meeting thresholds)
        pending_review_relevant = apply_relevant_threshold_filter(
            pending_review_query,
            thresholds
        ).count()
        
        # Decided (only for valid applications - use base query)
        decided = get_base_application_query(db).join(
            ApplicationDecision, EmailMetadata.email_id == ApplicationDecision.email_id
        ).count()
        
        # My pending reviews (applications I haven't rated yet) - only for valid applications
        my_pending_query = get_base_application_query(db).outerjoin(
            ApplicationReview,
            and_(
                ApplicationReview.email_id == EmailMetadata.email_id,
                ApplicationReview.lab_member_id == current_user.id
            )
        ).filter(
            ApplicationReview.id.is_(None)
        )
        my_pending = my_pending_query.count()
        
        # My pending reviews relevant (meeting thresholds)
        my_pending_reviews_relevant = apply_relevant_threshold_filter(
            my_pending_query,
            thresholds
        ).count()
        
        # Deadline approaching (within 3 days) - only for valid applications
        three_days_from_now = datetime.utcnow() + timedelta(days=3)
        deadline_approaching_query = get_base_application_query(db).filter(
            EmailMetadata.review_deadline.isnot(None),
            EmailMetadata.review_deadline <= three_days_from_now,
            EmailMetadata.review_deadline > datetime.utcnow()
        )
        deadline_approaching = deadline_approaching_query.count()
        
        # Relevant deadline approaching (meeting thresholds)
        deadline_approaching_relevant = apply_relevant_threshold_filter(
            deadline_approaching_query,
            thresholds
        ).count()
        
        # Past deadline (deadline has passed) - only for valid applications
        past_deadline_query = get_base_application_query(db).filter(
            EmailMetadata.review_deadline.isnot(None),
            EmailMetadata.review_deadline < datetime.utcnow()
        )
        past_deadline = past_deadline_query.count()
        
        # Relevant past deadline (meeting thresholds)
        past_deadline_relevant = apply_relevant_threshold_filter(
            past_deadline_query,
            thresholds
        ).count()
        
        # Old applications (>30 days without decision) - only for valid applications
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        old_applications = get_base_application_query(db).outerjoin(
            ApplicationDecision, EmailMetadata.email_id == ApplicationDecision.email_id
        ).filter(
            Email.date < thirty_days_ago,
            ApplicationDecision.id.is_(None)
        ).count()
        
        # Average rating (only for valid applications - use base query)
        # Join reviews to valid applications to ensure we only average ratings for actual applications
        avg_rating_result = db.query(func.avg(ApplicationReview.rating)).join(
            EmailMetadata, ApplicationReview.email_id == EmailMetadata.email_id
        ).join(
            Email, EmailMetadata.email_id == Email.id
        ).filter(
            EmailMetadata.ai_category.like('application-%'),
            # Check if application has been processed (has scores)
            or_(
                EmailMetadata.overall_recommendation_score.isnot(None),
                EmailMetadata.research_fit_score.isnot(None),
                text("(email_metadata.category_metadata->>'scientific_excellence_score') IS NOT NULL")
            ),
            # Exclude applications marked as not applications
            or_(
                text("(email_metadata.category_metadata->'red_flags'->>'is_not_application')::boolean = false"),
                text("email_metadata.category_metadata->'red_flags'->>'is_not_application' IS NULL"),
                text("email_metadata.category_metadata->'red_flags' IS NULL")
            )
        ).scalar()
        avg_rating_all = float(avg_rating_result) if avg_rating_result else None
        
        # Applications by category (only valid applications)
        category_counts = get_base_application_query(db).with_entities(
            EmailMetadata.ai_category,
            func.count(EmailMetadata.id)
        ).group_by(EmailMetadata.ai_category).all()
        
        applications_by_category = {cat: count for cat, count in category_counts}

        # Assignment stats for current user
        my_pending_assignments = 0
        my_overdue_assignments = 0
        try:
            from backend.core.database.models import ApplicationReviewAssignment, AssignmentBatch
            my_pending_assignments = db.query(func.count(ApplicationReviewAssignment.id)).filter(
                ApplicationReviewAssignment.assigned_to == current_user.id,
                ApplicationReviewAssignment.status == "pending",
            ).scalar() or 0

            my_overdue_assignments = db.query(func.count(ApplicationReviewAssignment.id)).join(
                AssignmentBatch, ApplicationReviewAssignment.batch_id == AssignmentBatch.id,
            ).filter(
                ApplicationReviewAssignment.assigned_to == current_user.id,
                ApplicationReviewAssignment.status == "pending",
                AssignmentBatch.deadline.isnot(None),
                AssignmentBatch.deadline < datetime.now(timezone.utc),
            ).scalar() or 0
        except Exception as e:
            logger.warning(f"Failed to get assignment stats: {e}")

        return {
            "total_applications": total_applications,
            "relevant_applications": relevant_applications,
            "pending_review": pending_review,
            "pending_review_relevant": pending_review_relevant,
            "decided": decided,
            "my_pending_reviews": my_pending,
            "my_pending_reviews_relevant": my_pending_reviews_relevant,
            "deadline_approaching": deadline_approaching,
            "deadline_approaching_relevant": deadline_approaching_relevant,
            "past_deadline": past_deadline,
            "past_deadline_relevant": past_deadline_relevant,
            "old_applications": old_applications,
            "avg_rating_all": avg_rating_all,
            "applications_by_status": applications_by_status,
            "applications_by_category": applications_by_category,
            "my_pending_assignments": my_pending_assignments,
            "my_overdue_assignments": my_overdue_assignments
        }
    except Exception as e:
        logger.error(f"Error getting overview stats: {e}", exc_info=True)
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get("/applications/weekly")
async def get_weekly_application_counts(
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get number of applications per week for the last 3 months.
    Returns data grouped by week (ISO week format).
    """
    try:
        three_months_ago = datetime.utcnow() - timedelta(days=90)
        
        # Get base query for valid applications (already joins Email)
        base_query = get_base_application_query(db)
        
        # Filter to last 3 months and group by week
        # Use date_trunc to group by week (Monday as start of week)
        weekly_counts = base_query.filter(
            Email.date >= three_months_ago
        ).with_entities(
            func.date_trunc('week', Email.date).label('week'),
            func.count(EmailMetadata.email_id).label('count')
        ).group_by(
            func.date_trunc('week', Email.date)
        ).order_by(
            func.date_trunc('week', Email.date)
        ).all()
        
        # Format results
        result = []
        for week, count in weekly_counts:
            result.append({
                "week": week.isoformat() if week else None,
                "count": count
            })
        
        return {"data": result}
    except Exception as e:
        logger.error(f"Error getting weekly application counts: {e}", exc_info=True)
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve weekly statistics: {str(e)}"
        )


@router.get("/applications/daily")
async def get_daily_application_counts(
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get number of applications per day for the last 2 weeks.
    Returns data grouped by day.
    """
    try:
        two_weeks_ago = datetime.utcnow() - timedelta(days=14)
        
        # Get base query for valid applications (already joins Email)
        base_query = get_base_application_query(db)
        
        # Filter to last 2 weeks and group by day
        # Use date_trunc to group by day
        daily_counts = base_query.filter(
            Email.date >= two_weeks_ago
        ).with_entities(
            func.date_trunc('day', Email.date).label('day'),
            func.count(EmailMetadata.email_id).label('count')
        ).group_by(
            func.date_trunc('day', Email.date)
        ).order_by(
            func.date_trunc('day', Email.date)
        ).all()
        
        # Format results
        result = []
        for day, count in daily_counts:
            result.append({
                "day": day.isoformat() if day else None,
                "count": count
            })
        
        return {"data": result}
    except Exception as e:
        logger.error(f"Error getting daily application counts: {e}", exc_info=True)
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve daily statistics: {str(e)}"
        )


@router.get("/applications/categories")
async def get_application_categories(
    current_user: LabMember = Depends(get_current_user_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get list of all application categories with counts.
    
    Returns both the predefined categories (for tabs) and actual counts from database.
    This allows frontend to dynamically build category tabs/filters.
    """
    try:
        # Predefined application categories in display order
        predefined_categories = [
            {"id": "application-phd", "label": "PhD", "short": "phd"},
            {"id": "application-postdoc", "label": "Postdoc", "short": "postdoc"},
            {"id": "application-intern", "label": "Internship", "short": "intern"},
            {"id": "application-visiting", "label": "Visiting", "short": "visiting"},
            {"id": "application-bsc-msc-thesis", "label": "BSc/MSc Thesis", "short": "thesis"},
            {"id": "application-other", "label": "Other", "short": "other"},
        ]
        
        # Get actual counts from database
        category_counts = get_base_application_query(db).with_entities(
            EmailMetadata.ai_category,
            func.count(EmailMetadata.id)
        ).group_by(EmailMetadata.ai_category).all()
        
        counts_dict = {cat: count for cat, count in category_counts}
        
        # Build result with counts
        categories = []
        total = 0
        for cat in predefined_categories:
            count = counts_dict.get(cat["id"], 0)
            total += count
            categories.append({
                **cat,
                "count": count
            })
        
        return {
            "categories": categories,
            "total": total
        }
    except Exception as e:
        logger.error(f"Error getting application categories: {e}", exc_info=True)
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve categories: {str(e)}"
        )

