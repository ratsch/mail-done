"""
Application Review Routes

Endpoints for listing, viewing, and managing applications
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status, Request, Response
from fastapi import status as http_status
from fastapi.responses import Response as FastAPIResponse
from sqlalchemy.orm import Session
import hashlib
import json
from sqlalchemy import func, and_, or_, desc, asc, Integer, cast, Boolean, text, select
from sqlalchemy.types import JSON
from uuid import UUID
from datetime import datetime, timedelta
import logging

from backend.core.database import get_db
from backend.core.database.models import (
    Email, EmailMetadata, ApplicationReview, ApplicationDecision, ApplicationPrivateNote, LabMember, SystemSettings,
    ApplicationCollectionItem
)
from backend.api.review_auth import get_current_reviewer, get_current_admin, get_current_reviewer_hybrid, get_current_admin_hybrid
from backend.api.review_schemas import (
    ApplicationListItem, ApplicationDetailResponse, ReviewRequest, ReviewResponse,
    ReviewSummaryResponse, DecisionRequest, DecisionResponse, AvailableTagsResponse,
    PrivateNotesRequest
)
from backend.api.review_middleware import log_audit_event, check_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/applications", tags=["applications"])


def get_category_thresholds(db: Session) -> dict:
    """
    Get category thresholds from SystemSettings.
    Returns dict mapping category to threshold score.
    Default thresholds if not configured:
    - application-phd: 7
    - application-postdoc: 7
    - application-internship: 7
    - application-other: 6
    - application-msc-thesis: 1
    """
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
        thresholds = setting.value.get("value", {})
        if isinstance(thresholds, dict):
            # Merge with defaults (user config overrides defaults)
            result = default_thresholds.copy()
            result.update(thresholds)
            return result
    
    return default_thresholds


def meets_category_threshold(metadata: EmailMetadata, db: Session) -> bool:
    """
    Check if application meets the threshold for its category.
    Returns True if overall_recommendation_score >= threshold for category.
    """
    category = metadata.ai_category
    if not category or not category.startswith('application-'):
        return False
    
    thresholds = get_category_thresholds(db)
    threshold = thresholds.get(category, 7)  # Default to 7 if category not in thresholds
    
    score = metadata.overall_recommendation_score
    if score is None:
        return False
    
    return score >= threshold


def build_application_query(
    db: Session,
    category: Optional[str] = None,
    min_recommendation_score: Optional[int] = None,
    min_excellence_score: Optional[int] = None,
    min_research_fit_score: Optional[int] = None,
    search_name: Optional[str] = None,
    search_text: Optional[str] = None,  # Full-text search across AI analysis fields
    received_after: Optional[str] = None,
    received_before: Optional[str] = None,
    application_status: Optional[str] = None,
    has_decision: Optional[bool] = None,
    exclude_rejected: Optional[bool] = None,  # If True, exclude applications with decision='reject' or 'delete'
    decision_type: Optional[str] = None,  # Filter to specific decision type: 'accept', 'reject', 'interview', 'request_more_info'
    is_mass_email: Optional[bool] = None,
    no_research_background: Optional[bool] = None,
    insufficient_materials: Optional[bool] = None,
    is_cold_email: Optional[bool] = None,
    needs_review_by_me: Optional[UUID] = None,
    deadline_approaching: Optional[bool] = None,
    only_relevant: Optional[bool] = None,
    require_score: bool = True,
    profile_tags: Optional[List[str]] = None,  # List of tag names (AND logic - all must be present)
    highest_degree: Optional[List[str]] = None,  # List of degree values (OR logic - any match)
    is_not_application_filter: Optional[bool] = None,  # None=show all, True=only show is_not_application=true, False=only show is_not_application=false
    application_source: Optional[str] = None,  # Filter by application source (e.g., 'ai_center')
    collection_id: Optional[UUID] = None,  # Filter by collection
    assigned_to_me: Optional[UUID] = None  # Filter by pending assignments for this user
) -> Query:
    """Build optimized query for applications with filters."""
    try:
        query = db.query(Email, EmailMetadata).join(EmailMetadata, Email.id == EmailMetadata.email_id)
        
        # Base filters: applications only, reprocessed, and is application
        # Exclude inquiry-* categories (these are information requests, not applications)
        query = query.filter(
            EmailMetadata.ai_category.like('application-%'),
            ~EmailMetadata.ai_category.like('inquiry-%')  # Exclude inquiries
        )
        
        # Filter by is_not_application flag
        # Default behavior (is_not_application_filter=False): only show actual applications
        # If is_not_application_filter is True: only show non-applications
        # If is_not_application_filter is None (admin only): show all (no filter)
        if is_not_application_filter is False:
            # Only show actual applications (is_not_application=false or NULL) - DEFAULT BEHAVIOR
            query = query.filter(
                or_(
                    text("(email_metadata.category_metadata->'red_flags'->>'is_not_application')::boolean = false"),
                    text("email_metadata.category_metadata->'red_flags'->>'is_not_application' IS NULL"),
                    text("email_metadata.category_metadata->'red_flags' IS NULL")
                )
            )
        elif is_not_application_filter is True:
            # Only show non-applications (is_not_application=true)
            query = query.filter(
                text("(email_metadata.category_metadata->'red_flags'->>'is_not_application')::boolean = true")
            )
        # If is_not_application_filter is None (admin only), don't filter (show all)
        
        # Include applications that have been processed by reprocess_applications.py
        # (has overall_recommendation_score OR scientific_excellence_score OR research_fit_score)
        # Only apply this filter if require_score is True (default)
        # When require_score is False, we show all applications (that aren't marked as not-applications)
        if require_score:
            query = query.filter(
                or_(
                    EmailMetadata.overall_recommendation_score.isnot(None),
                    EmailMetadata.research_fit_score.isnot(None),
                    EmailMetadata.scientific_excellence_score.isnot(None)
                )
            )
        
        # Category filter
        if category:
            # Handle both formats: "application-internship" or just "internship"
            if category.startswith("application-"):
                category_filter = category
            else:
                category_filter = f"application-{category}"
            
            # Map legacy category names to actual database values
            category_mapping = {
                "application-internship": "application-intern",  # Legacy name -> actual DB value
                "application-msc-thesis": "application-bsc-msc-thesis",  # Legacy name -> actual DB value
            }
            category_filter = category_mapping.get(category_filter, category_filter)
            
            query = query.filter(EmailMetadata.ai_category == category_filter)
        
        # Application source filter (stored in category_metadata.application_source)
        if application_source:
            query = query.filter(
                text("email_metadata.category_metadata->>'application_source' = :app_source").bindparams(app_source=application_source)
            )
        
        # Collection filter (applications in a specific collection)
        if collection_id:
            # Use select() for the IN clause to avoid SAWarning
            collection_items_select = select(ApplicationCollectionItem.email_id).where(
                ApplicationCollectionItem.collection_id == collection_id
            )
            query = query.filter(Email.id.in_(collection_items_select))
        
        # Score filters
        if min_recommendation_score:
            query = query.filter(EmailMetadata.overall_recommendation_score >= min_recommendation_score)
        
        if min_excellence_score:
            query = query.filter(
                EmailMetadata.scientific_excellence_score >= min_excellence_score
            )
        
        if min_research_fit_score:
            query = query.filter(EmailMetadata.research_fit_score >= min_research_fit_score)
        
        # Name and institution search (case-insensitive, OR logic)
        if search_name:
            sanitized_search = search_name.strip().lower()
            query = query.filter(
                or_(
                    func.lower(EmailMetadata.applicant_name).ilike(f"%{sanitized_search}%"),
                    func.lower(EmailMetadata.applicant_institution).ilike(f"%{sanitized_search}%")
                )
            )
        
        # Date filters
        if received_after:
            query = query.filter(Email.date >= datetime.fromisoformat(received_after))
        
        if received_before:
            query = query.filter(Email.date <= datetime.fromisoformat(received_before))
        
        # Status filter
        if application_status:
            query = query.filter(EmailMetadata.application_status == application_status)
        
        # Has decision
        if has_decision is not None:
            decision_select = select(ApplicationDecision.email_id)
            if has_decision:
                query = query.filter(Email.id.in_(decision_select))
            else:
                query = query.filter(Email.id.notin_(decision_select))

        # Exclude rejected and deleted applications
        if exclude_rejected:
            rejected_select = select(ApplicationDecision.email_id).where(
                ApplicationDecision.decision.in_(['reject', 'delete'])
            )
            query = query.filter(Email.id.notin_(rejected_select))

        # Filter by specific decision type
        if decision_type:
            decision_type_select = select(ApplicationDecision.email_id).where(ApplicationDecision.decision == decision_type)
            query = query.filter(Email.id.in_(decision_type_select))

        # Red flags
        if is_mass_email is not None:
            # Use parameterized query to avoid SQL injection
            query = query.filter(
                text("(email_metadata.category_metadata->'red_flags'->>'is_mass_email')::boolean = :is_mass_email").bindparams(is_mass_email=is_mass_email)
            )
        
        if no_research_background is not None:
            query = query.filter(
                text("(email_metadata.category_metadata->'red_flags'->>'no_research_background')::boolean = :no_research_background").bindparams(no_research_background=no_research_background)
            )

        if insufficient_materials is not None:
            query = query.filter(
                text("(email_metadata.category_metadata->'red_flags'->>'insufficient_materials')::boolean = :insufficient_materials").bindparams(insufficient_materials=insufficient_materials)
            )

        if is_cold_email is not None:
            query = query.filter(EmailMetadata.is_cold_email == is_cold_email)
        
        # Needs review by me
        if needs_review_by_me:
            reviewed_select = select(ApplicationReview.email_id).where(
                ApplicationReview.lab_member_id == needs_review_by_me
            )
            query = query.filter(Email.id.notin_(reviewed_select))

        # Assigned to me (pending assignments)
        if assigned_to_me:
            from backend.core.database.models import ApplicationReviewAssignment
            assignment_exists = db.query(ApplicationReviewAssignment.id).filter(
                ApplicationReviewAssignment.email_id == Email.id,
                ApplicationReviewAssignment.assigned_to == assigned_to_me,
                ApplicationReviewAssignment.status == "pending",
            ).exists()
            query = query.filter(assignment_exists)

        # Deadline approaching (next 3 days)
        if deadline_approaching:
            now = datetime.utcnow()
            three_days = now + timedelta(days=3)
            query = query.filter(
                EmailMetadata.review_deadline.isnot(None),
                EmailMetadata.review_deadline <= three_days,
                EmailMetadata.review_deadline > now
            )
        
        # Only relevant applications (based on category thresholds)
        if only_relevant:
            thresholds = get_category_thresholds(db)
            threshold_filters = []
            for cat, threshold in thresholds.items():
                threshold_filters.append(
                    and_(
                        EmailMetadata.ai_category == cat,
                        EmailMetadata.overall_recommendation_score >= threshold
                    )
                )
            if threshold_filters:
                query = query.filter(or_(*threshold_filters))
        
        # Profile tags filter (AND logic - all selected tags must be present)
        if profile_tags and len(profile_tags) > 0:
            # For each tag, check that it exists in the profile_tags array
            # profile_tags is stored as: [{"tag": "graph_neural_networks", ...}, ...]
            # Build conditions for each tag (AND logic - all must exist)
            # Note: category_metadata is JSON (not JSONB), so we cast to JSONB for array operations
            for tag_name in profile_tags:
                # Cast JSON to JSONB for jsonb_array_elements function
                # This works because PostgreSQL can cast JSON to JSONB
                query = query.filter(
                    text("EXISTS (SELECT 1 FROM jsonb_array_elements((email_metadata.category_metadata::jsonb)->'profile_tags') AS tag_elem WHERE tag_elem->>'tag' = :tag_name)")
                ).params(tag_name=tag_name)
        
        # Note: highest_degree filter cannot be applied here because category_specific_data is EncryptedJSON
        # Filtering will be done in Python after decryption (see list_applications endpoint)
        
        return query
    except Exception as e:
        logger.error(f"Error building application query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error querying applications")


@router.get("", response_model=dict)
async def list_applications(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_recommendation_score: Optional[int] = Query(None, ge=0, le=10),
    min_excellence_score: Optional[int] = Query(None, ge=0, le=10),
    min_research_fit_score: Optional[int] = Query(None, ge=0, le=10),
    search_name: Optional[str] = Query(None, description="Search by applicant name or institution (case-insensitive)"),
    search_text: Optional[str] = Query(None, description="Full-text search across AI analysis fields (key_strengths, concerns, ai_reasoning, current_situation, additional_notes, red_flags)"),
    received_after: Optional[str] = Query(None, description="Received after date (YYYY-MM-DD)"),
    received_before: Optional[str] = Query(None, description="Received before date (YYYY-MM-DD)"),
    status: Optional[str] = Query(None, description="Application status"),
    has_decision: Optional[bool] = Query(None),
    exclude_rejected: Optional[bool] = Query(None, description="Exclude applications with decision='reject' or 'delete'"),
    decision_type: Optional[str] = Query(None, description="Filter to specific decision type: 'accept', 'reject', 'interview', 'request_more_info'"),
    is_mass_email: Optional[bool] = Query(None),
    no_research_background: Optional[bool] = Query(None),
    insufficient_materials: Optional[bool] = Query(None),
    is_cold_email: Optional[bool] = Query(None),
    needs_review_by_me: Optional[bool] = Query(None),
    deadline_approaching: Optional[bool] = Query(None),
    only_relevant: Optional[bool] = Query(None, description="Filter to only applications meeting category thresholds"),
    require_score: bool = Query(True, description="Require applications to have scores assigned (default: True)"),
    profile_tags: Optional[str] = Query(None, description="Comma-separated list of profile tags (AND logic - all must be present)"),
    highest_degree: Optional[str] = Query(None, description="Comma-separated list of highest degrees (OR logic - any match)"),
    is_not_application_filter: Optional[bool] = Query(None, description="Filter by is_not_application: false=only actual applications (default), true=only non-applications, null=show all (admin only)"),
    application_source: Optional[str] = Query(None, description="Filter by application source (e.g., 'ai_center' for AI Center applications)"),
    collection_id: Optional[UUID] = Query(None, description="Filter by collection ID"),
    assigned_to_me: Optional[bool] = Query(None, description="Show only applications with pending assignments for me"),
    sort_by: str = Query("date", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order"),
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1, le=10000, description="Number of results per page (max 10000 for 'All')"),
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    List applications with filtering and pagination.
    
    Requires can_review permission.
    Rate limit: 60 requests per minute (members), 120 requests per minute (admins).
    """
    try:
        # Check rate limit (60 requests per minute for members, 120 for admins)
        is_allowed, limit_info = check_rate_limit(str(current_user.id), "/applications", "requests", db, user_role=current_user.role)
        if not is_allowed:
            retry_after = limit_info.get("retry_after", 60) if limit_info else 60
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Please try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )
        # Parse profile_tags from comma-separated string
        profile_tags_list = None
        if profile_tags:
            profile_tags_list = [tag.strip() for tag in profile_tags.split(",") if tag.strip()]
        
        # Parse highest_degree from comma-separated string
        highest_degree_list = None
        if highest_degree:
            highest_degree_list = [deg.strip() for deg in highest_degree.split(",") if deg.strip()]
        
        # Handle is_not_application_filter: default is False (only show actual applications)
        # Admins can set to None (show all) or True (only non-applications)
        # Non-admins can ONLY use False (actual applications) - cannot access non-applications
        # Check if user is admin
        is_admin = current_user.role == "admin"
        effective_is_not_application_filter = is_not_application_filter
        if is_not_application_filter is None:
            # Default: only show actual applications (is_not_application=false)
            effective_is_not_application_filter = False
        elif not is_admin:
            # Non-admins cannot use None (show all) or True (non-applications) - restrict to False
            # This prevents non-admins from accessing non-applications even if they try to bypass frontend
            effective_is_not_application_filter = False
        
        # Build query
        query = build_application_query(
            db=db,
            category=category,
            min_recommendation_score=min_recommendation_score,
            min_excellence_score=min_excellence_score,
            min_research_fit_score=min_research_fit_score,
            search_name=search_name,
            search_text=search_text,
            received_after=received_after,
            received_before=received_before,
            application_status=status,
            has_decision=has_decision,
            exclude_rejected=exclude_rejected,
            decision_type=decision_type,
            is_mass_email=is_mass_email,
            no_research_background=no_research_background,
            insufficient_materials=insufficient_materials,
            is_cold_email=is_cold_email,
            needs_review_by_me=current_user.id if needs_review_by_me else None,
            deadline_approaching=deadline_approaching,
            only_relevant=only_relevant,
            require_score=require_score,
            profile_tags=profile_tags_list,
            highest_degree=highest_degree_list,
            is_not_application_filter=effective_is_not_application_filter,
            application_source=application_source,
            collection_id=collection_id,
            assigned_to_me=current_user.id if assigned_to_me else None
        )
        
        # Optimize: Use subqueries to avoid N+1 queries
        # Subquery for review summaries
        review_summary_subq = db.query(
            ApplicationReview.email_id,
            func.avg(ApplicationReview.rating).label('avg_rating'),
            func.count(ApplicationReview.id).label('num_ratings')
        ).group_by(ApplicationReview.email_id).subquery()
        
        # Subquery for my reviews
        my_review_subq = db.query(
            ApplicationReview.email_id,
            ApplicationReview.rating
        ).filter(
            ApplicationReview.lab_member_id == current_user.id
        ).subquery()
        
        # Subquery for decisions
        decision_subq = db.query(
            ApplicationDecision.email_id,
            ApplicationDecision.decision
        ).subquery()
        
        # Join with subqueries BEFORE counting/sorting
        query = query.outerjoin(
            review_summary_subq,
            Email.id == review_summary_subq.c.email_id
        ).outerjoin(
            my_review_subq,
            Email.id == my_review_subq.c.email_id
        ).outerjoin(
            decision_subq,
            Email.id == decision_subq.c.email_id
        )
        
        # Add subquery columns to select
        query = query.add_columns(
            review_summary_subq.c.avg_rating,
            review_summary_subq.c.num_ratings,
            my_review_subq.c.rating.label('my_rating'),
            decision_subq.c.decision.label('decision')
        )
        
        # Get total count (before sorting)
        total = query.count()
        
        # Apply sorting (now we can use avg_rating from subquery)
        # Validate sort_by parameter against allowed fields
        allowed_sort_fields = {
            "date": Email.date,
            "rating": review_summary_subq.c.avg_rating,
            "recommendation_score": EmailMetadata.overall_recommendation_score,
            "excellence_score": EmailMetadata.scientific_excellence_score,
            "research_fit_score": EmailMetadata.research_fit_score,
            "relevance_score": EmailMetadata.relevance_score,
            "applicant_name": EmailMetadata.applicant_name,
            "institution": EmailMetadata.applicant_institution,
            "deadline": EmailMetadata.review_deadline,
            # Technical experience scores (using text() for JSON extraction)
            "coding_experience_score": text("(email_metadata.category_metadata->'technical_experience_scores'->>'coding_experience')::integer"),
            "medical_data_experience_score": text("(email_metadata.category_metadata->'technical_experience_scores'->>'medical_data_experience')::integer"),
            "omics_genomics_experience_score": text("(email_metadata.category_metadata->'technical_experience_scores'->>'omics_genomics_experience')::integer"),
            "sequence_analysis_experience_score": text("(email_metadata.category_metadata->'technical_experience_scores'->>'sequence_analysis_algorithms_experience')::integer"),
            "image_analysis_experience_score": text("(email_metadata.category_metadata->'technical_experience_scores'->>'image_analysis_experience')::integer"),
        }
        
        # Default to date if invalid sort field provided
        if sort_by not in allowed_sort_fields:
            logger.warning(f"Invalid sort_by parameter: {sort_by}, defaulting to 'date'")
            sort_by = "date"
        
        order_col = allowed_sort_fields[sort_by]
        
        if sort_order == "asc":
            query = query.order_by(asc(order_col))
        else:
            query = query.order_by(desc(order_col))
        
        # Apply pagination (but we may need to filter by highest_degree or search_text in Python after decryption)
        # If highest_degree or search_text filter is active, we need to fetch more results to account for filtering
        offset = (page - 1) * limit

        # Helper function for full-text search across encrypted fields
        def matches_search_text(metadata, search_query: str) -> bool:
            """Check if any of the searchable fields contain the search query (case-insensitive)."""
            search_lower = search_query.lower()

            # Search in ai_reasoning (EncryptedText)
            ai_reasoning = metadata.ai_reasoning or ""
            if search_lower in ai_reasoning.lower():
                return True

            # Search in category_specific_data (EncryptedJSON) fields
            category_specific = metadata.category_specific_data or {}

            # Search in key_strengths (array of strings)
            key_strengths = category_specific.get('key_strengths', [])
            if isinstance(key_strengths, list):
                for strength in key_strengths:
                    if isinstance(strength, str) and search_lower in strength.lower():
                        return True

            # Search in concerns (array of strings)
            concerns = category_specific.get('concerns', [])
            if isinstance(concerns, list):
                for concern in concerns:
                    if isinstance(concern, str) and search_lower in concern.lower():
                        return True

            # Search in current_situation (string)
            current_situation = category_specific.get('current_situation', "")
            if isinstance(current_situation, str) and search_lower in current_situation.lower():
                return True

            # Search in additional_notes (string)
            additional_notes = category_specific.get('additional_notes', "")
            if isinstance(additional_notes, str) and search_lower in additional_notes.lower():
                return True

            # Search in red_flags keys (from category_metadata, which is unencrypted)
            category_metadata = metadata.category_metadata or {}
            red_flags = category_metadata.get('red_flags', {})
            if isinstance(red_flags, dict):
                for flag_key, flag_value in red_flags.items():
                    # Only search keys where value is True
                    if flag_value is True and search_lower in flag_key.lower():
                        return True

            return False

        # If highest_degree or search_text filter is active, we need to fetch all results, filter, then paginate
        needs_python_filtering = (highest_degree_list and len(highest_degree_list) > 0) or search_text

        if needs_python_filtering:
            # Fetch all results (without limit) to filter in Python
            all_results = query.all()

            # Filter in Python (after decryption)
            filtered_results = []
            for row in all_results:
                metadata = row[1]

                # Apply highest_degree filter if active
                if highest_degree_list and len(highest_degree_list) > 0:
                    category_specific = metadata.category_specific_data or {}
                    degree_value = category_specific.get('highest_degree_completed')

                    # Check if degree matches any of the selected degrees (OR logic)
                    if degree_value not in highest_degree_list:
                        continue  # Skip this result

                # Apply search_text filter if active
                if search_text:
                    if not matches_search_text(metadata, search_text):
                        continue  # Skip this result

                filtered_results.append(row)

            # Now paginate the filtered results
            total = len(filtered_results)
            results = filtered_results[offset:offset + limit]
        else:
            # Normal pagination when no Python-level filtering needed
            results = query.offset(offset).limit(limit).all()
        
        # Build response items from joined results
        items = []
        for row in results:
            # Unpack: email, metadata, avg_rating, num_ratings, my_rating, decision
            email = row[0]
            metadata = row[1]
            avg_rating = row[2]
            num_ratings = row[3] or 0
            my_rating = row[4]
            decision = row[5]
            
            # Extract Google Drive links for email_text_link and folder_path
            gdrive_links = metadata.google_drive_links or {}
            email_text_link = None
            folder_path = metadata.google_drive_folder_id
            
            if isinstance(gdrive_links, dict):
                email_text_info = gdrive_links.get('email_text', {})
                email_text_link = email_text_info.get('link')
            
            # Extract technical experience scores from category_metadata
            category_metadata = metadata.category_metadata or {}
            tech_scores = category_metadata.get('technical_experience_scores', {})
            profile_tags = category_metadata.get('profile_tags', [])
            
            # Extract category_specific_data for highest_degree
            category_specific = metadata.category_specific_data or {}
            
            items.append(ApplicationListItem(
                email_id=email.id,
                applicant_name=metadata.applicant_name,
                applicant_institution=metadata.applicant_institution,
                date=email.date,
                category=metadata.ai_category,
                scientific_excellence_score=metadata.scientific_excellence_score or category_metadata.get('scientific_excellence_score'),
                research_fit_score=metadata.research_fit_score,
                overall_recommendation_score=metadata.overall_recommendation_score,
                relevance_score=metadata.relevance_score,
                avg_rating=float(avg_rating) if avg_rating else None,
                num_ratings=num_ratings,
                my_rating=my_rating,
                status=metadata.application_status or "pending",
                review_deadline=metadata.review_deadline,
                decision=decision,
                received_date=email.date,
                email_text_link=email_text_link,
                folder_path=folder_path,
                # Technical experience scores
                coding_experience_score=tech_scores.get('coding_experience'),
                medical_data_experience_score=tech_scores.get('medical_data_experience'),
                omics_genomics_experience_score=tech_scores.get('omics_genomics_experience'),
                sequence_analysis_experience_score=tech_scores.get('sequence_analysis_algorithms_experience'),
                image_analysis_experience_score=tech_scores.get('image_analysis_experience'),
                # Profile tags
                profile_tags=profile_tags if profile_tags else None,
                # Highest degree
                highest_degree=category_specific.get('highest_degree_completed'),
                # Application source
                application_source=category_metadata.get('application_source'),
            ))
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit
        }
    except Exception as e:
        logger.error(f"Error listing applications: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list applications: {e}")


@router.get("/export")
async def export_applications(
    export_format: str = Query("excel", alias="format", description="Export format: excel or csv"),
    exclude_pii: bool = Query(False, description="Exclude PII fields"),
    category: Optional[str] = Query(None),
    min_recommendation_score: Optional[int] = Query(None, ge=0, le=10),
    current_admin: LabMember = Depends(get_current_admin_hybrid),  # Admin only
    db: Session = Depends(get_db)
):
    """
    Export applications to Excel or CSV (admin only).
    
    Maximum 10,000 rows. Includes watermark with user and timestamp.
    """
    # Build query (same as list_applications, but without user-specific filters)
    # Exclude inquiry-* categories (information requests, not applications)
    query = db.query(Email, EmailMetadata).join(
        EmailMetadata, Email.id == EmailMetadata.email_id
    ).filter(
        EmailMetadata.ai_category.like('application-%'),
        ~EmailMetadata.ai_category.like('inquiry-%')
    )
    
    # Apply filters
    if category:
        query = query.filter(EmailMetadata.ai_category == category)
    if min_recommendation_score:
        query = query.filter(EmailMetadata.overall_recommendation_score >= min_recommendation_score)
    
    # Limit to 10,000 rows
    results = query.limit(10000).all()
    
    if len(results) >= 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Export limit exceeded. Maximum 10,000 rows. Please use filters to reduce the dataset."
        )
    
    # Prepare data for export
    export_data = []
    for email, metadata in results:
        row = {
            "email_id": str(email.id),
            "applicant_name": metadata.applicant_name if not exclude_pii else "[REDACTED]",
            "applicant_institution": metadata.applicant_institution if not exclude_pii else "[REDACTED]",
            "date": email.date.isoformat() if email.date else None,
            "category": metadata.ai_category,
            "scientific_excellence_score": metadata.scientific_excellence_score or (metadata.category_metadata.get('scientific_excellence_score') if metadata.category_metadata else None),
            "research_fit_score": metadata.research_fit_score,
            "overall_recommendation_score": metadata.overall_recommendation_score,
            "application_status": metadata.application_status or "pending",
            "review_deadline": metadata.review_deadline.isoformat() if metadata.review_deadline else None,
        }
        
        # Add PII fields only if not excluded
        if not exclude_pii:
            if metadata.category_specific_data:
                row["nationality"] = metadata.category_specific_data.get('nationality')
                row["highest_degree"] = metadata.category_specific_data.get('highest_degree')
                row["current_situation"] = metadata.category_specific_data.get('current_situation')
                row["github_account"] = metadata.category_specific_data.get('github_account')
                row["linkedin_account"] = metadata.category_specific_data.get('linkedin_account')
        
        # Get review summary
        reviews = db.query(ApplicationReview).filter(ApplicationReview.email_id == email.id).all()
        row["num_ratings"] = len(reviews)
        row["avg_rating"] = sum(r.rating for r in reviews) / len(reviews) if reviews else None
        
        # Get decision
        decision = db.query(ApplicationDecision).filter(ApplicationDecision.email_id == email.id).first()
        row["decision"] = decision.decision if decision else None
        
        export_data.append(row)
    
    # Generate filename with watermark
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"applications_export_{timestamp}_{current_admin.email.split('@')[0]}"
    
    if export_format.lower() == "csv":
        import csv
        import io
        
        output = io.StringIO()
        if export_data:
            writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
            writer.writeheader()
            writer.writerows(export_data)
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}.csv"',
                "X-Export-Metadata": json.dumps({
                    "exported_by": current_admin.email,
                    "exported_at": datetime.utcnow().isoformat(),
                    "row_count": len(export_data),
                    "exclude_pii": exclude_pii
                })
            }
        )
    else:  # Excel
        try:
            import pandas as pd
            from io import BytesIO
            from openpyxl.comments import Comment
            
            df = pd.DataFrame(export_data)
            output = BytesIO()
            
            # Create Excel with watermark in metadata
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Applications')
                worksheet = writer.sheets['Applications']
                # Add watermark note in first cell comment
                comment = Comment(f"Exported by {current_admin.email} on {datetime.utcnow().isoformat()}", "System")
                worksheet.cell(row=1, column=1).comment = comment
            
            output.seek(0)
            
            return Response(
                content=output.read(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}.xlsx"',
                    "X-Export-Metadata": json.dumps({
                        "exported_by": current_admin.email,
                        "exported_at": datetime.utcnow().isoformat(),
                        "row_count": len(export_data),
                        "exclude_pii": exclude_pii
                    })
                }
            )
        except ImportError:
            # Fallback to CSV if pandas/openpyxl not available
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Excel export requires pandas and openpyxl. Use CSV format or install dependencies."
            )


@router.get("/available-tags", response_model=AvailableTagsResponse)
async def get_available_tags(
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get all unique profile tags that exist in applications.
    
    Returns a list of unique tag names (strings) that can be used for filtering.
    """
    try:
        # Query all applications (we'll filter in Python since category_metadata is JSON, not JSONB)
        # This is more reliable than trying to use JSONB operators on a JSON column
        # Exclude inquiry-* categories (information requests, not applications)
        results = db.query(EmailMetadata).filter(
            EmailMetadata.ai_category.like('application-%'),
            ~EmailMetadata.ai_category.like('inquiry-%')
        ).all()
        
        logger.info(f"Found {len(results)} total applications")
        
        # Extract all unique tag names
        all_tags = set()
        apps_with_tags = 0
        for metadata in results:
            category_metadata = metadata.category_metadata or {}
            profile_tags = category_metadata.get('profile_tags', [])
            
            if isinstance(profile_tags, list) and len(profile_tags) > 0:
                apps_with_tags += 1
                for tag_obj in profile_tags:
                    if isinstance(tag_obj, dict):
                        tag_name = tag_obj.get('tag')
                        if tag_name:
                            all_tags.add(tag_name)
                    elif isinstance(tag_obj, str):
                        all_tags.add(tag_obj)
        
        logger.info(f"Found {apps_with_tags} applications with profile_tags")
        logger.info(f"Extracted {len(all_tags)} unique tags: {sorted(list(all_tags))}")
        
        return AvailableTagsResponse(tags=sorted(list(all_tags)))
    except Exception as e:
        logger.error(f"Error fetching available tags: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching available tags")


def parse_application_id(email_id_str: str, db: Session) -> tuple[Email, EmailMetadata]:
    """
    Parse application ID - supports both UUID and name-based lookup.
    
    Returns (email, metadata) tuple or raises HTTPException.
    """
    email = None
    metadata = None
    
    # First, try to parse as UUID
    try:
        email_uuid = UUID(email_id_str)
        email = db.query(Email).filter(Email.id == email_uuid).first()
        if email:
            metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == email_uuid).first()
    except ValueError:
        # Not a valid UUID - try lookup by applicant name
        # Convert slug format (e.g., "olga-krestinskaya") to search term
        search_name = email_id_str.replace("-", " ").strip()
        
        # Search for application by applicant name (case-insensitive)
        result = db.query(Email, EmailMetadata).join(
            EmailMetadata, Email.id == EmailMetadata.email_id
        ).filter(
            EmailMetadata.ai_category.like('application-%'),
            func.lower(EmailMetadata.applicant_name).ilike(f"%{search_name.lower()}%")
        ).order_by(desc(Email.date)).first()
        
        if result:
            email, metadata = result
    
    if not email:
        raise HTTPException(
            status_code=404, 
            detail=f"Application not found. Please use a valid application ID (UUID format like '58efb994-3189-4b92-9e19-595a61432c7e')."
        )
    
    if not metadata or not metadata.ai_category.startswith('application-'):
        raise HTTPException(status_code=404, detail="Application not found")
    
    return email, metadata


@router.get("/{email_id}")
async def get_application_detail(
    email_id: str,
    request: Request,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get full application details.
    
    Accepts either a UUID or applicant name for lookup.
    Requires can_review permission.
    Logs view action to audit log.
    """
    # Parse application ID (supports UUID or name-based lookup)
    email, metadata = parse_application_id(email_id, db)
    actual_email_id = email.id  # Use the actual UUID for subsequent queries
    
    # Log view action
    try:
        if request:
            log_audit_event(
                db=db,
                user_id=str(current_user.id),
                email_id=str(actual_email_id),
                action_type="view",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get('user-agent')
            )
    except Exception as e:
        logger.error(f"Failed to log view: {e}")
    
    # Get reviews
    reviews = db.query(ApplicationReview, LabMember).join(
        LabMember, ApplicationReview.lab_member_id == LabMember.id
    ).filter(ApplicationReview.email_id == actual_email_id).all()
    
    review_responses = [
        ReviewResponse(
            id=r.id,
            email_id=r.email_id,
            lab_member_id=r.lab_member_id,
            rater_name=m.full_name or m.email,
            rating=r.rating,
            comment=r.comment,  # Decrypted automatically by EncryptedText
            created_at=r.created_at,
            updated_at=r.updated_at
        )
        for r, m in reviews
    ]
    
    # Get my review
    my_review_obj = db.query(ApplicationReview).filter(
        ApplicationReview.email_id == actual_email_id,
        ApplicationReview.lab_member_id == current_user.id
    ).first()
    
    my_review = None
    if my_review_obj:
        my_review = ReviewResponse(
            id=my_review_obj.id,
            email_id=my_review_obj.email_id,
            lab_member_id=my_review_obj.lab_member_id,
            rater_name=current_user.full_name or current_user.email,
            rating=my_review_obj.rating,
            comment=my_review_obj.comment,
            created_at=my_review_obj.created_at,
            updated_at=my_review_obj.updated_at
        )
    
    # Get my private notes
    my_private_note_obj = db.query(ApplicationPrivateNote).filter(
        ApplicationPrivateNote.email_id == email_id,
        ApplicationPrivateNote.lab_member_id == current_user.id
    ).first()
    
    my_private_notes = None
    if my_private_note_obj:
        my_private_notes = my_private_note_obj.notes  # Decrypted automatically
    
    # Calculate review summary
    avg_rating = None
    num_ratings = len(reviews)
    if num_ratings > 0:
        avg_rating = sum(r.rating for r, _ in reviews) / num_ratings
    
    # Get decision
    decision_obj = db.query(ApplicationDecision, LabMember).join(
        LabMember, ApplicationDecision.admin_id == LabMember.id
    ).filter(ApplicationDecision.email_id == email_id).first()
    
    decision = None
    if decision_obj:
        d, admin = decision_obj
        decision = DecisionResponse(
            id=d.id,
            email_id=d.email_id,
            admin_id=d.admin_id,
            admin_name=admin.full_name or admin.email,
            decision=d.decision,
            notes=d.notes,  # Decrypted automatically
            decided_at=d.decided_at
        )
    
    # Extract category_metadata (non-PII)
    category_metadata = metadata.category_metadata or {}
    red_flags_dict = category_metadata.get('red_flags', {})
    profile_tags = category_metadata.get('profile_tags', [])
    information_used = category_metadata.get('information_used', {})
    tech_scores = category_metadata.get('technical_experience_scores', {})
    
    # Extract category_specific_data (PII - encrypted, but decrypted automatically)
    category_specific = metadata.category_specific_data or {}
    online_profiles = category_specific.get('online_profiles', {}) if isinstance(category_specific.get('online_profiles'), dict) else {}
    
    # Extract Google Drive links (stored as nested structure)
    gdrive_links = metadata.google_drive_links or {}
    attachment_links = []
    email_text_info = {}
    llm_response_info = {}
    
    if isinstance(gdrive_links, dict):
        # Extract attachments (list of dicts with filename, link, file_id)
        attachment_links = gdrive_links.get('attachments', [])
        
        # Extract email_text info (dict with filename, link)
        email_text_info = gdrive_links.get('email_text', {})
        
        # Extract llm_response info (dict with filename, link)
        llm_response_info = gdrive_links.get('llm_response', {})
    
    # Build response with all fields matching reprocess_applications.py export
    response_obj = ApplicationDetailResponse(
    # Basic email info
    email_id=email.id,
    date=email.date,
    from_address=email.from_address,
    from_name=email.from_name,
    curated_sender_name=metadata.curated_sender_name,
    subject=email.subject,
    message_id=email.message_id,  # RFC822 Message-ID for opening in mail clients
        
        # Classification
        category=metadata.ai_category,
        subcategory=metadata.ai_subcategory,
        confidence=metadata.ai_confidence,
        application_source=category_metadata.get('application_source'),
        
        # Scores and their explanations
        scientific_excellence_score=metadata.scientific_excellence_score or category_metadata.get('scientific_excellence_score'),
        scientific_excellence_reason=category_metadata.get('scientific_excellence_reason'),
        research_fit_score=metadata.research_fit_score,
        research_fit_reason=category_metadata.get('research_fit_reason'),
        overall_recommendation_score=metadata.overall_recommendation_score,
        recommendation_reason=category_metadata.get('recommendation_reason'),
        relevance_score=metadata.relevance_score,
        relevance_reason=metadata.relevance_reason or category_metadata.get('relevance_reason'),
        prestige_score=metadata.prestige_score or category_metadata.get('prestige_score'),
        prestige_reason=metadata.prestige_reason or category_metadata.get('prestige_reason'),
        urgency_score=metadata.ai_urgency_score,
        urgency=metadata.ai_urgency,
        urgency_reason=category_metadata.get('urgency_reason'),
        sentiment=metadata.ai_sentiment,
        
        # AI summary and reasoning
        summary=metadata.ai_summary,  # Decrypted automatically
        reasoning=category_metadata.get('reasoning'),
        
        # Status flags
        needs_reply=metadata.needs_reply,
        reply_deadline=category_metadata.get('reply_deadline'),
        reply_suggestion=category_metadata.get('reply_suggestion'),
        action_items=metadata.ai_action_items or category_metadata.get('action_items', []),
        is_followup=metadata.is_followup if metadata.is_followup else red_flags_dict.get('is_followup', False),
        followup_to_date=metadata.followup_to_date.isoformat() if metadata.followup_to_date else category_metadata.get('followup_to_date'),
        reprocessed=category_metadata.get('reprocessed_with_application_prompt', False),
        reprocessed_at=category_metadata.get('reprocessed_at'),
        was_already_processed=category_metadata.get('reprocessed_with_application_prompt', False),
        passed_filters=None,  # This is calculated dynamically in export, not stored
        
        # Metadata (JSON fields)
        profile_tags=profile_tags if profile_tags else None,
        red_flags=red_flags_dict if red_flags_dict else None,
        information_used=information_used if information_used else None,
        
        # Individual red flags
        is_mass_email=red_flags_dict.get('is_mass_email', False),
        no_research_background=red_flags_dict.get('no_research_background', False),
        irrelevant_field=red_flags_dict.get('irrelevant_field', False),
        possible_spam=red_flags_dict.get('possible_spam', False),
        insufficient_materials=red_flags_dict.get('insufficient_materials', False),
        prompt_manipulation_detected=red_flags_dict.get('prompt_manipulation_detected', False),
        prompt_manipulation_indicators=red_flags_dict.get('prompt_manipulation_indicators', []),
        is_not_application=red_flags_dict.get('is_not_application', False),
        is_not_application_reason=category_metadata.get('is_not_application_reason'),
        correct_category=category_metadata.get('correct_category'),
        is_cold_email=metadata.is_cold_email,
        
        # Technical experience (scores and evidence)
        coding_experience_score=tech_scores.get('coding_experience'),
        coding_experience_evidence=category_specific.get('coding_experience', {}).get('evidence') if isinstance(category_specific.get('coding_experience'), dict) else None,
        omics_genomics_experience_score=tech_scores.get('omics_genomics_experience'),
        omics_genomics_experience_evidence=category_specific.get('omics_genomics_experience', {}).get('evidence') if isinstance(category_specific.get('omics_genomics_experience'), dict) else None,
        medical_data_experience_score=tech_scores.get('medical_data_experience'),
        medical_data_experience_evidence=category_specific.get('medical_data_experience', {}).get('evidence') if isinstance(category_specific.get('medical_data_experience'), dict) else None,
        sequence_analysis_experience_score=tech_scores.get('sequence_analysis_algorithms_experience'),
        sequence_analysis_experience_evidence=category_specific.get('sequence_analysis_algorithms_experience', {}).get('evidence') if isinstance(category_specific.get('sequence_analysis_algorithms_experience'), dict) else None,
        image_analysis_experience_score=tech_scores.get('image_analysis_experience'),
        image_analysis_experience_evidence=category_specific.get('image_analysis_experience', {}).get('evidence') if isinstance(category_specific.get('image_analysis_experience'), dict) else None,
        
        # Event/invitation specific fields
        event_date=category_metadata.get('event_date'),
        deadline=category_metadata.get('deadline'),
        location=category_metadata.get('location'),
        time_commitment_hours=category_metadata.get('time_commitment_hours'),
        time_commitment_reason=category_metadata.get('time_commitment_reason'),
        
        # Additional info request
        should_request_additional_info=category_metadata.get('should_request_additional_info', False),
        missing_information_items=category_metadata.get('missing_information_items', []),
        potential_recommendation_score=category_metadata.get('potential_recommendation_score'),
        
        # Applicant info (PII)
        applicant_name=metadata.applicant_name,
        applicant_email=metadata.applicant_email or category_metadata.get('applicant_email') or category_specific.get('applicant_email'),
        applicant_institution=metadata.applicant_institution,
        nationality=category_specific.get('nationality'),
        highest_degree=category_specific.get('highest_degree_completed'),
        current_situation=category_specific.get('current_situation'),
        recent_thesis_title=category_specific.get('recent_thesis_title'),
        recommendation_source=category_specific.get('recommendation_source'),
        github_account=online_profiles.get('github_account'),
        linkedin_account=online_profiles.get('linkedin_account'),
        google_scholar_account=online_profiles.get('google_scholar_account'),
        
        # Evaluation details
        key_strengths=category_specific.get('key_strengths', []),
        concerns=category_specific.get('concerns', []),
        next_steps=category_specific.get('next_steps'),
        additional_notes=category_specific.get('additional_notes'),
        ai_reasoning=metadata.ai_reasoning,  # Decrypted automatically
        # Score reasoning - extract from evaluation_reasoning in category_specific_data
        score_reasoning=category_specific.get('evaluation_reasoning', {}),
        
        # Google Drive files (all links are webViewLink - viewable in browser)
        folder_path=metadata.google_drive_folder_id,  # Alias for folder_path
        email_text_file=email_text_info.get('filename'),
        email_text_link=email_text_info.get('link'),  # webViewLink - opens in Google Drive viewer
        email_text=email.body_text if email.body_text else None,  # Most recent email text content (decrypted automatically)
        attachments_list=attachment_links,  # List of dicts: {filename, link, file_id, source_email_id, source_email_date, is_from_current_email} - link is webViewLink
        
        # Consolidated attachments from category_specific_data (includes full source tracking)
        consolidated_attachments=category_specific.get('consolidated_attachments'),
        reference_letter_attachments=category_specific.get('reference_letter_attachments'),
        llm_response_file=llm_response_info.get('filename'),
        llm_response_link=llm_response_info.get('link'),  # webViewLink - opens in Google Drive viewer
        
        # AI suggestions
        suggested_folder=metadata.suggested_folder or category_metadata.get('suggested_folder'),
        suggested_labels=metadata.suggested_labels or category_metadata.get('suggested_labels', []),
        answer_options=category_metadata.get('answer_options', []),
        
        # Receipt-specific fields
        vendor=category_metadata.get('vendor'),
        amount=category_metadata.get('amount'),
        currency=category_metadata.get('currency'),
        
        # Review system fields
        reviews=review_responses,
        avg_rating=avg_rating,
        num_ratings=num_ratings,
        my_review=my_review,
        my_private_notes=my_private_notes,
        decision=decision,
        review_deadline=metadata.review_deadline,
        application_status=metadata.application_status or "pending",
        status_updated_at=metadata.application_status_updated_at
    )
    
    # Generate ETag from response content
    response_dict = response_obj.dict()
    etag_content = json.dumps(response_dict, sort_keys=True, default=str)
    etag = hashlib.md5(etag_content.encode()).hexdigest()
    
    # Check If-None-Match header
    if_none_match = request.headers.get("If-None-Match")
    if if_none_match and if_none_match.strip('"') == etag:
        return Response(status_code=304, headers={"ETag": f'"{etag}"'})
    
    # Return response with ETag and Cache-Control
    return Response(
        content=json.dumps(response_dict, default=str),
        media_type="application/json",
        headers={
            "ETag": f'"{etag}"',
            "Cache-Control": "private, max-age=60"  # Cache for 60 seconds
        }
    )


@router.post("/{email_id}/review", response_model=ReviewSummaryResponse)
async def submit_review(
    email_id: UUID,
    review: ReviewRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Submit or update a review for an application.
    
    Upsert behavior: Creates new review or updates existing one.
    Rate limited: 50 reviews per hour (members), 100 reviews per hour (admins).
    Burst handling: Allows 5 rapid reviews before throttling.
    """
    # Check rate limit (50/hour for members, 100/hour for admins)
    is_allowed, limit_info = check_rate_limit(str(current_user.id), f"/applications/{email_id}/review", "reviews", db, user_role=current_user.role)
    if not is_allowed:
        limit = limit_info.get("limit", 50) if limit_info else (100 if current_user.role == "admin" else 50)
        retry_after = limit_info.get("retry_after", 3600) if limit_info else 3600
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {limit} reviews per hour. Please try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Verify application exists
    metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == email_id).first()
    if not metadata or not metadata.ai_category.startswith('application-'):
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Get or create review with row-level locking to prevent race conditions
    # Use select_for_update to lock the row during update
    existing_review = db.query(ApplicationReview).filter(
        ApplicationReview.email_id == email_id,
        ApplicationReview.lab_member_id == current_user.id
    ).with_for_update().first()  # Lock row for update
    
    action_type = "review_submit"
    if existing_review:
        # Update existing review
        existing_review.rating = review.rating
        existing_review.comment = review.comment
        existing_review.updated_at = datetime.utcnow()
        action_type = "review_update"
        # Auto-complete pending assignments for this reviewer
        try:
            from backend.api.routes.review_assignments import complete_assignments_on_review
            complete_assignments_on_review(db, email_id, current_user.id)
        except Exception as e:
            logger.error(f"Failed to auto-complete assignments: {e}")
        db.commit()
        db.refresh(existing_review)
    else:
        # Create new review (check again to avoid race condition)
        # Another request might have created it between the query and now
        existing_review = db.query(ApplicationReview).filter(
            ApplicationReview.email_id == email_id,
            ApplicationReview.lab_member_id == current_user.id
        ).first()

        if existing_review:
            # Another request created it, update instead
            existing_review.rating = review.rating
            existing_review.comment = review.comment
            existing_review.updated_at = datetime.utcnow()
            action_type = "review_update"
        else:
            # Create new review
            existing_review = ApplicationReview(
                email_id=email_id,
                lab_member_id=current_user.id,
                rating=review.rating,
                comment=review.comment
            )
            db.add(existing_review)

        # Auto-complete pending assignments for this reviewer
        try:
            from backend.api.routes.review_assignments import complete_assignments_on_review
            complete_assignments_on_review(db, email_id, current_user.id)
        except Exception as e:
            logger.error(f"Failed to auto-complete assignments: {e}")
        db.commit()
        db.refresh(existing_review)
    
    # Log audit event
    try:
        log_audit_event(
            db=db,
            user_id=str(current_user.id),
            email_id=str(email_id),
            action_type=action_type,
            action_details={
                "rating": review.rating,
                "comment_length": len(review.comment) if review.comment else 0
            }
        )
    except Exception as e:
        logger.error(f"Failed to log review action: {e}")
    
    # Calculate and return summary
    reviews = db.query(ApplicationReview).filter(ApplicationReview.email_id == email_id).all()
    avg_rating = sum(r.rating for r in reviews) / len(reviews) if reviews else None
    num_ratings = len(reviews)
    
    return ReviewSummaryResponse(
        avg_rating=avg_rating,
        num_ratings=num_ratings
    )


@router.delete("/{email_id}/review")
async def delete_review(
    email_id: UUID,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """Delete current user's review for an application."""
    review = db.query(ApplicationReview).filter(
        ApplicationReview.email_id == email_id,
        ApplicationReview.lab_member_id == current_user.id
    ).first()
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    db.delete(review)
    db.commit()
    
    # Log audit event
    try:
        log_audit_event(
            db=db,
            user_id=str(current_user.id),
            email_id=str(email_id),
            action_type="review_delete"
        )
    except Exception as e:
        logger.error(f"Failed to log delete action: {e}")
    
    return {"message": "Review deleted successfully"}


@router.get("/{email_id}/reviews", response_model=List[ReviewResponse])
async def list_reviews(
    email_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """List all reviews for an application. Accepts UUID or applicant name."""
    # Parse application ID (supports UUID or name-based lookup)
    email, metadata = parse_application_id(email_id, db)
    actual_email_id = email.id
    
    reviews = db.query(ApplicationReview, LabMember).join(
        LabMember, ApplicationReview.lab_member_id == LabMember.id
    ).filter(ApplicationReview.email_id == actual_email_id).all()
    
    return [
        ReviewResponse(
            id=r.id,
            email_id=r.email_id,
            lab_member_id=r.lab_member_id,
            rater_name=m.full_name or m.email,
            rating=r.rating,
            comment=r.comment,
            created_at=r.created_at,
            updated_at=r.updated_at
        )
        for r, m in reviews
    ]


@router.get("/{email_id}/reviews/summary", response_model=ReviewSummaryResponse)
async def get_review_summary(
    email_id: UUID,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """Get review summary (avg rating, count) for quick updates."""
    reviews = db.query(ApplicationReview).filter(ApplicationReview.email_id == email_id).all()
    avg_rating = sum(r.rating for r in reviews) / len(reviews) if reviews else None
    num_ratings = len(reviews)
    
    return ReviewSummaryResponse(
        avg_rating=avg_rating,
        num_ratings=num_ratings
    )


@router.put("/{email_id}/private-notes", response_model=dict)
async def update_private_notes(
    email_id: UUID,
    request_data: PrivateNotesRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Save or update private notes for an application (per-user, not visible to others).
    
    Creates or updates the private note for the current user.
    """
    # Verify application exists
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Application not found")
    
    metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == email_id).first()
    if not metadata or not metadata.ai_category.startswith('application-'):
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Get or create private note
    private_note = db.query(ApplicationPrivateNote).filter(
        ApplicationPrivateNote.email_id == email_id,
        ApplicationPrivateNote.lab_member_id == current_user.id
    ).first()
    
    if private_note:
        # Update existing note
        private_note.notes = request_data.notes
        private_note.updated_at = datetime.utcnow()
    else:
        # Create new note
        private_note = ApplicationPrivateNote(
            email_id=email_id,
            lab_member_id=current_user.id,
            notes=request_data.notes
        )
        db.add(private_note)
    
    db.commit()
    db.refresh(private_note)
    
    # Log audit event
    try:
        log_audit_event(
            db=db,
            user_id=str(current_user.id),
            email_id=str(email_id),
            action_type="private_note_update",
            action_details={
                "notes_length": len(request_data.notes) if request_data.notes else 0
            }
        )
    except Exception as e:
        logger.error(f"Failed to log private note update: {e}")
    
    return {
        "success": True,
        "message": "Private notes saved successfully"
    }


@router.get("/{email_id}/previous-emails")
async def get_previous_emails(
    email_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get recent emails from the same sender (applicant), including the current email.
    
    Accepts UUID or applicant name for lookup.
    Returns list of recent emails from the same from_address, including the current email.
    Useful for seeing application history and context.
    """
    # Parse application ID (supports UUID or name-based lookup)
    current_email, metadata = parse_application_id(email_id, db)
    actual_email_id = current_email.id
    
    # Find recent emails from same sender (including current email)
    # Use applicant_email if available (for forwarded emails), otherwise use from_address
    search_email = metadata.applicant_email if metadata.applicant_email else current_email.from_address
    
    # Look back up to 60 days
    cutoff_date = current_email.date - timedelta(days=60)
    recent_emails = db.query(Email, EmailMetadata).join(
        EmailMetadata, Email.id == EmailMetadata.email_id
    ).filter(
        or_(
            Email.from_address == search_email,  # Direct emails from applicant
            EmailMetadata.applicant_email == search_email  # Forwarded emails for same applicant
        ),
        Email.date >= cutoff_date  # Within 60 days
    ).order_by(desc(Email.date)).limit(11).all()  # Limit to 11 (current + 10 previous)
    
    # Extract Google Drive links for email text
    result = []
    for email, email_metadata in recent_emails:
        email_text_link = None
        gdrive_links = email_metadata.google_drive_links or {}
        if isinstance(gdrive_links, dict):
            email_text_info = gdrive_links.get('email_text', {})
            email_text_link = email_text_info.get('link')
        
        result.append({
            "email_id": str(email.id),
            "subject": email.subject,
            "date": email.date.isoformat() if email.date else None,
            "from_address": email.from_address,
            "from_name": email.from_name,
            "email_text_link": email_text_link,
            "email_text": email.body_text or email.body_markdown,  # Include email text content
            "category": email_metadata.ai_category if email_metadata else None,
            "is_current_email": str(email.id) == str(actual_email_id)  # Mark current email
        })
    
    return result


@router.get("/export")
async def export_applications(
    export_format: str = Query("excel", alias="format", description="Export format: excel or csv"),
    exclude_pii: bool = Query(False, description="Exclude PII fields"),
    category: Optional[str] = Query(None),
    min_recommendation_score: Optional[int] = Query(None, ge=0, le=10),
    current_admin: LabMember = Depends(get_current_admin_hybrid),  # Admin only
    db: Session = Depends(get_db)
):
    """
    Export applications to Excel or CSV (admin only).
    
    Maximum 10,000 rows. Includes watermark with user and timestamp.
    """
    # Build query (same as list_applications, but without user-specific filters)
    # Exclude inquiry-* categories (information requests, not applications)
    query = db.query(Email, EmailMetadata).join(
        EmailMetadata, Email.id == EmailMetadata.email_id
    ).filter(
        EmailMetadata.ai_category.like('application-%'),
        ~EmailMetadata.ai_category.like('inquiry-%')
    )
    
    # Apply filters
    if category:
        query = query.filter(EmailMetadata.ai_category == category)
    if min_recommendation_score:
        query = query.filter(EmailMetadata.overall_recommendation_score >= min_recommendation_score)
    
    # Limit to 10,000 rows
    results = query.limit(10000).all()
    
    if len(results) >= 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Export limit exceeded. Maximum 10,000 rows. Please use filters to reduce the dataset."
        )
    
    # Prepare data for export
    export_data = []
    for email, metadata in results:
        row = {
            "email_id": str(email.id),
            "applicant_name": metadata.applicant_name if not exclude_pii else "[REDACTED]",
            "applicant_institution": metadata.applicant_institution if not exclude_pii else "[REDACTED]",
            "date": email.date.isoformat() if email.date else None,
            "category": metadata.ai_category,
            "scientific_excellence_score": metadata.scientific_excellence_score or (metadata.category_metadata.get('scientific_excellence_score') if metadata.category_metadata else None),
            "research_fit_score": metadata.research_fit_score,
            "overall_recommendation_score": metadata.overall_recommendation_score,
            "application_status": metadata.application_status or "pending",
            "review_deadline": metadata.review_deadline.isoformat() if metadata.review_deadline else None,
        }
        
        # Add PII fields only if not excluded
        if not exclude_pii:
            if metadata.category_specific_data:
                row["nationality"] = metadata.category_specific_data.get('nationality')
                row["highest_degree"] = metadata.category_specific_data.get('highest_degree')
                row["current_situation"] = metadata.category_specific_data.get('current_situation')
                row["github_account"] = metadata.category_specific_data.get('github_account')
                row["linkedin_account"] = metadata.category_specific_data.get('linkedin_account')
        
        # Get review summary
        reviews = db.query(ApplicationReview).filter(ApplicationReview.email_id == email.id).all()
        row["num_ratings"] = len(reviews)
        row["avg_rating"] = sum(r.rating for r in reviews) / len(reviews) if reviews else None
        
        # Get decision
        decision = db.query(ApplicationDecision).filter(ApplicationDecision.email_id == email.id).first()
        row["decision"] = decision.decision if decision else None
        
        export_data.append(row)
    
    # Generate filename with watermark
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"applications_export_{timestamp}_{current_admin.email.split('@')[0]}"
    
    if export_format.lower() == "csv":
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
                    "row_count": len(export_data),
                    "exclude_pii": exclude_pii
                })
            }
        )
    else:  # Excel
        try:
            import pandas as pd
            from io import BytesIO
            
            df = pd.DataFrame(export_data)
            output = BytesIO()
            
            # Create Excel with watermark in metadata
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Applications')
                worksheet = writer.sheets['Applications']
                # Add watermark note in first cell comment or header
                worksheet.cell(row=1, column=1).comment = f"Exported by {current_admin.email} on {datetime.utcnow().isoformat()}"
            
            output.seek(0)
            
            return Response(
                content=output.read(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}.xlsx"',
                    "X-Export-Metadata": json.dumps({
                        "exported_by": current_admin.email,
                        "exported_at": datetime.utcnow().isoformat(),
                        "row_count": len(export_data),
                        "exclude_pii": exclude_pii
                    })
                }
            )
        except ImportError:
            # Fallback to CSV if pandas/openpyxl not available
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Excel export requires pandas and openpyxl. Use CSV format or install dependencies."
            )

