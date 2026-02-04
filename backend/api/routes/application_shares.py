"""
Application Share Token Routes

Endpoints for creating, listing, and revoking share tokens for applications.
Also includes a public endpoint for accessing shared application data.
"""
import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.database.models import (
    Email, EmailMetadata, ApplicationReview, ApplicationDecision,
    LabMember, ApplicationShareToken, SecurityLog
)
from backend.api.review_auth import (
    get_current_reviewer_hybrid,
    create_share_token_jwt,
    decode_share_token_jwt
)
from backend.api.review_schemas import (
    CreateShareTokenRequest,
    ShareTokenResponse,
    ShareTokenListItem,
    SharedApplicationResponse,
    ShareTokenPermissions,
    ReviewResponse,
    DecisionResponse
)

logger = logging.getLogger(__name__)

# Authenticated routes for managing share tokens
router = APIRouter(prefix="/applications", tags=["application-shares"])

# Public routes for accessing shared applications
public_router = APIRouter(prefix="/shared", tags=["shared-applications"])


def get_share_base_url() -> str:
    """Get the base URL for share links from environment or use default."""
    return os.getenv("SHARE_BASE_URL", os.getenv("FRONTEND_URL", "http://localhost:3000"))


def hash_token(token: str) -> str:
    """Create SHA256 hash of a token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()


@router.post("/{email_id}/share", response_model=ShareTokenResponse)
async def create_share_token(
    email_id: UUID,
    request_body: CreateShareTokenRequest,
    request: Request,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Create a shareable link for an application.

    The returned share_url can be shared with external parties (collaborators,
    committee members) to view the application without requiring portal auth.

    Requires can_review permission.

    **Security:**
    - Token is a signed JWT with expiration
    - SHA256 hash stored in DB (not plain token)
    - Revocable at any time
    - Usage tracked (count, last access, IP)
    """
    # Verify application exists
    email = db.query(Email).filter(Email.id == email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Application not found")

    metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == email_id).first()
    if not metadata or not metadata.ai_category or not metadata.ai_category.startswith('application-'):
        raise HTTPException(status_code=404, detail="Application not found")

    # Calculate expiration
    expires_at = datetime.utcnow() + timedelta(hours=request_body.expires_in_hours)

    # Build permissions dict
    permissions = {
        "can_view_reviews": request_body.can_view_reviews,
        "can_view_decision": request_body.can_view_decision
    }

    # Create the share token record first to get the ID
    import uuid
    share_id = uuid.uuid4()

    # Create JWT with share details
    token = create_share_token_jwt(
        share_id=str(share_id),
        email_id=str(email_id),
        permissions=permissions,
        expires_at=expires_at
    )

    # Store hash of token (not the token itself)
    token_hash = hash_token(token)

    # Create database record
    share_token = ApplicationShareToken(
        id=share_id,
        email_id=email_id,
        created_by=current_user.id,
        token_hash=token_hash,
        permissions=permissions,
        expires_at=expires_at,
        max_uses=request_body.max_uses,
        uses_count=0,
        is_revoked=False
    )
    db.add(share_token)
    db.commit()
    db.refresh(share_token)

    # Build share URL
    base_url = get_share_base_url()
    share_url = f"{base_url}/shared/{token}"

    logger.info(f"Share token created for application {email_id} by user {current_user.id}")

    return ShareTokenResponse(
        id=share_token.id,
        share_url=share_url,
        token=token,  # Only returned on creation
        email_id=share_token.email_id,
        permissions=ShareTokenPermissions(**permissions),
        expires_at=share_token.expires_at,
        max_uses=share_token.max_uses,
        uses_count=share_token.uses_count,
        created_at=share_token.created_at,
        created_by_name=current_user.full_name or current_user.email
    )


@router.get("/{email_id}/shares", response_model=List[ShareTokenListItem])
async def list_share_tokens(
    email_id: UUID,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    List all share tokens for an application.

    Returns all tokens (active, expired, revoked) for the application.
    Does NOT return the raw token - only metadata.

    Requires can_review permission.
    """
    # Verify application exists
    metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == email_id).first()
    if not metadata or not metadata.ai_category or not metadata.ai_category.startswith('application-'):
        raise HTTPException(status_code=404, detail="Application not found")

    # Get all share tokens for this application
    tokens = db.query(ApplicationShareToken, LabMember).outerjoin(
        LabMember, ApplicationShareToken.created_by == LabMember.id
    ).filter(
        ApplicationShareToken.email_id == email_id
    ).order_by(ApplicationShareToken.created_at.desc()).all()

    now = datetime.utcnow()
    result = []
    for share_token, creator in tokens:
        is_expired = share_token.expires_at < now
        is_exhausted = share_token.max_uses is not None and share_token.uses_count >= share_token.max_uses

        result.append(ShareTokenListItem(
            id=share_token.id,
            email_id=share_token.email_id,
            permissions=ShareTokenPermissions(**share_token.permissions),
            expires_at=share_token.expires_at,
            max_uses=share_token.max_uses,
            uses_count=share_token.uses_count,
            is_revoked=share_token.is_revoked,
            is_expired=is_expired,
            is_exhausted=is_exhausted,
            last_used_at=share_token.last_used_at,
            created_at=share_token.created_at,
            created_by_name=creator.full_name or creator.email if creator else None
        ))

    return result


@router.delete("/{email_id}/shares/{share_id}")
async def revoke_share_token(
    email_id: UUID,
    share_id: UUID,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db)
):
    """
    Revoke a share token.

    Once revoked, the share link will no longer work.
    Revocation is permanent and cannot be undone.

    Requires can_review permission.
    """
    # Find the share token
    share_token = db.query(ApplicationShareToken).filter(
        ApplicationShareToken.id == share_id,
        ApplicationShareToken.email_id == email_id
    ).first()

    if not share_token:
        raise HTTPException(status_code=404, detail="Share token not found")

    if share_token.is_revoked:
        raise HTTPException(status_code=400, detail="Share token is already revoked")

    # Revoke the token
    share_token.is_revoked = True
    share_token.revoked_at = datetime.utcnow()
    share_token.revoked_by = current_user.id
    db.commit()

    logger.info(f"Share token {share_id} revoked by user {current_user.id}")

    return {"status": "revoked", "share_id": str(share_id)}


# ============================================================================
# Public Endpoint - No Authentication Required
# ============================================================================

@public_router.get("/{token}")
async def get_shared_application(
    token: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Access shared application data via share token.

    **No authentication required.**

    Returns filtered application data based on token permissions:
    - Always includes: applicant info, scores, AI analysis, attachments
    - Conditional: reviews (if can_view_reviews), decision (if can_view_decision)
    - NEVER includes: email body, from_address, private notes

    **Rate limited:** 30 requests per minute per IP.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Decode and validate JWT
    try:
        payload = decode_share_token_jwt(token)
    except HTTPException as e:
        # Log security event for invalid token
        _log_share_access(db, None, client_ip, "invalid_token", {"error": e.detail})
        raise

    share_id = payload.get("share_id")
    email_id = payload.get("email_id")
    permissions = payload.get("permissions", {})

    # Verify token exists in database and is valid
    share_token = db.query(ApplicationShareToken).filter(
        ApplicationShareToken.id == share_id
    ).first()

    if not share_token:
        _log_share_access(db, None, client_ip, "token_not_found", {"share_id": share_id})
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found"
        )

    # Check if revoked
    if share_token.is_revoked:
        _log_share_access(db, share_id, client_ip, "token_revoked", {})
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Share link has been revoked"
        )

    # Check if expired (redundant with JWT, but defense in depth)
    now = datetime.utcnow()
    if share_token.expires_at < now:
        _log_share_access(db, share_id, client_ip, "token_expired", {})
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Share link has expired"
        )

    # Check if usage limit exceeded
    if share_token.max_uses is not None and share_token.uses_count >= share_token.max_uses:
        _log_share_access(db, share_id, client_ip, "usage_limit_exceeded", {"max_uses": share_token.max_uses})
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Share link usage limit exceeded"
        )

    # Verify email_id matches (defense in depth)
    if str(share_token.email_id) != email_id:
        _log_share_access(db, share_id, client_ip, "email_id_mismatch", {})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid share token"
        )

    # Get application data
    email = db.query(Email).filter(Email.id == share_token.email_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Application not found")

    metadata = db.query(EmailMetadata).filter(EmailMetadata.email_id == share_token.email_id).first()
    if not metadata:
        raise HTTPException(status_code=404, detail="Application not found")

    # Update usage tracking
    share_token.uses_count += 1
    share_token.last_used_at = now
    share_token.last_used_ip = client_ip
    db.commit()

    # Log successful access
    _log_share_access(db, share_id, client_ip, "access_granted", {"uses_count": share_token.uses_count})

    # Build filtered response
    return _build_shared_response(
        db=db,
        email=email,
        metadata=metadata,
        permissions=permissions,
        share_token=share_token
    )


def _log_share_access(
    db: Session,
    share_id: Optional[str],
    ip_address: str,
    event_type: str,
    details: dict
):
    """Log share access to SecurityLog (no user_id since anonymous)."""
    try:
        security_log = SecurityLog(
            user_id=None,  # Anonymous access
            event_type=f"share_{event_type}",
            endpoint="/shared/",
            ip_address=ip_address,
            details={
                "share_id": str(share_id) if share_id else None,
                **details
            }
        )
        db.add(security_log)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log share access: {e}")
        # Don't fail the request if logging fails
        db.rollback()


def _build_shared_response(
    db: Session,
    email: Email,
    metadata: EmailMetadata,
    permissions: dict,
    share_token: ApplicationShareToken
) -> SharedApplicationResponse:
    """Build filtered application response for shared view."""
    # Look up the creator's name for shared_by field
    creator = db.query(LabMember).filter(LabMember.id == share_token.created_by).first()
    shared_by = creator.full_name or creator.email if creator else "Unknown"

    # Extract category_metadata (non-PII)
    category_metadata = metadata.category_metadata or {}
    red_flags_dict = category_metadata.get('red_flags', {})
    profile_tags = category_metadata.get('profile_tags', [])
    tech_scores = category_metadata.get('technical_experience_scores', {})

    # Extract category_specific_data (PII - encrypted, but decrypted automatically)
    category_specific = metadata.category_specific_data or {}
    online_profiles = category_specific.get('online_profiles', {}) if isinstance(category_specific.get('online_profiles'), dict) else {}

    # Extract Google Drive links for attachments
    gdrive_links = metadata.google_drive_links or {}
    attachment_links = gdrive_links.get('attachments', []) if isinstance(gdrive_links, dict) else []

    # Conditional: reviews
    reviews = None
    avg_rating = None
    num_ratings = None
    if permissions.get("can_view_reviews"):
        review_results = db.query(ApplicationReview, LabMember).join(
            LabMember, ApplicationReview.lab_member_id == LabMember.id
        ).filter(ApplicationReview.email_id == email.id).all()

        reviews = [
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
            for r, m in review_results
        ]
        num_ratings = len(reviews)
        if num_ratings > 0:
            avg_rating = sum(r.rating for r in reviews) / num_ratings

    # Conditional: decision
    decision = None
    application_status = None
    if permissions.get("can_view_decision"):
        decision_obj = db.query(ApplicationDecision, LabMember).join(
            LabMember, ApplicationDecision.admin_id == LabMember.id
        ).filter(ApplicationDecision.email_id == email.id).first()

        if decision_obj:
            d, admin = decision_obj
            decision = DecisionResponse(
                id=d.id,
                email_id=d.email_id,
                admin_id=d.admin_id,
                admin_name=admin.full_name or admin.email,
                decision=d.decision,
                notes=d.notes,
                decided_at=d.decided_at
            )
        application_status = metadata.application_status or "pending"

    return SharedApplicationResponse(
        # Application identifier
        email_id=email.id,

        # Basic info (always included)
        applicant_name=metadata.applicant_name,
        applicant_institution=metadata.applicant_institution,
        nationality=category_specific.get('nationality'),
        highest_degree=category_specific.get('highest_degree_completed'),
        current_situation=category_specific.get('current_situation'),
        date=email.date,

        # Classification
        category=metadata.ai_category,
        subcategory=metadata.ai_subcategory,

        # Online profiles (always included)
        github_account=online_profiles.get('github_account'),
        linkedin_account=online_profiles.get('linkedin_account'),
        google_scholar_account=online_profiles.get('google_scholar_account'),

        # Scores (always included)
        scientific_excellence_score=metadata.scientific_excellence_score or category_metadata.get('scientific_excellence_score'),
        scientific_excellence_reason=category_metadata.get('scientific_excellence_reason'),
        research_fit_score=metadata.research_fit_score,
        research_fit_reason=category_metadata.get('research_fit_reason'),
        overall_recommendation_score=metadata.overall_recommendation_score,
        recommendation_reason=category_metadata.get('recommendation_reason'),

        # Technical experience (always included)
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

        # AI evaluation (always included)
        summary=metadata.ai_summary,
        key_strengths=category_specific.get('key_strengths', []),
        concerns=category_specific.get('concerns', []),
        next_steps=category_specific.get('next_steps'),
        profile_tags=profile_tags if profile_tags else None,
        red_flags=red_flags_dict if red_flags_dict else None,

        # Attachments (always included - links only, no email body)
        attachments_list=attachment_links,
        consolidated_attachments=category_specific.get('consolidated_attachments'),
        reference_letter_attachments=category_specific.get('reference_letter_attachments'),

        # Conditional fields
        reviews=reviews,
        avg_rating=avg_rating,
        num_ratings=num_ratings,
        decision=decision,
        application_status=application_status,

        # Share metadata
        shared_at=datetime.utcnow(),
        share_expires_at=share_token.expires_at,
        shared_by=shared_by
    )
