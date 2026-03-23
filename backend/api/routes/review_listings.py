"""
Property Listing Review Routes

Endpoints for listing, viewing, and managing property listings.
Follows patterns from review_applications.py adapted for real estate domain.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc, select, and_, or_, cast
from sqlalchemy.dialects.postgresql import JSONB
from uuid import UUID
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import os
import secrets
import logging
import io

from backend.core.database import get_db
from backend.core.database.models import LabMember
from backend.core.database.models_property import (
    PropertyListing, PropertyListingSource, PropertyReview, PropertyAction,
    PropertyPrivateNote, PropertyDueDiligence, PropertyDocument,
    PropertyCollection, PropertyCollectionItem, PropertyShareToken,
)
from backend.api.review_auth import get_current_reviewer_hybrid, get_current_admin_hybrid
from backend.api.listing_schemas import (
    ListingListItem, ListingDetailResponse, SharedListingResponse,
    ReviewRequest, ReviewResponse, ReviewSummaryResponse,
    ListingActionRequest, ActionResponse,
    ListingUpdateRequest, PrivateNotesRequest,
    DueDiligenceUpdateRequest, DueDiligenceItemResponse,
    AddDocumentRequest, DocumentResponse, SourceResponse,
    CreateCollectionRequest, AddCollectionItemRequest,
    CollectionResponse, CollectionDetailResponse,
    HouzyUpdateRequest, RequestInfoRequest,
    CreateShareTokenRequest, ShareTokenResponse, ShareTokenListItem, ShareTokenPermissions,
    DashboardStatsResponse, MapDataItem,
)
from backend.api.review_middleware import log_audit_event, check_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/listings", tags=["listings"])


# =============================================================================
# Status transition map
# =============================================================================

ACTION_STATUS_MAP = {
    "interested": "under_evaluation",
    "request_info": "needs_info",
    "request_documentation": "needs_info",
    "request_viewing": "viewing_scheduled",
    "request_bank_eval": "under_evaluation",
    "make_offer": "offer_made",
    "not_interested": "decided",
    "archive": "archived",
}

# Statuses ordered by progression — don't go backwards
STATUS_ORDER = [
    "new", "insufficient_data", "needs_info", "scored",
    "under_evaluation", "viewing_scheduled", "offer_made",
    "decided", "archived",
]


def _resolve_status(current_status: str, action: str) -> str:
    """Resolve the new status based on action and current status.

    For 'request_bank_eval' → 'under_evaluation', don't regress if already
    further along (e.g. viewing_scheduled). Terminal statuses (decided, archived)
    always apply.
    """
    new_status = ACTION_STATUS_MAP[action]
    # Terminal actions always override
    if new_status in ("decided", "archived"):
        return new_status
    # Don't regress
    try:
        current_idx = STATUS_ORDER.index(current_status)
        new_idx = STATUS_ORDER.index(new_status)
        if new_idx <= current_idx:
            return current_status
    except ValueError:
        pass
    return new_status


def _record_action_internal(
    db: Session,
    listing: PropertyListing,
    user: LabMember,
    action: str,
    notes: Optional[str] = None,
) -> PropertyAction:
    """Shared helper: append a PropertyAction row and update listing_status.

    Used by both record_action() and request_info() to avoid duplicating
    the status-transition logic.
    """
    new_status = _resolve_status(listing.listing_status or "new", action)

    action_row = PropertyAction(
        listing_id=listing.id,
        acted_by=user.id,
        action=action,
        notes=notes,
        resulting_status=new_status,
    )
    db.add(action_row)

    listing.listing_status = new_status
    if new_status == "archived":
        listing.archived_at = datetime.now(timezone.utc)

    _audit(db, str(user.id), str(listing.id), "listing_action",
           action=action, resulting_status=new_status)

    return action_row


def _audit(db: Session, user_id: str, entity_id: str, action_type: str, **details):
    """Best-effort audit log — never raises."""
    try:
        log_audit_event(db=db, user_id=user_id, email_id=entity_id,
                        action_type=action_type, action_details=details or None)
    except Exception:
        pass


def _get_listing_or_404(db: Session, listing_id: str) -> PropertyListing:
    """Fetch listing by UUID or raise 404."""
    try:
        uid = UUID(listing_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid listing ID")
    listing = db.query(PropertyListing).filter(PropertyListing.id == uid).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    return listing


# =============================================================================
# STATIC ROUTES — must be registered before /{listing_id} catch-all
# =============================================================================


# ---- Collections -----------------------------------------------------------

@router.get("/collections")
async def list_collections(
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """List all property collections with item counts."""
    rows = (
        db.query(
            PropertyCollection,
            func.count(PropertyCollectionItem.id).label("item_count"),
        )
        .outerjoin(PropertyCollectionItem, PropertyCollection.id == PropertyCollectionItem.collection_id)
        .group_by(PropertyCollection.id)
        .order_by(PropertyCollection.name)
        .all()
    )
    items = []
    for coll, count in rows:
        creator_name = None
        if coll.creator:
            creator_name = coll.creator.full_name or coll.creator.email
        items.append(CollectionResponse(
            id=str(coll.id),
            name=coll.name,
            description=coll.description,
            created_by_name=creator_name,
            item_count=count,
            created_at=coll.created_at,
        ))
    return items


@router.post("/collections", status_code=201)
async def create_collection(
    req: CreateCollectionRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new property collection."""
    existing = db.query(PropertyCollection).filter(
        PropertyCollection.name == req.name.strip()
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Collection with this name already exists")

    coll = PropertyCollection(
        name=req.name.strip(),
        description=req.description,
        created_by=current_user.id,
    )
    db.add(coll)
    db.commit()
    db.refresh(coll)
    _audit(db, str(current_user.id), str(coll.id), "collection_create", name=req.name.strip())
    return CollectionResponse(
        id=str(coll.id),
        name=coll.name,
        description=coll.description,
        created_by_name=current_user.full_name or current_user.email,
        item_count=0,
        created_at=coll.created_at,
    )


@router.get("/collections/{collection_id}")
async def get_collection(
    collection_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Get collection detail with items."""
    try:
        uid = UUID(collection_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid collection ID")

    coll = db.query(PropertyCollection).filter(PropertyCollection.id == uid).first()
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")

    items_rows = (
        db.query(PropertyCollectionItem, PropertyListing)
        .join(PropertyListing, PropertyCollectionItem.listing_id == PropertyListing.id)
        .filter(PropertyCollectionItem.collection_id == uid)
        .order_by(PropertyCollectionItem.added_at.desc())
        .all()
    )
    items = []
    for ci, listing in items_rows:
        items.append({
            "item_id": str(ci.id),
            "listing_id": str(listing.id),
            "address": listing.address,
            "plz": listing.plz,
            "price_chf": listing.price_chf,
            "overall_recommendation": listing.overall_recommendation,
            "listing_status": listing.listing_status,
            "added_at": ci.added_at.isoformat() if ci.added_at else None,
            "notes": ci.notes,
        })

    creator_name = None
    if coll.creator:
        creator_name = coll.creator.full_name or coll.creator.email
    return CollectionDetailResponse(
        id=str(coll.id),
        name=coll.name,
        description=coll.description,
        created_by_name=creator_name,
        item_count=len(items),
        created_at=coll.created_at,
        items=items,
    )


@router.post("/collections/{collection_id}/items", status_code=201)
async def add_to_collection(
    collection_id: str,
    req: AddCollectionItemRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Add a listing to a collection."""
    try:
        coll_uid = UUID(collection_id)
        listing_uid = UUID(req.listing_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")

    coll = db.query(PropertyCollection).filter(PropertyCollection.id == coll_uid).first()
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")

    listing = db.query(PropertyListing).filter(PropertyListing.id == listing_uid).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")

    existing = db.query(PropertyCollectionItem).filter(
        PropertyCollectionItem.collection_id == coll_uid,
        PropertyCollectionItem.listing_id == listing_uid,
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Listing already in collection")

    item = PropertyCollectionItem(
        collection_id=coll_uid,
        listing_id=listing_uid,
        notes=req.notes,
    )
    db.add(item)
    db.commit()
    _audit(db, str(current_user.id), str(listing_uid), "collection_add", collection_id=collection_id)
    return {"success": True, "message": "Listing added to collection"}


@router.delete("/collections/{collection_id}/items/{item_id}")
async def remove_from_collection(
    collection_id: str,
    item_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Remove a listing from a collection."""
    try:
        item_uid = UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")

    item = db.query(PropertyCollectionItem).filter(
        PropertyCollectionItem.id == item_uid,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Collection item not found")

    db.delete(item)
    db.commit()
    _audit(db, str(current_user.id), str(item_uid), "collection_remove", collection_id=collection_id)
    return {"success": True, "message": "Listing removed from collection"}


# ---- Compare ---------------------------------------------------------------

@router.get("/compare")
async def compare_listings(
    ids: str = Query(..., description="Comma-separated listing UUIDs (2-3)"),
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Side-by-side comparison of 2-3 listings."""
    id_list = [s.strip() for s in ids.split(",") if s.strip()]
    if len(id_list) < 2 or len(id_list) > 3:
        raise HTTPException(status_code=400, detail="Provide 2-3 listing IDs")

    results = []
    for lid in id_list:
        listing = _get_listing_or_404(db, lid)
        results.append(_build_detail_response(db, listing, current_user))
    return results


# ---- Map -------------------------------------------------------------------

@router.get("/map")
async def get_map_data(
    scenario: Optional[str] = None,
    listing_status: Optional[str] = None,
    min_recommendation: Optional[int] = None,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Return geocoded listings for map view."""
    query = db.query(PropertyListing).filter(
        PropertyListing.geocoded == True,
        PropertyListing.latitude.isnot(None),
        PropertyListing.longitude.isnot(None),
        PropertyListing.listing_status != "archived",
    )
    if scenario:
        query = query.filter(
            cast(PropertyListing.applicable_scenarios, JSONB).contains([scenario])
        )
    if listing_status:
        query = query.filter(PropertyListing.listing_status == listing_status)
    if min_recommendation is not None:
        query = query.filter(PropertyListing.overall_recommendation >= min_recommendation)

    listings = query.all()
    return [
        MapDataItem(
            id=str(l.id),
            latitude=l.latitude,
            longitude=l.longitude,
            address=l.address,
            plz=l.plz,
            price_chf=l.price_chf,
            overall_recommendation=l.overall_recommendation,
            houzy_assessment=l.houzy_assessment,
            listing_status=l.listing_status,
            best_scenario=l.best_scenario,
            property_type=l.property_type,
        )
        for l in listings
    ]


# ---- Stats -----------------------------------------------------------------

@router.get("/stats")
async def get_stats(
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Dashboard statistics."""
    all_statuses = [
        "new", "insufficient_data", "needs_info", "scored",
        "under_evaluation", "viewing_scheduled", "offer_made",
        "decided", "archived",
    ]
    by_status = {}
    for s in all_statuses:
        by_status[s] = db.query(func.count(PropertyListing.id)).filter(
            PropertyListing.listing_status == s
        ).scalar() or 0

    total = sum(by_status.values())

    # By scenario — count listings where applicable_scenarios contains each letter
    by_scenario = {}
    for sc in ["A", "B", "C", "D"]:
        by_scenario[sc] = db.query(func.count(PropertyListing.id)).filter(
            PropertyListing.best_scenario == sc
        ).scalar() or 0

    # By source
    source_rows = (
        db.query(PropertyListing.listing_source, func.count(PropertyListing.id))
        .filter(PropertyListing.listing_source.isnot(None))
        .group_by(PropertyListing.listing_source)
        .all()
    )
    by_source = {row[0]: row[1] for row in source_rows}

    # By tier
    tier_rows = (
        db.query(PropertyListing.tier, func.count(PropertyListing.id))
        .group_by(PropertyListing.tier)
        .all()
    )
    by_tier = {str(row[0]): row[1] for row in tier_rows}

    avg_rec = db.query(func.avg(PropertyListing.overall_recommendation)).filter(
        PropertyListing.overall_recommendation.isnot(None)
    ).scalar()

    undervalued_count = db.query(func.count(PropertyListing.id)).filter(
        PropertyListing.houzy_assessment.in_(["undervalued", "below_minimum"])
    ).scalar() or 0

    pending_info = by_status.get("needs_info", 0)

    return DashboardStatsResponse(
        total_listings=total,
        by_status=by_status,
        by_scenario=by_scenario,
        by_source=by_source,
        by_tier=by_tier,
        avg_recommendation=round(avg_rec, 2) if avg_rec else None,
        undervalued_count=undervalued_count,
        pending_info=pending_info,
    )


# ---- Tags ------------------------------------------------------------------

@router.get("/tags")
async def get_available_tags(
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Return all distinct property_tags values across listings."""
    rows = db.query(PropertyListing.property_tags).filter(
        PropertyListing.property_tags.isnot(None)
    ).all()
    tag_set: set = set()
    for (tags,) in rows:
        if isinstance(tags, list):
            tag_set.update(tags)
    return sorted(tag_set)


# ---- Export -----------------------------------------------------------------

@router.get("/export")
async def export_listings(
    export_format: str = Query("xlsx", alias="format", description="xlsx or csv"),
    scenario: Optional[str] = None,
    plz: Optional[str] = None,
    listing_status: Optional[str] = None,
    min_recommendation: Optional[int] = None,
    current_user: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db),
):
    """Export filtered listings as spreadsheet."""
    query = db.query(PropertyListing)
    if scenario:
        query = query.filter(PropertyListing.best_scenario == scenario)
    if plz:
        plz_list = [p.strip() for p in plz.split(",")]
        query = query.filter(PropertyListing.plz.in_(plz_list))
    if listing_status:
        query = query.filter(PropertyListing.listing_status == listing_status)
    if min_recommendation is not None:
        query = query.filter(PropertyListing.overall_recommendation >= min_recommendation)

    listings = query.order_by(desc(PropertyListing.overall_recommendation)).limit(10000).all()

    # Build rows
    columns = [
        "address", "plz", "municipality", "price_chf", "rooms", "living_area_sqm",
        "property_type", "listing_source", "listing_status", "tier",
        "overall_recommendation", "best_scenario", "houzy_assessment",
        "price_vs_houzy_pct", "houzy_mid", "listing_url", "created_at",
    ]
    rows = []
    for l in listings:
        rows.append([getattr(l, c, None) for c in columns])

    if export_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([str(v) if v is not None else "" for v in row])
        content = output.getvalue().encode("utf-8")
        media_type = "text/csv"
        filename = "listings_export.csv"
    else:
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Listings"
            ws.append(columns)
            for row in rows:
                ws.append([str(v) if v is not None else "" for v in row])
            buf = io.BytesIO()
            wb.save(buf)
            content = buf.getvalue()
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = "listings_export.xlsx"
        except ImportError:
            raise HTTPException(status_code=500, detail="openpyxl not installed for xlsx export")

    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---- Shared listing (public, no auth) -------------------------------------

@router.get("/shared/{token}")
async def get_shared_listing(
    token: str,
    db: Session = Depends(get_db),
):
    """Public (no auth) listing detail via share token."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    share = db.query(PropertyShareToken).filter(
        PropertyShareToken.token_hash == token_hash,
    ).first()
    if not share:
        raise HTTPException(status_code=404, detail="Invalid or expired share link")
    if share.is_revoked:
        raise HTTPException(status_code=410, detail="Share link has been revoked")

    now = datetime.now(timezone.utc)
    if share.expires_at:
        expires_aware = share.expires_at if share.expires_at.tzinfo else share.expires_at.replace(tzinfo=timezone.utc)
        if expires_aware < now:
            raise HTTPException(status_code=410, detail="Share link has expired")

    if share.max_uses is not None and share.uses_count >= share.max_uses:
        raise HTTPException(status_code=410, detail="Share link usage limit reached")

    # Increment usage
    share.uses_count += 1
    share.last_used_at = now
    db.commit()

    listing = db.query(PropertyListing).filter(PropertyListing.id == share.listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")

    perms = share.permissions or {}

    # Build response — no private notes ever
    detail = _build_detail_response(db, listing, current_user=None)

    # Filter based on permissions
    if not perms.get("can_view_reviews", False):
        detail.reviews = []
        detail.avg_rating = None
        detail.num_ratings = 0
    if not perms.get("can_view_actions", False):
        detail.actions = []
        detail.latest_action = None
        detail.latest_action_at = None
    if not perms.get("can_view_documents", False):
        detail.documents = []

    detail.my_private_notes = None
    detail.my_rating = None

    creator = db.query(LabMember).filter(LabMember.id == share.created_by).first()
    creator_name = (creator.full_name or creator.email) if creator else "Unknown"

    return SharedListingResponse(
        **detail.model_dump(),
        shared_at=now,
        share_expires_at=share.expires_at,
        shared_by=creator_name,
    )


# =============================================================================
# DYNAMIC ROUTES — /{listing_id}/*
# =============================================================================


# ---- List & Filter ---------------------------------------------------------

@router.get("")
async def list_listings(
    scenario: Optional[str] = None,
    plz: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    min_sqm: Optional[int] = None,
    max_sqm: Optional[int] = None,
    min_rooms: Optional[float] = None,
    property_type: Optional[str] = None,
    listing_status: Optional[str] = None,
    tier: Optional[int] = None,
    min_recommendation: Optional[int] = Query(None, ge=1, le=10),
    houzy_assessment: Optional[str] = None,
    property_tags: Optional[str] = None,
    listing_source: Optional[str] = None,
    collection_id: Optional[str] = None,
    has_action: Optional[bool] = None,
    exclude_archived: Optional[bool] = True,
    search: Optional[str] = None,
    sort_by: Optional[str] = "overall_recommendation",
    sort_order: Optional[str] = "desc",
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1, le=10000),
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """List property listings with filters."""
    query = db.query(PropertyListing)

    # --- Filters ---
    if exclude_archived:
        query = query.filter(PropertyListing.listing_status != "archived")

    if listing_status:
        query = query.filter(PropertyListing.listing_status == listing_status)

    if tier is not None:
        query = query.filter(PropertyListing.tier == tier)

    if scenario:
        query = query.filter(PropertyListing.best_scenario == scenario)

    if plz:
        plz_list = [p.strip() for p in plz.split(",")]
        query = query.filter(PropertyListing.plz.in_(plz_list))

    if min_price is not None:
        query = query.filter(PropertyListing.price_chf >= min_price)
    if max_price is not None:
        query = query.filter(PropertyListing.price_chf <= max_price)

    if min_sqm is not None:
        query = query.filter(PropertyListing.living_area_sqm >= min_sqm)
    if max_sqm is not None:
        query = query.filter(PropertyListing.living_area_sqm <= max_sqm)

    if min_rooms is not None:
        query = query.filter(PropertyListing.rooms >= min_rooms)

    if property_type:
        query = query.filter(PropertyListing.property_type == property_type)

    if min_recommendation is not None:
        query = query.filter(PropertyListing.overall_recommendation >= min_recommendation)

    if houzy_assessment:
        query = query.filter(PropertyListing.houzy_assessment == houzy_assessment)

    if listing_source:
        query = query.filter(PropertyListing.listing_source == listing_source)

    if property_tags:
        tag_list = [t.strip() for t in property_tags.split(",")]
        for tag in tag_list:
            query = query.filter(
                cast(PropertyListing.property_tags, JSONB).contains([tag])
            )

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                PropertyListing.address.ilike(search_term),
                PropertyListing.municipality.ilike(search_term),
                PropertyListing.description.ilike(search_term),
            )
        )

    # EXISTS subquery filters
    if collection_id:
        try:
            coll_uid = UUID(collection_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid collection_id")
        coll_subq = select(PropertyCollectionItem.listing_id).where(
            PropertyCollectionItem.collection_id == coll_uid
        )
        query = query.filter(PropertyListing.id.in_(coll_subq))

    if has_action is not None:
        action_subq = select(PropertyAction.listing_id).distinct()
        if has_action:
            query = query.filter(PropertyListing.id.in_(action_subq))
        else:
            query = query.filter(PropertyListing.id.notin_(action_subq))

    # --- Count BEFORE joining subqueries (joins can inflate count) ---
    total = query.count()

    # --- Sorting ---
    allowed_sort = {
        "overall_recommendation": PropertyListing.overall_recommendation,
        "price": PropertyListing.price_chf,
        "price_per_sqm": PropertyListing.price_per_sqm,
        "created_at": PropertyListing.created_at,
        "price_vs_houzy_pct": PropertyListing.price_vs_houzy_pct,
        "days_on_market": PropertyListing.days_on_market,
    }
    sort_col = allowed_sort.get(sort_by, PropertyListing.overall_recommendation)
    if sort_order == "asc":
        query = query.order_by(asc(sort_col).nullslast())
    else:
        query = query.order_by(desc(sort_col).nullslast())

    # --- Pagination ---
    offset = (page - 1) * limit
    listings = query.offset(offset).limit(limit).all()

    # --- Batch-load computed fields (avoids N+1 but also avoids
    #     PostgreSQL-only constructs like DISTINCT ON) ---
    listing_ids = [l.id for l in listings]

    # Reviews aggregate
    review_stats = {}
    if listing_ids:
        review_rows = (
            db.query(
                PropertyReview.listing_id,
                func.avg(PropertyReview.rating).label("avg_rating"),
                func.count(PropertyReview.id).label("num_ratings"),
            )
            .filter(PropertyReview.listing_id.in_(listing_ids))
            .group_by(PropertyReview.listing_id)
            .all()
        )
        review_stats = {row[0]: (row[1], row[2]) for row in review_rows}

    # My ratings
    my_ratings = {}
    if listing_ids:
        my_rows = (
            db.query(PropertyReview.listing_id, PropertyReview.rating)
            .filter(
                PropertyReview.listing_id.in_(listing_ids),
                PropertyReview.family_member_id == current_user.id,
            )
            .all()
        )
        my_ratings = {row[0]: row[1] for row in my_rows}

    # Latest action per listing (cross-DB compatible — use max acted_at)
    latest_actions = {}
    if listing_ids:
        # Subquery: max acted_at per listing
        max_acted = (
            db.query(
                PropertyAction.listing_id,
                func.max(PropertyAction.acted_at).label("max_at"),
            )
            .filter(PropertyAction.listing_id.in_(listing_ids))
            .group_by(PropertyAction.listing_id)
            .subquery()
        )
        action_rows = (
            db.query(PropertyAction)
            .join(max_acted, and_(
                PropertyAction.listing_id == max_acted.c.listing_id,
                PropertyAction.acted_at == max_acted.c.max_at,
            ))
            .all()
        )
        latest_actions = {a.listing_id: a for a in action_rows}

    items = []
    for listing in listings:
        avg_r, num_r = review_stats.get(listing.id, (None, 0))
        my_r = my_ratings.get(listing.id)
        lat_act = latest_actions.get(listing.id)

        items.append(ListingListItem(
            id=str(listing.id),
            address=listing.address,
            plz=listing.plz,
            municipality=listing.municipality,
            price_chf=listing.price_chf,
            price_known=listing.price_known or False,
            living_area_sqm=listing.living_area_sqm,
            rooms=listing.rooms,
            year_built=listing.year_built,
            property_type=listing.property_type,
            price_per_sqm=listing.price_per_sqm,
            listing_source=listing.listing_source,
            listing_url=listing.listing_url,
            listing_status=listing.listing_status or "new",
            tier=listing.tier or 1,
            macro_location_score=listing.macro_location_score,
            micro_location_score=listing.micro_location_score,
            property_quality_score=listing.property_quality_score,
            garden_outdoor_score=listing.garden_outdoor_score,
            financial_score=listing.financial_score,
            overall_recommendation=listing.overall_recommendation,
            applicable_scenarios=listing.applicable_scenarios,
            best_scenario=listing.best_scenario,
            houzy_mid=listing.houzy_mid,
            price_vs_houzy_pct=listing.price_vs_houzy_pct,
            houzy_assessment=listing.houzy_assessment,
            highlights=listing.highlights,
            red_flags=listing.red_flags,
            property_tags=listing.property_tags,
            completeness_pct=listing.completeness_pct,
            heating_type=listing.heating_type,
            heating_cost_yearly=listing.heating_cost_yearly,
            parking_spaces=listing.parking_spaces,
            parking_included=listing.parking_included,
            num_units_in_building=listing.num_units_in_building,
            erneuerungsfonds_chf=listing.erneuerungsfonds_chf,
            nebenkosten_yearly=listing.nebenkosten_yearly,
            wertquote=listing.wertquote,
            zweitwohnung_allowed=listing.zweitwohnung_allowed,
            has_mountain_view=listing.has_mountain_view,
            has_lake_view=listing.has_lake_view,
            has_garden_access=listing.has_garden_access,
            has_terrace=listing.has_terrace,
            photo_urls=listing.photo_urls,
            first_seen=listing.first_seen,
            days_on_market=listing.days_on_market,
            avg_rating=round(float(avg_r), 2) if avg_r else None,
            num_ratings=int(num_r) if num_r else 0,
            my_rating=int(my_r) if my_r else None,
            latest_action=lat_act.action if lat_act else None,
            latest_action_at=lat_act.acted_at if lat_act else None,
            created_at=listing.created_at,
        ))

    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": max(1, (total + limit - 1) // limit),
    }


# ---- Detail ----------------------------------------------------------------

@router.get("/{listing_id}")
async def get_listing_detail(
    listing_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Full listing detail."""
    listing = _get_listing_or_404(db, listing_id)

    _audit(db, str(current_user.id), str(listing.id), "listing_view")

    return _build_detail_response(db, listing, current_user)


def _build_detail_response(
    db: Session,
    listing: PropertyListing,
    current_user: Optional[LabMember],
) -> ListingDetailResponse:
    """Build a full detail response for a listing."""
    # Reviews
    reviews_rows = (
        db.query(PropertyReview, LabMember)
        .outerjoin(LabMember, PropertyReview.family_member_id == LabMember.id)
        .filter(PropertyReview.listing_id == listing.id)
        .order_by(PropertyReview.created_at.desc())
        .all()
    )
    reviews = []
    avg_rating = None
    my_rating = None
    for rev, member in reviews_rows:
        reviews.append(ReviewResponse(
            id=str(rev.id),
            listing_id=str(rev.listing_id),
            family_member_id=str(rev.family_member_id),
            rater_name=member.full_name or member.email if member else "Unknown",
            rating=rev.rating,
            comment=rev.comment,
            created_at=rev.created_at,
            updated_at=rev.updated_at,
        ))
        if current_user and member and member.id == current_user.id:
            my_rating = rev.rating

    if reviews:
        avg_rating = round(sum(r.rating for r in reviews) / len(reviews), 2)

    # Actions (newest first)
    action_rows = (
        db.query(PropertyAction, LabMember)
        .outerjoin(LabMember, PropertyAction.acted_by == LabMember.id)
        .filter(PropertyAction.listing_id == listing.id)
        .order_by(PropertyAction.acted_at.desc())
        .all()
    )
    actions = [
        ActionResponse(
            id=str(act.id),
            action=act.action,
            notes=act.notes,
            resulting_status=act.resulting_status,
            acted_at=act.acted_at,
            acted_by_name=member.full_name or member.email if member else "Unknown",
        )
        for act, member in action_rows
    ]
    latest_action = actions[0].action if actions else None
    latest_action_at = actions[0].acted_at if actions else None

    # Private notes
    my_private_notes = None
    if current_user:
        note = db.query(PropertyPrivateNote).filter(
            PropertyPrivateNote.listing_id == listing.id,
            PropertyPrivateNote.family_member_id == current_user.id,
        ).first()
        if note:
            my_private_notes = note.notes

    # Due diligence summary
    dd_items = db.query(PropertyDueDiligence).filter(
        PropertyDueDiligence.listing_id == listing.id
    ).all()
    dd_total = len(dd_items)
    dd_answered = sum(1 for d in dd_items if d.status == "answered")
    dd_deal_breakers = sum(1 for d in dd_items if d.is_deal_breaker)

    # Documents
    doc_rows = db.query(PropertyDocument).filter(
        PropertyDocument.listing_id == listing.id
    ).order_by(PropertyDocument.uploaded_at.desc()).all()
    documents = [
        DocumentResponse(
            id=str(d.id),
            document_type=d.document_type,
            filename=d.filename,
            gdrive_link=d.gdrive_link,
            gdrive_file_id=d.gdrive_file_id,
            source=d.source,
            uploaded_at=d.uploaded_at,
        )
        for d in doc_rows
    ]

    # Sources
    source_rows = db.query(PropertyListingSource).filter(
        PropertyListingSource.listing_id == listing.id
    ).order_by(PropertyListingSource.first_seen_at).all()
    sources = [
        SourceResponse(
            source=s.source,
            listing_url=s.listing_url,
            price_at_source=s.price_at_source,
            first_seen_at=s.first_seen_at,
        )
        for s in source_rows
    ]

    return ListingDetailResponse(
        id=str(listing.id),
        address=listing.address,
        plz=listing.plz,
        municipality=listing.municipality,
        price_chf=listing.price_chf,
        price_known=listing.price_known or False,
        living_area_sqm=listing.living_area_sqm,
        rooms=listing.rooms,
        year_built=listing.year_built,
        property_type=listing.property_type,
        price_per_sqm=listing.price_per_sqm,
        listing_source=listing.listing_source,
        listing_url=listing.listing_url,
        listing_status=listing.listing_status or "new",
        tier=listing.tier or 1,
        macro_location_score=listing.macro_location_score,
        micro_location_score=listing.micro_location_score,
        property_quality_score=listing.property_quality_score,
        garden_outdoor_score=listing.garden_outdoor_score,
        financial_score=listing.financial_score,
        overall_recommendation=listing.overall_recommendation,
        applicable_scenarios=listing.applicable_scenarios,
        best_scenario=listing.best_scenario,
        houzy_mid=listing.houzy_mid,
        price_vs_houzy_pct=listing.price_vs_houzy_pct,
        houzy_assessment=listing.houzy_assessment,
        highlights=listing.highlights,
        red_flags=listing.red_flags,
        property_tags=listing.property_tags,
        completeness_pct=listing.completeness_pct,
        heating_type=listing.heating_type,
        heating_cost_yearly=listing.heating_cost_yearly,
        parking_spaces=listing.parking_spaces,
        parking_included=listing.parking_included,
        num_units_in_building=listing.num_units_in_building,
        erneuerungsfonds_chf=listing.erneuerungsfonds_chf,
        nebenkosten_yearly=listing.nebenkosten_yearly,
        wertquote=listing.wertquote,
        zweitwohnung_allowed=listing.zweitwohnung_allowed,
        has_mountain_view=listing.has_mountain_view,
        has_lake_view=listing.has_lake_view,
        has_garden_access=listing.has_garden_access,
        has_terrace=listing.has_terrace,
        photo_urls=listing.photo_urls,
        first_seen=listing.first_seen,
        days_on_market=listing.days_on_market,
        created_at=listing.created_at,
        # Detail-only fields
        description=listing.description,
        land_area_sqm=listing.land_area_sqm,
        floor=listing.floor,
        last_renovation=listing.last_renovation,
        latitude=listing.latitude,
        longitude=listing.longitude,
        outcome=listing.outcome,
        outcome_notes=listing.outcome_notes,
        houzy_min=listing.houzy_min,
        houzy_max=listing.houzy_max,
        houzy_quality_pct=listing.houzy_quality_pct,
        zustand_rating=listing.zustand_rating,
        zustand_confirmed=listing.zustand_confirmed or False,
        ausbaustandard_rating=listing.ausbaustandard_rating,
        ausbaustandard_confirmed=listing.ausbaustandard_confirmed or False,
        houzy_location_scores=listing.houzy_location_scores,
        scenario_scores=listing.scenario_scores,
        estimated_renovation_low=listing.estimated_renovation_low,
        estimated_renovation_high=listing.estimated_renovation_high,
        estimated_nebenkosten=listing.estimated_nebenkosten,
        is_zweitwohnung=listing.is_zweitwohnung,
        is_baurecht=listing.is_baurecht,
        is_stockwerkeigentum=listing.is_stockwerkeigentum,
        ai_reasoning=listing.ai_reasoning,
        missing_fields=listing.missing_fields,
        agent_name=listing.agent_name,
        agent_email=listing.agent_email,
        agent_phone=listing.agent_phone,
        agent_company=listing.agent_company,
        last_contacted_at=listing.last_contacted_at,
        contact_method=listing.contact_method,
        photos_archived=listing.photos_archived or False,
        gdrive_folder_url=listing.gdrive_folder_url,
        price_history=listing.price_history,
        # Relationships
        sources=sources,
        reviews=reviews,
        avg_rating=avg_rating,
        num_ratings=len(reviews),
        my_rating=my_rating,
        latest_action=latest_action,
        latest_action_at=latest_action_at,
        actions=actions,
        my_private_notes=my_private_notes,
        due_diligence_total=dd_total,
        due_diligence_answered=dd_answered,
        due_diligence_deal_breakers=dd_deal_breakers,
        documents=documents,
    )


# ---- Reviews ---------------------------------------------------------------

@router.post("/{listing_id}/review")
async def submit_review(
    listing_id: str,
    review: ReviewRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Submit or update a review (upsert)."""
    listing = _get_listing_or_404(db, listing_id)

    # Rate limit
    is_allowed, limit_info = check_rate_limit(
        str(current_user.id), f"/listings/{listing_id}/review", "reviews", db,
        user_role=current_user.role,
    )
    if not is_allowed:
        retry_after = limit_info.get("retry_after", 3600) if limit_info else 3600
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for reviews",
            headers={"Retry-After": str(retry_after)},
        )

    existing = db.query(PropertyReview).filter(
        PropertyReview.listing_id == listing.id,
        PropertyReview.family_member_id == current_user.id,
    ).with_for_update().first()

    action_type = "review_submit"
    if existing:
        existing.rating = review.rating
        existing.comment = review.comment
        existing.updated_at = datetime.now(timezone.utc)
        action_type = "review_update"
    else:
        new_review = PropertyReview(
            listing_id=listing.id,
            family_member_id=current_user.id,
            rating=review.rating,
            comment=review.comment,
        )
        db.add(new_review)

    db.commit()

    _audit(db, str(current_user.id), str(listing.id), action_type,
           rating=review.rating, comment_length=len(review.comment) if review.comment else 0)

    reviews = db.query(PropertyReview).filter(PropertyReview.listing_id == listing.id).all()
    avg = sum(r.rating for r in reviews) / len(reviews) if reviews else None
    return ReviewSummaryResponse(avg_rating=round(avg, 2) if avg else None, num_ratings=len(reviews))


@router.delete("/{listing_id}/review")
async def delete_review(
    listing_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Delete current user's review."""
    listing = _get_listing_or_404(db, listing_id)
    review = db.query(PropertyReview).filter(
        PropertyReview.listing_id == listing.id,
        PropertyReview.family_member_id == current_user.id,
    ).first()
    if not review:
        raise HTTPException(status_code=404, detail="No review found")

    db.delete(review)
    db.commit()

    _audit(db, str(current_user.id), str(listing.id), "review_delete")

    return {"success": True, "message": "Review deleted"}


# ---- Actions ---------------------------------------------------------------

@router.post("/{listing_id}/actions")
async def record_action(
    listing_id: str,
    req: ListingActionRequest,
    current_user: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db),
):
    """Record an action and update listing_status."""
    listing = _get_listing_or_404(db, listing_id)

    action = _record_action_internal(db, listing, current_user, req.action, req.notes)
    db.commit()
    db.refresh(action)

    actor = current_user
    return ActionResponse(
        id=str(action.id),
        action=action.action,
        notes=action.notes,
        resulting_status=action.resulting_status,
        acted_at=action.acted_at,
        acted_by_name=actor.full_name or actor.email if actor else "Unknown",
    )


@router.get("/{listing_id}/actions")
async def get_action_history(
    listing_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Return full action log for a listing, newest first."""
    listing = _get_listing_or_404(db, listing_id)
    rows = (
        db.query(PropertyAction, LabMember)
        .outerjoin(LabMember, PropertyAction.acted_by == LabMember.id)
        .filter(PropertyAction.listing_id == listing.id)
        .order_by(PropertyAction.acted_at.desc())
        .all()
    )
    return [
        ActionResponse(
            id=str(act.id),
            action=act.action,
            notes=act.notes,
            resulting_status=act.resulting_status,
            acted_at=act.acted_at,
            acted_by_name=m.full_name or m.email if m else "Unknown",
        )
        for act, m in rows
    ]


# ---- Enrichment -----------------------------------------------------------

@router.post("/{listing_id}/enrich", status_code=202)
async def trigger_enrichment(
    listing_id: str,
    background_tasks: BackgroundTasks,
    current_user: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db),
):
    """Trigger Tier 2 enrichment (scrape + Houzy + LLM scoring).

    Uses FastAPI BackgroundTasks. Frontend polls GET /{listing_id}
    to detect completion (tier changes from 1→2, status→scored).
    """
    listing = _get_listing_or_404(db, listing_id)

    # Placeholder — actual enrichment calls Plan D modules:
    #   ListingScraper.scrape(), HouzyClient.get_valuation(),
    #   ListingReprocessor._score_listing()
    # TODO: Wire up once Plan D modules are implemented

    _audit(db, str(current_user.id), str(listing.id), "enrichment_trigger", current_tier=listing.tier)
    return {
        "message": "Enrichment queued",
        "listing_id": str(listing.id),
        "current_tier": listing.tier,
    }


@router.patch("/{listing_id}/houzy")
async def update_houzy_params(
    listing_id: str,
    req: HouzyUpdateRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """User confirms/adjusts Zustand + Ausbaustandard."""
    listing = _get_listing_or_404(db, listing_id)
    listing.zustand_rating = req.zustand
    listing.zustand_confirmed = True
    listing.ausbaustandard_rating = req.ausbaustandard
    listing.ausbaustandard_confirmed = True
    db.commit()

    _audit(db, str(current_user.id), str(listing.id), "houzy_update",
           zustand=req.zustand, ausbaustandard=req.ausbaustandard)

    # TODO: Trigger Houzy re-fetch with confirmed params

    return {"success": True, "message": "Houzy parameters updated"}


# ---- Due Diligence ---------------------------------------------------------

@router.post("/{listing_id}/due-diligence", status_code=201)
async def generate_checklist(
    listing_id: str,
    current_user: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db),
):
    """Generate due diligence checklist from template."""
    import yaml

    listing = _get_listing_or_404(db, listing_id)

    # Check if checklist already exists
    existing = db.query(PropertyDueDiligence).filter(
        PropertyDueDiligence.listing_id == listing.id
    ).count()
    if existing > 0:
        raise HTTPException(status_code=409, detail="Checklist already generated for this listing")

    # Load template
    config_dir = os.getenv("CONFIG_DIR", "config")
    template_path = os.path.join(config_dir, "due_diligence_template.yaml")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=500, detail="Due diligence template not found")

    with open(template_path) as f:
        template = yaml.safe_load(f)

    scenarios = set(listing.applicable_scenarios or [])
    items_created = 0

    for category in template.get("categories", []):
        for question in category.get("questions", []):
            # Filter by scenario applicability
            q_scenarios = set(question.get("scenarios", ["A", "B", "C", "D"]))
            if scenarios and not scenarios.intersection(q_scenarios):
                continue

            item = PropertyDueDiligence(
                listing_id=listing.id,
                category=category["name"],
                question=question["question"],
                priority=question.get("priority", "normal"),
            )
            db.add(item)
            items_created += 1

    # Upgrade to tier 3
    listing.tier = 3
    db.commit()

    _audit(db, str(current_user.id), str(listing.id), "due_diligence_generate",
           items_created=items_created)
    return {"success": True, "items_created": items_created}


@router.get("/{listing_id}/due-diligence")
async def get_checklist(
    listing_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Get all due diligence items for a listing."""
    listing = _get_listing_or_404(db, listing_id)
    items = (
        db.query(PropertyDueDiligence)
        .filter(PropertyDueDiligence.listing_id == listing.id)
        .order_by(PropertyDueDiligence.category, PropertyDueDiligence.created_at)
        .all()
    )
    return [
        DueDiligenceItemResponse(
            id=str(d.id), category=d.category, question=d.question,
            priority=d.priority, status=d.status or "open", answer=d.answer,
            source=d.source, date_answered=d.date_answered,
            is_deal_breaker=d.is_deal_breaker or False, created_at=d.created_at,
        )
        for d in items
    ]


@router.patch("/{listing_id}/due-diligence/{item_id}")
async def update_checklist_item(
    listing_id: str,
    item_id: str,
    req: DueDiligenceUpdateRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Update a due diligence checklist item."""
    _get_listing_or_404(db, listing_id)  # verify listing exists
    try:
        uid = UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid item ID")

    item = db.query(PropertyDueDiligence).filter(PropertyDueDiligence.id == uid).first()
    if not item:
        raise HTTPException(status_code=404, detail="Checklist item not found")

    if req.status is not None:
        item.status = req.status
    if req.answer is not None:
        item.answer = req.answer
        item.date_answered = datetime.now(timezone.utc)
    if req.source is not None:
        item.source = req.source
    if req.is_deal_breaker is not None:
        item.is_deal_breaker = req.is_deal_breaker

    db.commit()
    db.refresh(item)

    _audit(db, str(current_user.id), str(listing_id), "due_diligence_update",
           item_id=item_id, status=item.status)

    return DueDiligenceItemResponse(
        id=str(item.id), category=item.category, question=item.question,
        priority=item.priority, status=item.status or "open", answer=item.answer,
        source=item.source, date_answered=item.date_answered,
        is_deal_breaker=item.is_deal_breaker or False, created_at=item.created_at,
    )


# ---- Documents -------------------------------------------------------------

@router.post("/{listing_id}/documents", status_code=201)
async def add_document(
    listing_id: str,
    req: AddDocumentRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Add a document to a listing."""
    listing = _get_listing_or_404(db, listing_id)
    doc = PropertyDocument(
        listing_id=listing.id,
        document_type=req.document_type,
        filename=req.filename,
        gdrive_link=req.gdrive_link,
        gdrive_file_id=req.gdrive_file_id,
        source=req.source,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    _audit(db, str(current_user.id), str(listing.id), "document_add",
           document_type=req.document_type, filename=req.filename)

    return DocumentResponse(
        id=str(doc.id), document_type=doc.document_type, filename=doc.filename,
        gdrive_link=doc.gdrive_link, gdrive_file_id=doc.gdrive_file_id,
        source=doc.source, uploaded_at=doc.uploaded_at,
    )


@router.get("/{listing_id}/documents")
async def list_documents(
    listing_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """List documents for a listing."""
    listing = _get_listing_or_404(db, listing_id)
    docs = db.query(PropertyDocument).filter(
        PropertyDocument.listing_id == listing.id
    ).order_by(PropertyDocument.uploaded_at.desc()).all()
    return [
        DocumentResponse(
            id=str(d.id), document_type=d.document_type, filename=d.filename,
            gdrive_link=d.gdrive_link, gdrive_file_id=d.gdrive_file_id,
            source=d.source, uploaded_at=d.uploaded_at,
        )
        for d in docs
    ]


# ---- Manual Edit -----------------------------------------------------------

@router.patch("/{listing_id}")
async def update_listing(
    listing_id: str,
    req: ListingUpdateRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Manually update listing fields (whitelist only)."""
    listing = _get_listing_or_404(db, listing_id)

    update_data = req.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    for field, value in update_data.items():
        setattr(listing, field, value)

    # Recalculate price_per_sqm if price or area changed
    if "price_chf" in update_data or "living_area_sqm" in update_data:
        price = listing.price_chf
        area = listing.living_area_sqm
        listing.price_per_sqm = int(price / area) if price and area and area > 0 else None

    db.commit()
    db.refresh(listing)

    _audit(db, str(current_user.id), str(listing.id), "listing_update",
           fields_updated=list(update_data.keys()))

    return {"success": True, "message": f"Updated {len(update_data)} field(s)"}


# ---- Info Requests ---------------------------------------------------------

@router.post("/{listing_id}/request-info")
async def request_info(
    listing_id: str,
    req: RequestInfoRequest,
    current_user: LabMember = Depends(get_current_admin_hybrid),
    db: Session = Depends(get_db),
):
    """Send info request to listing agent.

    Delegates to record_action() for the action log + status transition.
    """
    listing = _get_listing_or_404(db, listing_id)

    # Check for dedup — warn if already contacted recently
    warning = None
    if listing.last_contacted_at:
        last = listing.last_contacted_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days_since = (datetime.now(timezone.utc) - last).days
        if days_since < 7:
            warning = f"Agent was already contacted {days_since} day(s) ago"

    # Record contact details
    listing.last_contacted_at = datetime.now(timezone.utc)
    listing.contact_method = "email" if listing.agent_email else "web_form"

    # Delegate to shared action helper
    _record_action_internal(
        db, listing, current_user, "request_info",
        notes=req.message_template or "Info request sent to agent",
    )
    db.commit()

    # TODO: Actual email sending via SMTP / Playwright web form

    result = {
        "success": True,
        "message": "Info request recorded",
        "contact_method": listing.contact_method,
        "agent_email": listing.agent_email,
    }
    if warning:
        result["warning"] = warning
    return result


# ---- Share Tokens ----------------------------------------------------------

@router.post("/{listing_id}/share-token", status_code=201)
async def create_share_token(
    listing_id: str,
    req: CreateShareTokenRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Create a share token for unauthenticated access."""
    listing = _get_listing_or_404(db, listing_id)

    raw_token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

    expires_at = datetime.now(timezone.utc) + timedelta(hours=req.expires_in_hours)

    permissions = {
        "can_view_reviews": req.can_view_reviews,
        "can_view_actions": req.can_view_actions,
        "can_view_documents": req.can_view_documents,
    }

    share = PropertyShareToken(
        listing_id=listing.id,
        token_hash=token_hash,
        created_by=current_user.id,
        permissions=permissions,
        expires_at=expires_at,
        max_uses=req.max_uses,
    )
    db.add(share)
    db.commit()
    db.refresh(share)

    base_url = os.getenv("FRONTEND_URL", "https://immo.tailed9d1e.ts.net")
    share_url = f"{base_url}/shared/{raw_token}"

    _audit(db, str(current_user.id), str(listing.id), "share_token_create",
           expires_in_hours=req.expires_in_hours)

    return ShareTokenResponse(
        id=str(share.id),
        share_url=share_url,
        token=raw_token,
        listing_id=str(listing.id),
        permissions=ShareTokenPermissions(**permissions),
        expires_at=expires_at,
        max_uses=req.max_uses,
        uses_count=0,
        created_at=share.created_at,
        created_by_name=current_user.full_name or current_user.email,
    )


@router.get("/{listing_id}/share-tokens")
async def get_share_tokens(
    listing_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """List active share tokens for a listing."""
    listing = _get_listing_or_404(db, listing_id)
    rows = (
        db.query(PropertyShareToken, LabMember)
        .outerjoin(LabMember, PropertyShareToken.created_by == LabMember.id)
        .filter(PropertyShareToken.listing_id == listing.id)
        .order_by(PropertyShareToken.created_at.desc())
        .all()
    )
    now = datetime.now(timezone.utc)
    items = []
    for st, creator in rows:
        expires_at = st.expires_at
        is_expired = False
        if expires_at:
            ea = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
            is_expired = ea < now
        is_exhausted = st.max_uses is not None and st.uses_count >= st.max_uses

        perms = st.permissions or {}
        items.append(ShareTokenListItem(
            id=str(st.id),
            listing_id=str(st.listing_id),
            permissions=ShareTokenPermissions(
                can_view_reviews=perms.get("can_view_reviews", False),
                can_view_actions=perms.get("can_view_actions", False),
                can_view_documents=perms.get("can_view_documents", False),
            ),
            expires_at=expires_at,
            max_uses=st.max_uses,
            uses_count=st.uses_count or 0,
            is_revoked=st.is_revoked or False,
            is_expired=is_expired,
            is_exhausted=is_exhausted,
            last_used_at=st.last_used_at,
            created_at=st.created_at,
            created_by_name=creator.full_name or creator.email if creator else None,
        ))
    return items


@router.delete("/{listing_id}/share-tokens/{token_id}")
async def revoke_share_token(
    listing_id: str,
    token_id: str,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Revoke a share token."""
    try:
        uid = UUID(token_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid token ID")

    st = db.query(PropertyShareToken).filter(PropertyShareToken.id == uid).first()
    if not st:
        raise HTTPException(status_code=404, detail="Share token not found")

    st.is_revoked = True
    db.commit()
    _audit(db, str(current_user.id), str(st.listing_id), "share_token_revoke", token_id=token_id)
    return {"success": True, "message": "Share token revoked"}


# ---- Private Notes ---------------------------------------------------------

@router.put("/{listing_id}/private-notes")
async def update_private_notes(
    listing_id: str,
    req: PrivateNotesRequest,
    current_user: LabMember = Depends(get_current_reviewer_hybrid),
    db: Session = Depends(get_db),
):
    """Upsert private notes for current user."""
    listing = _get_listing_or_404(db, listing_id)

    note = db.query(PropertyPrivateNote).filter(
        PropertyPrivateNote.listing_id == listing.id,
        PropertyPrivateNote.family_member_id == current_user.id,
    ).first()

    if note:
        note.notes = req.notes
        note.updated_at = datetime.now(timezone.utc)
    else:
        note = PropertyPrivateNote(
            listing_id=listing.id,
            family_member_id=current_user.id,
            notes=req.notes,
        )
        db.add(note)

    db.commit()

    _audit(db, str(current_user.id), str(listing.id), "private_note_update",
           notes_length=len(req.notes) if req.notes else 0)

    return {"success": True, "message": "Private notes saved"}
