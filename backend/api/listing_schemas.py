"""
Pydantic schemas for Property Listing Review System

Follows patterns from review_schemas.py but adapted for real estate domain.
No encryption — plain text fields (personal database about buildings, not people).
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from uuid import UUID


# =============================================================================
# Request Schemas
# =============================================================================

class ReviewRequest(BaseModel):
    """Request to submit/update a property review."""
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5 stars")
    comment: Optional[str] = Field(None, max_length=2000, description="Optional comment")


class ListingActionRequest(BaseModel):
    """Request to record an action on a listing."""
    action: str = Field(..., description="Action type")
    notes: Optional[str] = Field(None, max_length=5000, description="Optional notes")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {
            "interested", "request_info", "request_documentation",
            "request_viewing", "request_bank_eval", "make_offer",
            "not_interested", "archive",
        }
        if v not in valid:
            raise ValueError(f"Invalid action '{v}'. Must be one of: {', '.join(sorted(valid))}")
        return v


class ListingUpdateRequest(BaseModel):
    """Request to manually update listing fields.

    Only whitelisted fields — scores and status are set by
    enrichment and actions respectively.
    """
    address: Optional[str] = None
    plz: Optional[str] = None
    municipality: Optional[str] = None
    canton: Optional[str] = None
    price_chf: Optional[int] = None
    price_known: Optional[bool] = None
    living_area_sqm: Optional[int] = None
    land_area_sqm: Optional[int] = None
    rooms: Optional[float] = None
    year_built: Optional[int] = None
    last_renovation: Optional[int] = None
    floor: Optional[int] = None
    property_type: Optional[str] = None
    description: Optional[str] = None
    heating_type: Optional[str] = None
    parking_spaces: Optional[int] = None
    parking_included: Optional[bool] = None
    agent_name: Optional[str] = None
    agent_email: Optional[str] = None
    agent_phone: Optional[str] = None
    agent_company: Optional[str] = None
    outcome: Optional[str] = None
    outcome_notes: Optional[str] = None


class PrivateNotesRequest(BaseModel):
    """Request to save/update private notes."""
    notes: str = Field(..., max_length=10000, description="Private notes")


class DueDiligenceUpdateRequest(BaseModel):
    """Request to update a due diligence checklist item."""
    status: Optional[str] = Field(None, description="open, answered, not_applicable, deal_breaker")
    answer: Optional[str] = Field(None, max_length=5000)
    source: Optional[str] = Field(None, description="agent, document, research, viewing")
    is_deal_breaker: Optional[bool] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            valid = {"open", "answered", "not_applicable", "deal_breaker"}
            if v not in valid:
                raise ValueError(f"Invalid status '{v}'. Must be one of: {', '.join(sorted(valid))}")
        return v


class AddDocumentRequest(BaseModel):
    """Request to add a document to a listing."""
    document_type: str = Field(..., description="verkaufsbrochure, grundbuch, etc.")
    filename: str = Field(...)
    gdrive_link: Optional[str] = None
    gdrive_file_id: Optional[str] = None
    source: Optional[str] = Field(None, description="agent, portal, user, houzy")


class CreateCollectionRequest(BaseModel):
    """Request to create a property collection."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)


class AddCollectionItemRequest(BaseModel):
    """Request to add a listing to a collection."""
    listing_id: str = Field(..., description="UUID of listing to add")
    notes: Optional[str] = Field(None, max_length=2000)


class HouzyUpdateRequest(BaseModel):
    """Request to update Houzy parameters."""
    zustand: float = Field(..., ge=1.0, le=5.0)
    ausbaustandard: float = Field(..., ge=1.0, le=5.0)


class CreateShareTokenRequest(BaseModel):
    """Request to create a share token for a listing."""
    expires_in_hours: int = Field(168, ge=1, le=720, description="Hours until expiry (1-720)")
    max_uses: Optional[int] = Field(None, ge=1, le=1000)
    can_view_reviews: bool = Field(True, description="Allow viewing family member reviews")
    can_view_actions: bool = Field(True, description="Allow viewing action history")
    can_view_documents: bool = Field(False, description="Allow viewing linked documents")


class RequestInfoRequest(BaseModel):
    """Request to send info request to listing agent."""
    message_template: Optional[str] = Field(None, max_length=5000,
                                             description="Custom message template (uses default if omitted)")


# =============================================================================
# Response Schemas
# =============================================================================

class ReviewResponse(BaseModel):
    """Property review response."""
    id: str
    listing_id: str
    family_member_id: str
    rater_name: str
    rating: int
    comment: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ReviewSummaryResponse(BaseModel):
    """Review summary (avg rating, count)."""
    avg_rating: Optional[float]
    num_ratings: int


class ActionResponse(BaseModel):
    """Property action response."""
    id: str
    action: str
    notes: Optional[str]
    resulting_status: Optional[str]
    acted_at: datetime
    acted_by_name: str

    model_config = {"from_attributes": True}


class DueDiligenceItemResponse(BaseModel):
    """Due diligence checklist item response."""
    id: str
    category: str
    question: str
    priority: Optional[str]
    status: str
    answer: Optional[str]
    source: Optional[str]
    date_answered: Optional[datetime]
    is_deal_breaker: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    """Property document response."""
    id: str
    document_type: str
    filename: str
    gdrive_link: Optional[str]
    gdrive_file_id: Optional[str]
    source: Optional[str]
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class SourceResponse(BaseModel):
    """Property listing source response."""
    source: str
    listing_url: Optional[str]
    price_at_source: Optional[int]
    first_seen_at: datetime

    model_config = {"from_attributes": True}


class CollectionResponse(BaseModel):
    """Property collection response."""
    id: str
    name: str
    description: Optional[str]
    created_by_name: Optional[str]
    item_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class CollectionDetailResponse(CollectionResponse):
    """Collection with items."""
    items: List[Dict[str, Any]]  # [{listing_id, address, price_chf, added_at, notes}]


class ShareTokenPermissions(BaseModel):
    """Permissions for a listing share token."""
    can_view_reviews: bool = False
    can_view_actions: bool = False
    can_view_documents: bool = False


class ShareTokenResponse(BaseModel):
    """Response when creating a share token."""
    id: str
    share_url: str
    token: str  # Raw token — only returned on creation
    listing_id: str
    permissions: ShareTokenPermissions
    expires_at: Optional[datetime]
    max_uses: Optional[int]
    uses_count: int
    created_at: datetime
    created_by_name: Optional[str]

    model_config = {"from_attributes": True}


class ShareTokenListItem(BaseModel):
    """Share token in list response (no raw token)."""
    id: str
    listing_id: str
    permissions: ShareTokenPermissions
    expires_at: Optional[datetime]
    max_uses: Optional[int]
    uses_count: int
    is_revoked: bool
    is_expired: bool
    is_exhausted: bool
    last_used_at: Optional[datetime]
    created_at: datetime
    created_by_name: Optional[str]

    model_config = {"from_attributes": True}


# =============================================================================
# List / Detail Response Schemas
# =============================================================================

class ListingListItem(BaseModel):
    """Property listing summary for list view."""
    id: str
    address: Optional[str] = None
    plz: Optional[str] = None
    municipality: Optional[str] = None
    price_chf: Optional[int] = None
    price_known: bool = False
    living_area_sqm: Optional[int] = None
    rooms: Optional[float] = None
    year_built: Optional[int] = None
    property_type: Optional[str] = None
    price_per_sqm: Optional[int] = None
    listing_source: Optional[str] = None
    listing_url: Optional[str] = None
    listing_status: str = "new"
    tier: int = 1

    # LLM Scores
    macro_location_score: Optional[int] = None
    micro_location_score: Optional[int] = None
    property_quality_score: Optional[int] = None
    garden_outdoor_score: Optional[int] = None
    financial_score: Optional[int] = None
    overall_recommendation: Optional[int] = None

    # Scenario
    applicable_scenarios: Optional[List[str]] = None
    best_scenario: Optional[str] = None

    # Houzy
    houzy_mid: Optional[int] = None
    price_vs_houzy_pct: Optional[float] = None
    houzy_assessment: Optional[str] = None

    # Quick info
    highlights: Optional[List[str]] = None
    red_flags: Optional[List[str]] = None
    property_tags: Optional[List[str]] = None
    completeness_pct: Optional[int] = None
    heating_type: Optional[str] = None
    heating_cost_yearly: Optional[int] = None
    parking_spaces: Optional[int] = None
    parking_included: Optional[bool] = None
    num_units_in_building: Optional[int] = None
    erneuerungsfonds_chf: Optional[int] = None
    nebenkosten_yearly: Optional[int] = None
    wertquote: Optional[str] = None
    zweitwohnung_allowed: Optional[bool] = None
    has_mountain_view: Optional[bool] = None
    has_lake_view: Optional[bool] = None
    has_garden_access: Optional[bool] = None
    has_terrace: Optional[bool] = None

    photo_urls: Optional[List[str]] = None
    first_seen: Optional[datetime] = None
    days_on_market: Optional[int] = None

    # Reviews (computed from subquery)
    avg_rating: Optional[float] = None
    num_ratings: int = 0
    my_rating: Optional[int] = None

    # Latest action
    latest_action: Optional[str] = None
    latest_action_at: Optional[datetime] = None

    created_at: datetime

    model_config = {"from_attributes": True}


class ListingDetailResponse(ListingListItem):
    """Full listing detail — extends ListingListItem."""

    # Additional property details
    description: Optional[str] = None
    land_area_sqm: Optional[int] = None
    floor: Optional[int] = None
    last_renovation: Optional[int] = None

    # Geolocation
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Outcome
    outcome: Optional[str] = None
    outcome_notes: Optional[str] = None

    # Full Houzy data
    houzy_min: Optional[int] = None
    houzy_max: Optional[int] = None
    houzy_quality_pct: Optional[int] = None
    zustand_rating: Optional[float] = None
    zustand_confirmed: bool = False
    ausbaustandard_rating: Optional[float] = None
    ausbaustandard_confirmed: bool = False
    houzy_location_scores: Optional[Dict[str, Any]] = None

    # Full scenario scores
    scenario_scores: Optional[Dict[str, Any]] = None

    # Costs
    estimated_renovation_low: Optional[int] = None
    estimated_renovation_high: Optional[int] = None
    estimated_nebenkosten: Optional[int] = None

    # Financing
    is_zweitwohnung: Optional[bool] = None
    is_baurecht: Optional[bool] = None
    is_stockwerkeigentum: Optional[bool] = None

    # AI analysis
    ai_reasoning: Optional[str] = None
    missing_fields: Optional[List[str]] = None

    # Agent
    agent_name: Optional[str] = None
    agent_email: Optional[str] = None
    agent_phone: Optional[str] = None
    agent_company: Optional[str] = None
    last_contacted_at: Optional[datetime] = None
    contact_method: Optional[str] = None

    # Photos
    photos_archived: bool = False

    # Google Drive
    gdrive_folder_url: Optional[str] = None

    # Price history
    price_history: Optional[List[Dict[str, Any]]] = None

    # Sources
    sources: List[SourceResponse] = []

    # Reviews (all family members)
    reviews: List[ReviewResponse] = []

    # Action history (newest first)
    actions: List[ActionResponse] = []

    # Private notes (current user only)
    my_private_notes: Optional[str] = None

    # Due diligence summary
    due_diligence_total: int = 0
    due_diligence_answered: int = 0
    due_diligence_deal_breakers: int = 0

    # Documents
    documents: List[DocumentResponse] = []


class SharedListingResponse(ListingDetailResponse):
    """Listing detail for shared (unauthenticated) view.

    Excludes private_notes. Reviews/actions/documents conditional on token permissions.
    """
    shared_at: datetime
    share_expires_at: Optional[datetime] = None
    shared_by: str


# =============================================================================
# Stats / Map / Compare
# =============================================================================

class DashboardStatsResponse(BaseModel):
    """Dashboard statistics."""
    total_listings: int
    by_status: Dict[str, int]
    by_scenario: Dict[str, int]
    by_source: Dict[str, int]
    by_tier: Dict[str, int]
    avg_recommendation: Optional[float]
    undervalued_count: int
    pending_info: int


class MapDataItem(BaseModel):
    """Geocoded listing for map view."""
    id: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    plz: Optional[str] = None
    price_chf: Optional[int] = None
    overall_recommendation: Optional[int] = None
    houzy_assessment: Optional[str] = None
    listing_status: str
    best_scenario: Optional[str] = None
    property_type: Optional[str] = None

    model_config = {"from_attributes": True}
