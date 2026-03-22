"""
Property Listing Database Models

Tables for the real estate portal — property listings, reviews, decisions,
due diligence tracking, and document management.

Separate from the application models (models.py) — used on the real-estate branch
with its own database deployment.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, UniqueConstraint, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from backend.core.database.models import Base


# =============================================================================
# CORE: Property Listing
# =============================================================================

class PropertyListing(Base):
    __tablename__ = "property_listings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey("emails.id"), nullable=True)

    # --- Identity & Dedup ---
    address = Column(String, nullable=True)
    plz = Column(String, nullable=True)
    municipality = Column(String, nullable=True)
    canton = Column(String, nullable=True)
    listing_url = Column(String, nullable=True)
    listing_source = Column(String, nullable=True)
    listing_ref_id = Column(String, nullable=True)
    dedup_hash = Column(String, unique=True, nullable=True)

    # --- Property Details ---
    property_type = Column(String, nullable=True)       # "apartment", "single_family", "multi_family"
    price_chf = Column(Integer, nullable=True)
    price_known = Column(Boolean, default=False)
    living_area_sqm = Column(Integer, nullable=True)
    land_area_sqm = Column(Integer, nullable=True)
    rooms = Column(Float, nullable=True)
    year_built = Column(Integer, nullable=True)
    last_renovation = Column(Integer, nullable=True)
    price_per_sqm = Column(Integer, nullable=True)
    floor = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)

    # --- Property Specifics ---
    heating_type = Column(String, nullable=True)        # "oil", "heat_pump", "gas", "district", "wood"
    heating_cost_yearly = Column(Integer, nullable=True)
    parking_spaces = Column(Integer, nullable=True)
    parking_included = Column(Boolean, nullable=True)
    num_units_in_building = Column(Integer, nullable=True)
    wertquote = Column(String, nullable=True)           # STWE share e.g. "44/1000"
    erneuerungsfonds_chf = Column(Integer, nullable=True)
    nebenkosten_yearly = Column(Integer, nullable=True)
    zweitwohnung_allowed = Column(Boolean, nullable=True)
    has_mountain_view = Column(Boolean, nullable=True)
    has_lake_view = Column(Boolean, nullable=True)
    has_garden_access = Column(Boolean, nullable=True)
    has_terrace = Column(Boolean, nullable=True)

    # --- Financing ---
    is_zweitwohnung = Column(Boolean, nullable=True)
    is_baurecht = Column(Boolean, nullable=True)
    is_stockwerkeigentum = Column(Boolean, nullable=True)

    # --- LLM Scores (1-10, queryable) ---
    macro_location_score = Column(Integer, nullable=True)
    micro_location_score = Column(Integer, nullable=True)
    property_quality_score = Column(Integer, nullable=True)
    garden_outdoor_score = Column(Integer, nullable=True)
    financial_score = Column(Integer, nullable=True)
    overall_recommendation = Column(Integer, nullable=True)

    # --- Scenarios ---
    applicable_scenarios = Column(JSON, nullable=True)  # ["A", "D"]
    scenario_scores = Column(JSON, nullable=True)       # {"A": {...}, "D": {...}}
    best_scenario = Column(String, nullable=True)

    # --- Houzy/FPRE ---
    houzy_property_id = Column(String, nullable=True)
    houzy_min = Column(Integer, nullable=True)
    houzy_mid = Column(Integer, nullable=True)
    houzy_max = Column(Integer, nullable=True)
    houzy_quality_pct = Column(Integer, nullable=True)
    price_vs_houzy_pct = Column(Float, nullable=True)
    houzy_assessment = Column(String, nullable=True)    # "undervalued"/"fair"/"overpriced"
    zustand_rating = Column(Float, nullable=True)       # 1-5
    zustand_confirmed = Column(Boolean, default=False)
    ausbaustandard_rating = Column(Float, nullable=True)
    ausbaustandard_confirmed = Column(Boolean, default=False)
    houzy_location_scores = Column(JSON, nullable=True) # {gesamt, besonnung, sicht, ...}
    houzy_fetched_at = Column(DateTime, nullable=True)

    # --- Enrichment ---
    tier = Column(Integer, default=1)                   # 1=triage, 2=enriched, 3=due_diligence
    completeness_pct = Column(Integer, nullable=True)
    missing_fields = Column(JSON, nullable=True)
    estimated_renovation_low = Column(Integer, nullable=True)
    estimated_renovation_high = Column(Integer, nullable=True)
    estimated_nebenkosten = Column(Integer, nullable=True)

    # --- LLM Analysis ---
    highlights = Column(JSON, nullable=True)
    red_flags = Column(JSON, nullable=True)
    property_tags = Column(JSON, nullable=True)
    ai_reasoning = Column(Text, nullable=True)
    ai_full_response = Column(JSON, nullable=True)
    ai_model = Column(String, nullable=True)
    ai_scored_at = Column(DateTime, nullable=True)

    # --- Outcome ---
    outcome = Column(String, nullable=True)             # "purchased", "lost_bidding", "withdrew", "expired"
    outcome_notes = Column(Text, nullable=True)
    outcome_date = Column(DateTime, nullable=True)

    # --- Status & Tracking ---
    listing_status = Column(String, default="new")
    # Valid: new, insufficient_data, needs_info, scored, under_evaluation,
    #        viewing_scheduled, offer_made, decided, archived
    first_seen = Column(DateTime, nullable=True)
    days_on_market = Column(Integer, nullable=True)
    price_history = Column(JSON, nullable=True)

    # --- Agent ---
    agent_name = Column(String, nullable=True)
    agent_email = Column(String, nullable=True)
    agent_phone = Column(String, nullable=True)
    agent_company = Column(String, nullable=True)
    last_contacted_at = Column(DateTime, nullable=True)
    contact_method = Column(String, nullable=True)

    # --- Geolocation ---
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    geocoded = Column(Boolean, default=False)

    # --- Photos ---
    photo_urls = Column(JSON, nullable=True)
    photos_archived = Column(Boolean, default=False)

    # --- Google Drive ---
    gdrive_folder_id = Column(String, nullable=True)
    gdrive_folder_url = Column(String, nullable=True)

    # --- Timestamps ---
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    archived_at = Column(DateTime, nullable=True)

    # --- Relationships ---
    email = relationship("Email", foreign_keys=[email_id])
    source_emails = relationship("PropertyListingSource", back_populates="listing", cascade="all, delete-orphan")
    reviews = relationship("PropertyReview", back_populates="listing", cascade="all, delete-orphan")
    actions = relationship("PropertyAction", back_populates="listing", cascade="all, delete-orphan", order_by="PropertyAction.acted_at")
    private_notes = relationship("PropertyPrivateNote", back_populates="listing", cascade="all, delete-orphan")
    due_diligence = relationship("PropertyDueDiligence", back_populates="listing", cascade="all, delete-orphan")
    documents = relationship("PropertyDocument", back_populates="listing", cascade="all, delete-orphan")
    collection_items = relationship("PropertyCollectionItem", back_populates="listing", cascade="all, delete-orphan")
    share_tokens = relationship("PropertyShareToken", back_populates="listing", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_property_listings_plz", "plz"),
        Index("ix_property_listings_status", "listing_status"),
        Index("ix_property_listings_recommendation", "overall_recommendation"),
        Index("ix_property_listings_price", "price_chf"),
        Index("ix_property_listings_type", "property_type"),
        Index("ix_property_listings_dedup", "dedup_hash"),
        Index("ix_property_listings_source", "listing_source"),
        Index("ix_property_listings_tier", "tier"),
    )


# =============================================================================
# DEDUP: Multiple notification emails → one property
# =============================================================================

class PropertyListingSource(Base):
    __tablename__ = "property_listing_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    listing_id = Column(UUID(as_uuid=True), ForeignKey("property_listings.id"), nullable=False)
    email_id = Column(UUID(as_uuid=True), ForeignKey("emails.id"), nullable=True)
    source = Column(String, nullable=False)
    listing_url = Column(String, nullable=True)
    price_at_source = Column(Integer, nullable=True)
    first_seen_at = Column(DateTime, default=func.now())

    listing = relationship("PropertyListing", back_populates="source_emails")

    __table_args__ = (
        Index("ix_property_listing_sources_email", "email_id"),
        Index("ix_property_listing_sources_listing", "listing_id"),
    )


# =============================================================================
# REVIEWS: Family member ratings (1-5)
# =============================================================================

class PropertyReview(Base):
    __tablename__ = "property_reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    listing_id = Column(UUID(as_uuid=True), ForeignKey("property_listings.id"), nullable=False)
    family_member_id = Column(UUID(as_uuid=True), ForeignKey("lab_members.id"), nullable=False)
    rating = Column(Integer, nullable=False)            # 1-5
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, nullable=True)

    listing = relationship("PropertyListing", back_populates="reviews")
    reviewer = relationship("LabMember")

    __table_args__ = (
        UniqueConstraint("listing_id", "family_member_id", name="uq_property_review_member"),
    )


# =============================================================================
# ACTIONS: Append-only action log (workflow history)
# =============================================================================

class PropertyAction(Base):
    __tablename__ = "property_actions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    listing_id = Column(UUID(as_uuid=True), ForeignKey("property_listings.id"), nullable=False)
    acted_by = Column(UUID(as_uuid=True), ForeignKey("lab_members.id"), nullable=False)
    action = Column(String, nullable=False)
    # Valid: interested, request_info, request_documentation, request_viewing,
    #        request_bank_eval, make_offer, not_interested, archive
    notes = Column(Text, nullable=True)
    # Status the listing transitioned to as a result of this action
    resulting_status = Column(String, nullable=True)
    acted_at = Column(DateTime, default=func.now())

    listing = relationship("PropertyListing", back_populates="actions")
    actor = relationship("LabMember")

    __table_args__ = (
        Index("ix_property_actions_listing", "listing_id"),
        Index("ix_property_actions_acted_at", "acted_at"),
    )


# =============================================================================
# PRIVATE NOTES: Per-member notes (not visible to others)
# =============================================================================

class PropertyPrivateNote(Base):
    __tablename__ = "property_private_notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    listing_id = Column(UUID(as_uuid=True), ForeignKey("property_listings.id"), nullable=False)
    family_member_id = Column(UUID(as_uuid=True), ForeignKey("lab_members.id"), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, nullable=True)

    listing = relationship("PropertyListing", back_populates="private_notes")

    __table_args__ = (
        UniqueConstraint("listing_id", "family_member_id", name="uq_property_note_member"),
    )


# =============================================================================
# DUE DILIGENCE: Tier 3 checklist tracking
# =============================================================================

class PropertyDueDiligence(Base):
    __tablename__ = "property_due_diligence"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    listing_id = Column(UUID(as_uuid=True), ForeignKey("property_listings.id"), nullable=False)
    category = Column(String, nullable=False)           # "neighbor_development", "grundbuch", etc.
    question = Column(String, nullable=False)
    priority = Column(String, nullable=True)            # "critical", "high", "normal"
    status = Column(String, default="open")             # open, answered, not_applicable, deal_breaker
    answer = Column(Text, nullable=True)
    source = Column(String, nullable=True)              # "agent", "document", "research", "viewing"
    date_answered = Column(DateTime, nullable=True)
    is_deal_breaker = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())

    listing = relationship("PropertyListing", back_populates="due_diligence")

    __table_args__ = (
        Index("ix_property_due_diligence_listing", "listing_id"),
    )


# =============================================================================
# DOCUMENTS: Per-property document tracking
# =============================================================================

class PropertyDocument(Base):
    __tablename__ = "property_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    listing_id = Column(UUID(as_uuid=True), ForeignKey("property_listings.id"), nullable=False)
    document_type = Column(String, nullable=False)      # "verkaufsbrochure", "grundbuch", etc.
    filename = Column(String, nullable=False)
    gdrive_file_id = Column(String, nullable=True)
    gdrive_link = Column(String, nullable=True)
    local_path = Column(String, nullable=True)
    source = Column(String, nullable=True)              # "agent", "portal", "user", "houzy"
    uploaded_at = Column(DateTime, default=func.now())

    listing = relationship("PropertyListing", back_populates="documents")

    __table_args__ = (
        Index("ix_property_documents_listing", "listing_id"),
    )


# =============================================================================
# COLLECTIONS: Organize listings into groups
# =============================================================================

class PropertyCollection(Base):
    __tablename__ = "property_collections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("lab_members.id"), nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, nullable=True)

    items = relationship("PropertyCollectionItem", back_populates="collection", cascade="all, delete-orphan")
    creator = relationship("LabMember")


class PropertyCollectionItem(Base):
    __tablename__ = "property_collection_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey("property_collections.id"), nullable=False)
    listing_id = Column(UUID(as_uuid=True), ForeignKey("property_listings.id"), nullable=False)
    added_at = Column(DateTime, default=func.now())
    notes = Column(Text, nullable=True)

    collection = relationship("PropertyCollection", back_populates="items")
    listing = relationship("PropertyListing", back_populates="collection_items")

    __table_args__ = (
        UniqueConstraint("collection_id", "listing_id", name="uq_collection_listing"),
    )


# =============================================================================
# SHARE TOKENS: Shareable links for co-purchase partners (Scenario C)
# =============================================================================

class PropertyShareToken(Base):
    __tablename__ = "property_share_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    listing_id = Column(UUID(as_uuid=True), ForeignKey("property_listings.id"), nullable=False)
    token_hash = Column(String, unique=True, nullable=False)  # SHA256 of JWT token
    created_by = Column(UUID(as_uuid=True), ForeignKey("lab_members.id"), nullable=False)
    permissions = Column(JSON, nullable=True)         # {can_view_reviews, can_view_actions, can_view_documents}
    expires_at = Column(DateTime, nullable=True)
    max_uses = Column(Integer, nullable=True)
    uses_count = Column(Integer, default=0)
    is_revoked = Column(Boolean, default=False)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

    listing = relationship("PropertyListing", back_populates="share_tokens")
    creator = relationship("LabMember")

    __table_args__ = (
        Index("ix_property_share_tokens_token", "token_hash"),
        Index("ix_property_share_tokens_listing", "listing_id"),
    )
