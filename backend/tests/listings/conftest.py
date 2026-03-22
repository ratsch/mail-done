"""
Test fixtures for Property Listing Review System

Follows patterns from backend/tests/review/conftest.py.
Uses SQLite in-memory DB with UUID workaround for PostgreSQL UUID columns.
"""
import os
import pytest
import uuid as _uuid

# Set CONFIG_DIR to project config/ before any app imports
os.environ["CONFIG_DIR"] = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config"
)
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.main import app
from backend.core.database import get_db
from backend.core.database.models import LabMember, Base
from backend.core.database.models_property import (
    PropertyListing, PropertyListingSource, PropertyReview, PropertyAction,
    PropertyPrivateNote, PropertyDueDiligence, PropertyDocument,
    PropertyCollection, PropertyCollectionItem, PropertyShareToken,
)
from backend.api.review_auth import create_jwt_token


# Tables that use PostgreSQL-only features (ARRAY type) — exclude from SQLite
_PG_ONLY_TABLES = {"documents", "document_origins", "document_pages", "document_processing_queue"}


@pytest.fixture(scope="function")
def test_db():
    """Create isolated in-memory SQLite test database."""
    unique_url = f"sqlite:///:memory:{_uuid.uuid4().hex}"
    engine = create_engine(
        unique_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # SQLite doesn't have native UUID type — store as CHAR(32)
    # The PostgreSQL UUID columns auto-adapt via SQLAlchemy's type system

    tables_to_create = [
        t for t in Base.metadata.sorted_tables
        if t.name not in _PG_ONLY_TABLES
    ]
    Base.metadata.create_all(bind=engine, tables=tables_to_create)

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(scope="function")
def client(test_db):
    """Test client with DB override. Disables security monitor to prevent
    accumulated error counts from blocking subsequent tests."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Reset rate limiter and security monitor state to prevent cross-test contamination
    try:
        from backend.api.rate_limiting import rate_limiter
        with rate_limiter.lock:
            rate_limiter.ip_buckets.clear()
            rate_limiter.key_buckets.clear()
            rate_limiter.shared_buckets.clear()
            rate_limiter.failed_auth_attempts.clear()
    except (ImportError, AttributeError):
        pass
    try:
        from backend.api.security_monitor import _monitor
        if _monitor:
            _monitor.error_counts.clear()
            _monitor.locked_until = None
    except (ImportError, AttributeError):
        pass

    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# User fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def admin_user(test_db):
    user = LabMember(
        id=uuid4(), email="gunnar@test.com", full_name="Gunnar Raetsch",
        role="admin", can_review=True, is_active=True, gsuite_id="gunnar_gid",
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def reviewer_user(test_db):
    user = LabMember(
        id=uuid4(), email="nora@test.com", full_name="Nora Toussaint",
        role="member", can_review=True, is_active=True, gsuite_id="nora_gid",
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def regular_user(test_db):
    user = LabMember(
        id=uuid4(), email="visitor@test.com", full_name="Visitor",
        role="member", can_review=False, is_active=True, gsuite_id="visitor_gid",
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user):
    return create_jwt_token(user_id=str(admin_user.id), email=admin_user.email, role=admin_user.role)


@pytest.fixture
def reviewer_token(reviewer_user):
    return create_jwt_token(user_id=str(reviewer_user.id), email=reviewer_user.email, role=reviewer_user.role)


@pytest.fixture
def regular_token(regular_user):
    return create_jwt_token(user_id=str(regular_user.id), email=regular_user.email, role=regular_user.role)


# ---------------------------------------------------------------------------
# Listing fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_listing(test_db):
    """Single property listing with basic data."""
    listing = PropertyListing(
        id=uuid4(),
        address="Talgasse 9, 7250 Klosters",
        plz="7250",
        municipality="Klosters",
        canton="GR",
        listing_url="https://homegate.ch/buy/123",
        listing_source="homegate",
        property_type="apartment",
        price_chf=1200000,
        price_known=True,
        living_area_sqm=120,
        rooms=4.5,
        year_built=1985,
        price_per_sqm=10000,
        listing_status="scored",
        tier=2,
        overall_recommendation=7,
        macro_location_score=8,
        micro_location_score=6,
        property_quality_score=7,
        garden_outdoor_score=5,
        financial_score=7,
        applicable_scenarios=["A"],
        best_scenario="A",
        houzy_mid=1100000,
        houzy_assessment="undervalued",
        price_vs_houzy_pct=9.1,
        highlights=["Mountain view", "Quiet location"],
        red_flags=["Needs renovation"],
        property_tags=["mountain_view", "quiet"],
        completeness_pct=85,
        description="Beautiful apartment in Klosters with mountain views.",
        latitude=46.884,
        longitude=9.878,
        geocoded=True,
        first_seen=datetime.utcnow() - timedelta(days=10),
        days_on_market=10,
        created_at=datetime.utcnow() - timedelta(days=10),
    )
    test_db.add(listing)
    test_db.commit()
    test_db.refresh(listing)
    return listing


@pytest.fixture
def multiple_listings(test_db):
    """Five listings with varied data for filter/sort testing."""
    listings = []
    configs = [
        {"plz": "7250", "municipality": "Klosters", "price": 1200000, "rooms": 4.5, "score": 8, "scenario": "A", "status": "scored", "source": "homegate", "tier": 2},
        {"plz": "8044", "municipality": "Zürich", "price": 2500000, "rooms": 5.5, "score": 6, "scenario": "B", "status": "new", "source": "engelvoelkers", "tier": 1},
        {"plz": "8006", "municipality": "Zürich", "price": 800000, "rooms": 3.5, "score": 9, "scenario": "C", "status": "viewing_scheduled", "source": "homegate", "tier": 2},
        {"plz": "8032", "municipality": "Zürich", "price": 1500000, "rooms": 4.0, "score": 5, "scenario": "D", "status": "archived", "source": "immoscout", "tier": 2},
        {"plz": "7250", "municipality": "Klosters", "price": None, "rooms": 6.0, "score": None, "scenario": "A", "status": "insufficient_data", "source": "homegate", "tier": 1},
    ]
    for i, cfg in enumerate(configs):
        listing = PropertyListing(
            id=uuid4(),
            address=f"Teststrasse {i+1}, {cfg['plz']} {cfg['municipality']}",
            plz=cfg["plz"],
            municipality=cfg["municipality"],
            listing_source=cfg["source"],
            property_type="apartment",
            price_chf=cfg["price"],
            price_known=cfg["price"] is not None,
            living_area_sqm=100 + i * 20,
            rooms=cfg["rooms"],
            overall_recommendation=cfg["score"],
            best_scenario=cfg["scenario"],
            applicable_scenarios=[cfg["scenario"]],
            listing_status=cfg["status"],
            tier=cfg["tier"],
            property_tags=["quiet"] if i % 2 == 0 else ["urban"],
            latitude=46.88 + i * 0.01 if i < 4 else None,
            longitude=9.87 + i * 0.01 if i < 4 else None,
            geocoded=i < 4,
            created_at=datetime.utcnow() - timedelta(days=i),
        )
        test_db.add(listing)
        listings.append(listing)
    test_db.commit()
    for l in listings:
        test_db.refresh(l)
    return listings


@pytest.fixture
def listing_with_review(test_db, sample_listing, reviewer_user):
    """Listing that has a review from reviewer_user."""
    review = PropertyReview(
        listing_id=sample_listing.id,
        family_member_id=reviewer_user.id,
        rating=4,
        comment="Nice property, good location",
    )
    test_db.add(review)
    test_db.commit()
    test_db.refresh(review)
    return review


@pytest.fixture
def listing_with_action(test_db, sample_listing, admin_user):
    """Listing that has an action recorded."""
    action = PropertyAction(
        listing_id=sample_listing.id,
        acted_by=admin_user.id,
        action="interested",
        notes="Looks promising",
        resulting_status="under_evaluation",
    )
    test_db.add(action)
    sample_listing.listing_status = "under_evaluation"
    test_db.commit()
    test_db.refresh(action)
    return action


@pytest.fixture
def sample_collection(test_db, admin_user):
    """A property collection."""
    coll = PropertyCollection(
        id=uuid4(),
        name="Klosters Shortlist",
        description="Best properties in Klosters",
        created_by=admin_user.id,
    )
    test_db.add(coll)
    test_db.commit()
    test_db.refresh(coll)
    return coll


@pytest.fixture
def collection_with_item(test_db, sample_collection, sample_listing):
    """Collection with one listing in it."""
    item = PropertyCollectionItem(
        collection_id=sample_collection.id,
        listing_id=sample_listing.id,
    )
    test_db.add(item)
    test_db.commit()
    test_db.refresh(item)
    return item


@pytest.fixture
def sample_share_token(test_db, sample_listing, admin_user):
    """A share token for a listing."""
    import hashlib, secrets
    raw_token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    st = PropertyShareToken(
        listing_id=sample_listing.id,
        token_hash=token_hash,
        created_by=admin_user.id,
        permissions={"can_view_reviews": True, "can_view_actions": True, "can_view_documents": False},
        expires_at=datetime.utcnow() + timedelta(hours=168),
        max_uses=100,
        uses_count=0,
        is_revoked=False,
    )
    test_db.add(st)
    test_db.commit()
    test_db.refresh(st)
    return st, raw_token
