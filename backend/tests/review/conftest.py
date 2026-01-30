"""
Test fixtures for Lab Application Review System
"""
import pytest
import os
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.main import app
from backend.core.database import get_db
from backend.core.database.models import (
    LabMember, Email, EmailMetadata, ApplicationReview, 
    ApplicationDecision, SystemSettings, JWTBlacklist, Base
)
from backend.api.review_auth import create_jwt_token


# Test database setup - use unique in-memory database per test
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture(scope="function")
def test_db():
    """Create test database and tables."""
    # Create a new engine for each test to ensure isolation
    if "sqlite" in TEST_DATABASE_URL:
        # Use unique in-memory database for each test
        import uuid
        unique_db_url = f"sqlite:///:memory:{uuid.uuid4().hex}"
        engine = create_engine(
            unique_db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(TEST_DATABASE_URL)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Clean up
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(scope="function")
def client(test_db):
    """Create test client with database override."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def admin_user(test_db):
    """Create admin user for testing."""
    user = LabMember(
        id=uuid4(),
        email="admin@test.com",
        full_name="Admin User",
        role="admin",
        can_review=True,
        is_active=True,
        gsuite_id="admin_gsuite_id"
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def reviewer_user(test_db):
    """Create reviewer user for testing."""
    user = LabMember(
        id=uuid4(),
        email="reviewer@test.com",
        full_name="Reviewer User",
        role="member",
        can_review=True,
        is_active=True,
        gsuite_id="reviewer_gsuite_id"
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def regular_user(test_db):
    """Create regular user (no review permission) for testing."""
    user = LabMember(
        id=uuid4(),
        email="user@test.com",
        full_name="Regular User",
        role="member",
        can_review=False,
        is_active=True,
        gsuite_id="user_gsuite_id"
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user):
    """Create JWT token for admin user."""
    return create_jwt_token(
        user_id=str(admin_user.id),
        email=admin_user.email,
        role=admin_user.role
    )


@pytest.fixture
def reviewer_token(reviewer_user):
    """Create JWT token for reviewer user."""
    return create_jwt_token(
        user_id=str(reviewer_user.id),
        email=reviewer_user.email,
        role=reviewer_user.role
    )


@pytest.fixture
def regular_token(regular_user):
    """Create JWT token for regular user."""
    return create_jwt_token(
        user_id=str(regular_user.id),
        email=regular_user.email,
        role=regular_user.role
    )


@pytest.fixture
def sample_application(test_db):
    """Create sample application for testing."""
    email = Email(
        id=uuid4(),
        message_id="<test-app@example.com>",
        uid="12345",  # Required field
        subject="PhD Application",
        from_name="John Doe",
        from_address="john.doe@university.edu",
        to_addresses=["lab@university.edu"],  # Required field
        date=datetime.utcnow() - timedelta(days=5),
        body_text="I am applying for a PhD position...",
        body_html="<p>I am applying for a PhD position...</p>"
    )
    test_db.add(email)
    test_db.flush()
    
    metadata = EmailMetadata(
        email_id=email.id,
        ai_category="application-phd",
        applicant_name="John Doe",
        applicant_institution="University of Example",
        research_fit_score=7,
        overall_recommendation_score=8,
        category_metadata={"scientific_excellence_score": 8},  # Stored in JSON
        application_status="pending",
        review_deadline=datetime.utcnow() + timedelta(days=2)
    )
    test_db.add(metadata)
    test_db.commit()
    test_db.refresh(email)
    test_db.refresh(metadata)
    
    return email, metadata


@pytest.fixture
def multiple_applications(test_db):
    """Create multiple sample applications."""
    applications = []
    for i in range(5):
        email = Email(
            id=uuid4(),
            message_id=f"<test-app-{i}@example.com>",
            uid=f"uid_{i}",  # Required field
            subject=f"Application {i}",
            from_name=f"Applicant {i}",
            from_address=f"applicant{i}@university.edu",
            to_addresses=["lab@university.edu"],  # Required field
            date=datetime.utcnow() - timedelta(days=i),
            body_text=f"Application {i} body",
            body_html=f"<p>Application {i} body</p>"
        )
        test_db.add(email)
        test_db.flush()
        
        metadata = EmailMetadata(
            email_id=email.id,
            ai_category="application-phd",
            applicant_name=f"Applicant {i}",
            applicant_institution=f"University {i}",
            research_fit_score=5 + i,
            overall_recommendation_score=5 + i,
            category_metadata={"scientific_excellence_score": 5 + i},  # Stored in JSON
            application_status="pending"
        )
        test_db.add(metadata)
        applications.append((email, metadata))
    
    test_db.commit()
    return applications


@pytest.fixture
def sample_review(test_db, sample_application, reviewer_user):
    """Create sample review."""
    email, metadata = sample_application
    review = ApplicationReview(
        email_id=email.id,
        lab_member_id=reviewer_user.id,
        rating=4,
        comment="Good candidate"
    )
    test_db.add(review)
    test_db.commit()
    test_db.refresh(review)
    return review


@pytest.fixture
def sample_decision(test_db, sample_application, admin_user):
    """Create sample decision."""
    email, metadata = sample_application
    decision = ApplicationDecision(
        email_id=email.id,
        admin_id=admin_user.id,
        decision="accept",
        notes="Accepted for interview"
    )
    test_db.add(decision)
    test_db.commit()
    test_db.refresh(decision)
    return decision


@pytest.fixture
def system_settings(test_db):
    """Create system settings for testing."""
    settings = [
        SystemSettings(
            key="rate_limit_reviews_per_hour",
            value={"value": 50, "type": "integer"},
            description="Maximum reviews per hour per user"
        ),
        SystemSettings(
            key="rate_limit_requests_per_minute",
            value={"value": 100, "type": "integer"},
            description="Maximum requests per minute per user"
        ),
        SystemSettings(
            key="gdrive_archive_root_id",
            value={"value": "test_archive_root_id", "type": "string"},
            description="Google Drive archive root folder ID"
        ),
    ]
    for setting in settings:
        test_db.add(setting)
    test_db.commit()
    return settings


@pytest.fixture
def mock_google_oauth():
    """Mock Google OAuth responses."""
    with patch('backend.api.routes.review_auth.verify_google_token') as mock:
        mock.return_value = {
            "sub": "test_gsuite_id",
            "email": "test@example.com",
            "name": "Test User",
            "hd": "example.com"
        }
        yield mock


@pytest.fixture
def mock_gdrive_client():
    """Mock Google Drive client."""
    with patch('backend.core.google.drive_client.GoogleDriveClient') as mock_client_class:
        mock_client = Mock()
        mock_client.get_or_create_folder_structure.return_value = "test_folder_id"
        mock_client.drive_service.files().get().execute.return_value = {"parents": ["parent_id"]}
        mock_client.drive_service.files().update().execute.return_value = {"id": "test_folder_id"}
        mock_client_class.return_value = mock_client
        yield mock_client

