"""
Unit tests for authentication module
"""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import patch, Mock

from backend.api.review_auth import (
    create_jwt_token,
    decode_jwt_token,
    generate_oauth_state,
    verify_oauth_state,
    blacklist_token,
    handle_failed_login
)
from backend.core.database.models import LabMember, JWTBlacklist


def test_create_jwt_token():
    """Test JWT token creation."""
    token = create_jwt_token(
        user_id="test_user_id",
        email="test@example.com",
        role="member"
    )
    
    assert token is not None
    assert isinstance(token, str)
    
    # Decode and verify
    payload = decode_jwt_token(token)
    assert payload["sub"] == "test_user_id"
    assert payload["email"] == "test@example.com"
    assert payload["role"] == "member"
    assert "exp" in payload
    assert "jti" in payload


def test_decode_jwt_token_expired():
    """Test decoding expired token."""
    import os
    import jwt
    
    # Create expired token manually
    secret = os.getenv("JWT_SECRET", "test_secret")
    if not secret or secret == "test_secret":
        # Use a test secret if JWT_SECRET not set
        secret = "test_secret_for_expired_token_test"
    
    payload = {
        "sub": "test_user",
        "exp": datetime.utcnow() - timedelta(hours=1)  # Expired
    }
    expired_token = jwt.encode(payload, secret, algorithm="HS256")
    
    # Should raise HTTPException (wrapped as Exception in pytest)
    with pytest.raises(Exception):  # HTTPException from decode_jwt_token
        decode_jwt_token(expired_token)


def test_oauth_state_generation():
    """Test OAuth state generation."""
    state1 = generate_oauth_state()
    state2 = generate_oauth_state()
    
    assert state1 != state2
    assert len(state1) > 20  # Should be reasonably long


def test_oauth_state_verification():
    """Test OAuth state verification."""
    state = generate_oauth_state()
    
    # Valid state (can be verified multiple times until expiration)
    assert verify_oauth_state(state) is True
    assert verify_oauth_state(state) is True  # JWT-based state can be verified multiple times
    
    # Invalid state
    assert verify_oauth_state("invalid_state") is False
    assert verify_oauth_state("not.a.valid.jwt.token") is False


def test_blacklist_token(test_db, admin_user):
    """Test token blacklisting."""
    token = create_jwt_token(
        user_id=str(admin_user.id),
        email=admin_user.email,
        role=admin_user.role
    )
    
    # Blacklist token
    blacklist_token(token, str(admin_user.id), test_db)
    
    # Verify blacklisted
    payload = decode_jwt_token(token)
    jti = payload.get("jti")
    
    blacklisted = test_db.query(JWTBlacklist).filter(
        JWTBlacklist.token_jti == jti
    ).first()
    
    assert blacklisted is not None
    assert blacklisted.user_id == admin_user.id


def test_handle_failed_login(test_db, reviewer_user):
    """Test failed login handling."""
    initial_attempts = reviewer_user.failed_login_attempts or 0
    
    # First 4 failures
    for i in range(4):
        handle_failed_login(reviewer_user.email, test_db)
        test_db.refresh(reviewer_user)
        assert reviewer_user.failed_login_attempts == initial_attempts + i + 1
        assert reviewer_user.locked_until is None
    
    # 5th failure should lock account
    handle_failed_login(reviewer_user.email, test_db)
    test_db.refresh(reviewer_user)
    assert reviewer_user.failed_login_attempts == initial_attempts + 5
    assert reviewer_user.locked_until is not None
    assert reviewer_user.locked_until > datetime.utcnow()

