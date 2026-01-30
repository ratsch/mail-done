"""
Authentication routes for Lab Application Review System

Google OAuth2 flow and JWT token management
"""
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from starlette.requests import Request as StarletteRequest
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.responses import Response as StarletteResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import httpx
from urllib.parse import urlencode

from backend.core.database import get_db
from backend.core.database.models import LabMember
from backend.api.review_auth import (
    generate_oauth_state,
    verify_oauth_state,
    create_jwt_token,
    decode_jwt_token,
    verify_google_token,
    get_current_user,
    get_current_user_hybrid,
    blacklist_token,
    handle_failed_auth
)
import os
import logging

# Get JWT expiration hours from environment (same as review_auth.py)
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["review-auth"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID_V0_PORTAL")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET_V0_PORTAL")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI_V0_PORTAL")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3001")


class RefreshTokenRequest(BaseModel):
    token: str


class LogoutRequest(BaseModel):
    token: str


@router.get("/google/init")
async def google_oauth_init():
    """
    Initiate Google OAuth2 flow.
    
    Generates state parameter for CSRF protection and redirects to Google.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_REDIRECT_URI:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth not configured"
        )
    
    # Generate state for CSRF protection
    state = generate_oauth_state()
    
    # Build Google OAuth URL
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={GOOGLE_REDIRECT_URI}&"
        "response_type=code&"
        "scope=openid email profile&"
        f"state={state}&"
        "access_type=offline&"
        "prompt=consent"
    )
    
    return RedirectResponse(url=google_auth_url)


@router.get("/google/callback")
async def google_oauth_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Handle Google OAuth2 callback.
    
    Validates state, exchanges code for token, verifies user, creates/updates lab member,
    and returns JWT token.
    """
    if error:
        logger.error(f"OAuth callback error: {error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth error: {error}"
        )
    
    if not code or not state:
        logger.error(f"Missing OAuth parameters - code: {bool(code)}, state: {bool(state)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing code or state parameter"
        )
    
    # URL decode state if needed (FastAPI should do this automatically, but let's be safe)
    from urllib.parse import unquote
    try:
        # FastAPI automatically URL-decodes query parameters, but verify
        decoded_state = unquote(state) if '%' in state else state
        logger.info(f"Verifying OAuth state (length: {len(decoded_state)}, starts with: {decoded_state[:20]}...)")
    except Exception as e:
        logger.error(f"Failed to decode state parameter: {e}")
        decoded_state = state
    
    # Verify state (CSRF protection)
    if not verify_oauth_state(decoded_state):
        logger.error(f"OAuth state verification failed. State length: {len(decoded_state)}, starts with: {decoded_state[:20]}...")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state parameter"
        )
    logger.info("OAuth state verified successfully")
    
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth not configured"
        )
    
    # Exchange authorization code for tokens
    try:
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code"
                }
            )
            token_response.raise_for_status()
            tokens = token_response.json()
            id_token_str = tokens.get("id_token")
            
            if not id_token_str:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No ID token in response"
                )
    except httpx.HTTPStatusError as e:
        logger.error(f"Token exchange failed: {e.response.text}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to exchange authorization code"
        )
    
    # Verify Google ID token
    google_user = None
    try:
        google_user = await verify_google_token(id_token_str)
    except HTTPException:
        # Token verification failed - we can't track by email since we don't have it yet
        raise
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to verify Google token"
        )
    
    gsuite_id = google_user["sub"]
    email = google_user["email"]
    name = google_user.get("name", "")
    avatar_url = google_user.get("picture")  # Google OAuth provides 'picture' field
    
    # Find or create lab member
    # First check by gsuite_id, then by email (in case gsuite_id changed or wasn't set)
    lab_member = db.query(LabMember).filter(
        LabMember.gsuite_id == gsuite_id
    ).first()
    
    if not lab_member:
        # Check if user exists by email (might have been created manually or with different gsuite_id)
        lab_member = db.query(LabMember).filter(
            LabMember.email == email
        ).first()
        
        if lab_member:
            # Update existing user with gsuite_id and other info
            lab_member.gsuite_id = gsuite_id
            lab_member.full_name = name
            lab_member.avatar_url = avatar_url
            lab_member.last_login_at = datetime.utcnow()
            db.commit()
            db.refresh(lab_member)
            logger.info(f"Updated existing lab member with gsuite_id: {email}")
        else:
            # Create new lab member (default: can_review=False, needs admin approval)
            lab_member = LabMember(
                gsuite_id=gsuite_id,
                email=email,
                full_name=name,
                role="member",
                can_review=False,  # Default: no access until admin grants permission
                is_active=True,
                avatar_url=avatar_url  # Store Google profile picture
            )
            db.add(lab_member)
            db.commit()
            db.refresh(lab_member)
            logger.info(f"Created new lab member: {email}")
    else:
        # Update last login time and avatar if changed
        lab_member.last_login_at = datetime.utcnow()
        if avatar_url and lab_member.avatar_url != avatar_url:
            lab_member.avatar_url = avatar_url
        if name and lab_member.full_name != name:
            lab_member.full_name = name
        # Update gsuite_id if it changed (shouldn't happen, but handle it)
        if lab_member.gsuite_id != gsuite_id:
            lab_member.gsuite_id = gsuite_id
        db.commit()
        db.refresh(lab_member)
        
        # Check if account is locked
        if lab_member.locked_until and lab_member.locked_until > datetime.utcnow():
            # Track this login attempt while locked (potential attacker probing)
            handle_failed_auth(email, db, reason="account_locked")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account is locked until {lab_member.locked_until.isoformat()}"
            )
        
        # Check if account is suspended
        if not lab_member.is_active:
            # Track login attempts to suspended accounts
            handle_failed_auth(email, db, reason="account_suspended")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is suspended. Contact an admin."
            )
        
        # Update last login, reset failed attempts, and update avatar if available
        lab_member.last_login_at = datetime.utcnow()
        lab_member.failed_login_attempts = 0
        lab_member.locked_until = None  # Clear lockout on successful login
        if avatar_url:  # Update avatar URL if provided
            lab_member.avatar_url = avatar_url
        db.commit()
    
    # Create JWT token
    jwt_token = create_jwt_token(
        user_id=str(lab_member.id),
        email=lab_member.email,
        role=lab_member.role
    )
    
    # Redirect to frontend callback with token
    # Frontend will handle setting the cookie and redirecting to dashboard
    if not FRONTEND_URL:
        logger.error("FRONTEND_URL not configured, cannot redirect")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Frontend URL not configured"
        )
    
    # Ensure FRONTEND_URL doesn't have trailing slash
    frontend_url_clean = FRONTEND_URL.rstrip('/')
    frontend_callback_url = f"{frontend_url_clean}/api/auth/callback"
    redirect_params = {
        "token": jwt_token,
        "user_id": str(lab_member.id),
        "email": lab_member.email,
        "role": lab_member.role,
        "can_review": str(lab_member.can_review).lower(),
        "is_active": str(lab_member.is_active).lower()
    }
    
    redirect_url = f"{frontend_callback_url}?{urlencode(redirect_params)}"
    url_length = len(redirect_url)
    
    logger.info(f"FRONTEND_URL from env: {FRONTEND_URL}")
    logger.info(f"Redirecting to frontend: {frontend_callback_url} (URL length: {url_length})")
    logger.info(f"Full redirect URL (first 200 chars): {redirect_url[:200]}")
    
    # Warn if URL is very long (some browsers have ~2000 char limit)
    if url_length > 1800:
        logger.warning(f"Redirect URL is very long ({url_length} chars), may cause issues in some browsers")
    
    # Validate URL format
    try:
        from urllib.parse import urlparse
        parsed = urlparse(redirect_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
    except Exception as e:
        logger.error(f"Invalid redirect URL format: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid frontend URL configuration: {str(e)}"
        )
    
    # Use FastAPI's RedirectResponse which handles redirects properly
    try:
        return RedirectResponse(url=redirect_url, status_code=307)
    except ValueError as e:
        # Invalid URL format
        logger.error(f"Invalid redirect URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid redirect URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to create redirect response: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to redirect to frontend. Please contact support."
        )


@router.post("/refresh")
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Refresh JWT token before expiry.
    
    Requires valid but expiring token. Returns new token and blacklists old one.
    """
    try:
        payload = decode_jwt_token(request.token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get user
        user = db.query(LabMember).filter(LabMember.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Blacklist old token
        blacklist_token(request.token, user_id, db)
        
        # Create new token
        new_token = create_jwt_token(
            user_id=str(user.id),
            email=user.email,
            role=user.role
        )
        
        return {
            "access_token": new_token,
            "token_type": "bearer"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to refresh token"
        )


@router.post("/logout")
async def logout(
    request: LogoutRequest,
    current_user: LabMember = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout user by blacklisting JWT token.
    """
    blacklist_token(request.token, str(current_user.id), db)
    
    return {
        "message": "Logged out successfully"
    }


@router.get("/me")
async def get_current_user_info(
    current_user: LabMember = Depends(get_current_user_hybrid)
):
    """
    Get current authenticated user's information.
    """
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "can_review": current_user.can_review,
        "is_active": current_user.is_active,
        "avatar_url": current_user.avatar_url,
        "last_login_at": current_user.last_login_at.isoformat() if current_user.last_login_at else None
    }

