"""
Authentication for Lab Application Review System

Google OAuth2 (GSuite) + JWT token authentication
"""
import os
import secrets
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from google.oauth2 import id_token
from google.auth.transport import requests
import httpx

from backend.core.database import get_db
from backend.core.database.models import LabMember, JWTBlacklist
from backend.core.config import get_settings

settings = get_settings()

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
JWT_ISSUER = os.getenv("JWT_ISSUER", "mail-done-review-system")  # Token issuer identifier
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "mail-done-review-api")  # Token audience identifier

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID_V0_PORTAL")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET_V0_PORTAL")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI_V0_PORTAL")

# OAuth state storage - use JWT-based state for multi-instance support
# JWT state is self-contained and doesn't require server-side storage
OAUTH_STATE_SECRET = os.getenv("OAUTH_STATE_SECRET", JWT_SECRET)  # Can use same secret or separate
OAUTH_STATE_EXPIRATION_MINUTES = 10

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


def generate_oauth_state() -> str:
    """
    Generate a JWT-based state parameter for CSRF protection.
    
    Uses JWT so state is self-contained and works across multiple instances.
    State expires after 10 minutes.
    """
    payload = {
        "type": "oauth_state",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=OAUTH_STATE_EXPIRATION_MINUTES),
        "nonce": secrets.token_urlsafe(16)  # Additional randomness
    }
    state = jwt.encode(payload, OAUTH_STATE_SECRET, algorithm=JWT_ALGORITHM)
    return state


def verify_oauth_state(state: str) -> bool:
    """
    Verify OAuth state parameter using JWT verification.
    
    Returns True if state is valid and not expired, False otherwise.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        payload = jwt.decode(state, OAUTH_STATE_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Verify it's an OAuth state token
        if payload.get("type") != "oauth_state":
            logger.warning(f"OAuth state verification failed: wrong token type. Got: {payload.get('type')}")
            return False
        
        # JWT decode already checks expiration, so if we get here, it's valid
        logger.debug(f"OAuth state verified successfully")
        return True
    except jwt.ExpiredSignatureError:
        # State expired
        logger.warning(f"OAuth state verification failed: expired")
        return False
    except jwt.InvalidTokenError as e:
        # Invalid state token
        logger.warning(f"OAuth state verification failed: invalid token. Error: {str(e)}")
        return False
    except Exception as e:
        # Any other error
        logger.error(f"OAuth state verification failed: unexpected error. Error: {str(e)}", exc_info=True)
        return False


def create_jwt_token(user_id: str, email: str, role: str) -> str:
    """Create a JWT token for authenticated user."""
    now = datetime.utcnow()
    payload = {
        "sub": user_id,  # Subject (user ID)
        "email": email,
        "role": role,
        "iat": now,  # Issued at
        "exp": now + timedelta(hours=JWT_EXPIRATION_HOURS),
        "jti": secrets.token_urlsafe(16),  # JWT ID for blacklisting
        "iss": JWT_ISSUER,  # Issuer - identifies who created the token
        "aud": JWT_AUDIENCE,  # Audience - identifies who the token is intended for
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> dict:
    """
    Decode and validate JWT token with strict security checks.
    
    Validates:
    - Signature (using JWT_SECRET)
    - Expiration (exp claim)
    - Issuer (iss claim) - must match JWT_ISSUER
    - Audience (aud claim) - must match JWT_AUDIENCE
    - Algorithm (must be HS256)
    
    Raises:
        HTTPException: If token is invalid, expired, or fails validation
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],  # Only accept HS256
            issuer=JWT_ISSUER,  # Validate issuer claim
            audience=JWT_AUDIENCE,  # Validate audience claim
            options={
                "verify_signature": True,  # Explicitly verify signature
                "verify_exp": True,  # Verify expiration
                "verify_iss": True,  # Verify issuer
                "verify_aud": True,  # Verify audience
            }
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidIssuerError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer"
        )
    except jwt.InvalidAudienceError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience"
        )
    except jwt.InvalidSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature"
        )
    except jwt.InvalidTokenError as e:
        # Catch-all for other invalid token errors
        logger = logging.getLogger(__name__)
        logger.warning(f"Invalid token error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


async def verify_google_token(token: str) -> dict:
    """Verify Google OAuth ID token and return user info."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth not configured"
        )
    
    try:
        # Verify the token
        idinfo = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            GOOGLE_CLIENT_ID
        )
        
        # Verify issuer
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')
        
        return {
            "sub": idinfo['sub'],  # Google user ID
            "email": idinfo['email'],
            "name": idinfo.get('name', ''),
            "picture": idinfo.get('picture', ''),
            "hd": idinfo.get('hd', '')  # GSuite domain
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Google token: {str(e)}"
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    db: Session = Depends(get_db)
) -> LabMember:
    """
    Get current authenticated user from JWT token.
    
    Raises 401 if token is missing, invalid, expired, or blacklisted.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    # Check if token is blacklisted
    try:
        payload = decode_jwt_token(token)
        jti = payload.get("jti")
        
        if jti:
            blacklisted = db.query(JWTBlacklist).filter(
                JWTBlacklist.token_jti == jti
            ).first()
            if blacklisted:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
    except HTTPException:
        raise
    except Exception:
        # If decoding fails, decode_jwt_token will raise HTTPException
        pass
    
    # Decode token
    payload = decode_jwt_token(token)
    user_id_str = payload.get("sub")
    
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    # Convert string UUID to UUID object
    from uuid import UUID
    try:
        user_id = UUID(user_id_str) if isinstance(user_id_str, str) else user_id_str
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID in token"
        )
    
    # Get user from database
    user = db.query(LabMember).filter(LabMember.id == user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is suspended"
        )
    
    return user


async def get_current_user_hybrid(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    db: Session = Depends(get_db)
) -> LabMember:
    """
    Get current authenticated user from EITHER signed request auth OR JWT Bearer token.
    
    This hybrid dependency supports:
    1. Signed request authentication (X-Signature, X-Client-Id headers) - used by V0 portal
    2. JWT Bearer token authentication - used by legacy clients
    
    Tries signed auth first, falls back to Bearer token.
    Returns LabMember in both cases.
    """
    from backend.api.signed_auth import get_auth_context, HEADER_SIGNATURE
    
    user_email: Optional[str] = None
    
    # Try signed auth first (check for X-Signature header)
    if request.headers.get(HEADER_SIGNATURE):
        try:
            auth_context = await get_auth_context(request, api_key=None)
            user_email = auth_context.user_email
            logging.getLogger(__name__).debug(f"Signed auth succeeded for {user_email}")
        except HTTPException as e:
            # Signed auth failed, will try Bearer token below
            logging.getLogger(__name__).debug(f"Signed auth failed: {e.detail}")
    
    # Fall back to JWT Bearer token if signed auth didn't work
    if not user_email and credentials:
        try:
            payload = decode_jwt_token(credentials.credentials)
            # Check blacklist
            jti = payload.get("jti")
            if jti:
                blacklisted = db.query(JWTBlacklist).filter(
                    JWTBlacklist.token_jti == jti
                ).first()
                if blacklisted:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked"
                    )
            user_email = payload.get("email", "").lower()
            logging.getLogger(__name__).debug(f"Bearer auth succeeded for {user_email}")
        except HTTPException:
            raise
        except Exception:
            pass
    
    # If neither auth method worked, return 401
    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Look up LabMember by email
    user = db.query(LabMember).filter(
        LabMember.email == user_email
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found in lab members database"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is suspended"
        )
    
    return user


async def get_current_reviewer_hybrid(
    current_user: LabMember = Depends(get_current_user_hybrid)
) -> LabMember:
    """
    Require user (via hybrid auth) to have can_review permission.
    """
    if not current_user.can_review:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to review applications"
        )
    return current_user


async def get_current_admin_hybrid(
    current_user: LabMember = Depends(get_current_user_hybrid)
) -> LabMember:
    """
    Require user (via hybrid auth) to be an admin.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def get_current_reviewer(
    current_user: LabMember = Depends(get_current_user)
) -> LabMember:
    """
    Require user to have can_review permission.
    
    Raises 403 if user cannot review applications.
    """
    if not current_user.can_review:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to review applications"
        )
    return current_user


async def get_current_admin(
    current_user: LabMember = Depends(get_current_user)
) -> LabMember:
    """
    Require user to be an admin.
    
    Raises 403 if user is not an admin.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def blacklist_token(token: str, user_id: str, db: Session):
    """Add token to blacklist."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        payload = decode_jwt_token(token)
        jti = payload.get("jti")
        exp = payload.get("exp")
        
        if jti and exp:
            from uuid import UUID
            
            # Convert user_id string to UUID if provided
            user_id_uuid = None
            if user_id:
                try:
                    user_id_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid user_id format for blacklist: {user_id}")
                    return
            
            expires_at = datetime.utcfromtimestamp(exp)
            blacklist_entry = JWTBlacklist(
                token_jti=jti,
                user_id=user_id_uuid,
                invalidated_at=datetime.utcnow(),
                expires_at=expires_at
            )
            db.add(blacklist_entry)
            db.commit()
    except Exception as e:
        # Log error but don't fail logout
        logger.error(f"Failed to blacklist token: {e}")


def handle_failed_auth(email: str, db: Session, reason: str = "unknown"):
    """
    Handle failed authentication attempt - increment counter and lock account if threshold exceeded.
    
    Locks account after 5 failed attempts for 30 minutes.
    
    USAGE NOTES:
    - For Google OAuth flow: Google handles brute-force protection at their end.
      This function is called when token verification fails AFTER Google auth succeeds,
      which could indicate token tampering or replay attacks.
    - For future non-OAuth login methods: Call this on any failed password attempt.
    
    Args:
        email: User email address (may not exist in DB - that's OK, we don't reveal it)
        db: Database session
        reason: Reason for failure (for logging). E.g., "invalid_token", "expired_token", "suspended"
    """
    logger = logging.getLogger(__name__)
    
    user = db.query(LabMember).filter(LabMember.email == email).first()
    if not user:
        # User doesn't exist - log but don't reveal this to caller
        logger.info(f"Failed auth attempt for non-existent user (not revealing): reason={reason}")
        return
    
    user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
    
    # Lock account after 5 failed attempts
    if user.failed_login_attempts >= 5:
        user.locked_until = datetime.utcnow() + timedelta(minutes=30)
        logger.warning(f"Account locked for user_id={user.id} after {user.failed_login_attempts} failed attempts")
    
    db.commit()
    
    # Log security event
    from backend.api.review_middleware import log_security_event
    log_security_event(
        db=db,
        user_id=str(user.id) if user else None,
        event_type="failed_auth",
        endpoint="/auth/google/callback",
        details={"failed_attempts": user.failed_login_attempts, "reason": reason}
    )


# Backwards compatibility alias (used by tests)
def handle_failed_login(email: str, db: Session):
    """Deprecated: Use handle_failed_auth() instead."""
    handle_failed_auth(email, db, reason="legacy_call")

