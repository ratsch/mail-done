"""
OAuth Handshake Endpoint

Handles the handshake between OAuth-authenticated clients (Web-UI, V0 Portal)
and the backend. Creates ephemeral session keys for signed request authentication.

Flow:
1. Client authenticates with Google OAuth
2. Client generates ephemeral Ed25519 keypair
3. Client sends id_token + public_key to this endpoint
4. Backend verifies id_token, checks user permissions, registers session
5. Backend returns session_id for use in X-Client-Id header

See: docs/SECURITY_DESIGN_REQUEST_SIGNING.md
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from google.oauth2 import id_token
from google.auth.transport import requests

from backend.core.signing.keys import base64_to_public_key
from backend.core.signing.scopes import parse_scopes
from backend.core.signing.ephemeral import ephemeral_registry
from backend.api.review_auth import decode_jwt_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["oauth-handshake"])

# Google OAuth configuration (no default - each client type must specify its own)
# Legacy fallback for backwards compatibility
DEFAULT_GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


class HandshakeRequest(BaseModel):
    """Request body for OAuth handshake."""
    id_token: Optional[str] = Field(None, description="Google OAuth id_token (use this OR jwt_token)")
    jwt_token: Optional[str] = Field(None, description="Backend-issued JWT token (use this OR id_token)")
    public_key: str = Field(..., description="Client's ephemeral Ed25519 public key (base64)")
    client_type: str = Field(..., description="Client type: 'web-ui' or 'v0-portal'")


class HandshakeResponse(BaseModel):
    """Response body for successful handshake."""
    session_id: str = Field(..., description="Session ID for X-Client-Id header")
    expires_in: int = Field(..., description="Session lifetime in seconds")
    scopes: List[str] = Field(..., description="Granted permission scopes")
    user_email: str = Field(..., description="Authenticated user's email")


class OAuthPolicy:
    """OAuth policy for a client type."""
    
    def __init__(self, data: dict):
        self.description = data.get("description", "")
        self.allowed_users: List[str] = data.get("allowed_users", [])
        self.allowed_domains: List[str] = data.get("allowed_domains", [])
        self.scopes: List[str] = data.get("scopes", [])
        self.session_ttl_seconds: int = data.get("session_ttl_seconds", 3600)
        
        # Store env var name, read value dynamically to avoid caching issues
        self._google_client_id_env = data.get("google_client_id_env", "GOOGLE_CLIENT_ID")
    
    @property
    def google_client_id(self) -> Optional[str]:
        """Get Google Client ID (reads from env dynamically to avoid cache issues)."""
        return os.getenv(self._google_client_id_env) or DEFAULT_GOOGLE_CLIENT_ID
    
    def is_user_allowed(self, email: str) -> bool:
        """Check if user is allowed by this policy."""
        email = email.lower()
        
        # Check explicit user list first
        if email in [u.lower() for u in self.allowed_users]:
            return True
        
        # Check domain
        if "@" in email:
            domain = email.split("@")[1]
            if domain.lower() in [d.lower() for d in self.allowed_domains]:
                return True
        
        return False


# Cache for loaded policies
_policies_cache: Optional[dict] = None


def load_oauth_policies() -> dict:
    """Load OAuth policies from config file."""
    global _policies_cache
    
    if _policies_cache is not None:
        return _policies_cache
    
    config_path = Path(__file__).parents[3] / "config" / "oauth_policies.yaml"
    
    if not config_path.exists():
        logger.warning(f"OAuth policies config not found: {config_path}")
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    
    policies = {}
    for client_type, policy_data in config.get("oauth_policies", {}).items():
        policies[client_type] = OAuthPolicy(policy_data)
    
    _policies_cache = policies
    logger.info(f"Loaded OAuth policies for: {list(policies.keys())}")
    return policies


def verify_google_id_token(token: str, google_client_id: Optional[str] = None) -> dict:
    """
    Verify Google id_token and return claims.
    
    Args:
        token: Google id_token JWT
        google_client_id: Expected audience (client ID). Defaults to DEFAULT_GOOGLE_CLIENT_ID.
        
    Returns:
        Token claims dict with 'email', 'email_verified', etc.
        
    Raises:
        HTTPException: If token is invalid
    """
    client_id = google_client_id or DEFAULT_GOOGLE_CLIENT_ID
    
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth not configured (GOOGLE_CLIENT_ID missing)",
        )
    
    try:
        # Verify the token
        claims = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            client_id,
        )
        
        # Check email is verified
        if not claims.get("email_verified", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Email not verified",
            )
        
        return claims
    
    except ValueError as e:
        logger.warning(f"Invalid id_token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid id_token: {e}",
        )


@router.post("/oauth-handshake", response_model=HandshakeResponse)
async def oauth_handshake(request: HandshakeRequest) -> HandshakeResponse:
    """
    Perform OAuth handshake to create ephemeral session.
    
    This endpoint:
    1. Verifies the token (either Google id_token or backend JWT)
    2. Checks user is allowed by the policy for client_type
    3. Validates the public key format
    4. Creates an ephemeral session
    5. Returns session_id and granted scopes
    
    The client should then use session_id as X-Client-Id and sign
    requests with the corresponding private key.
    
    Accepts either:
    - id_token: Google OAuth id_token (for clients doing their own OAuth)
    - jwt_token: Backend-issued JWT (for clients using backend OAuth flow)
    """
    # Validate that exactly one token type is provided
    if not request.id_token and not request.jwt_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either id_token or jwt_token",
        )
    
    if request.id_token and request.jwt_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide only one of id_token or jwt_token, not both",
        )
    
    # Load policies
    policies = load_oauth_policies()
    
    # Validate client type
    if request.client_type not in policies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown client type: {request.client_type}. "
                   f"Valid types: {list(policies.keys())}",
        )
    
    policy = policies[request.client_type]
    
    # Verify token and extract email
    if request.id_token:
        # Verify Google id_token using client-type-specific Client ID
        claims = verify_google_id_token(request.id_token, policy.google_client_id)
        email = claims.get("email", "").lower()
    else:
        # Verify backend JWT token
        try:
            jwt_payload = decode_jwt_token(request.jwt_token)
            email = jwt_payload.get("email", "").lower()
            logger.info(f"JWT handshake for {email} (client_type={request.client_type})")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid JWT token: {e}",
            )
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No email in id_token",
        )
    
    # Check if user is allowed
    if not policy.is_user_allowed(email):
        logger.warning(
            f"User {email} not allowed for {request.client_type}. "
            f"Allowed users: {policy.allowed_users}, "
            f"Allowed domains: {policy.allowed_domains}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User {email} is not authorized for {request.client_type}",
        )
    
    # Parse and validate public key
    try:
        public_key = base64_to_public_key(request.public_key)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid public key: {e}",
        )
    
    # Parse scopes
    scopes = parse_scopes(policy.scopes)
    
    # Create ephemeral session
    session = ephemeral_registry.create_session(
        client_type=request.client_type,
        user_email=email,
        public_key=public_key,
        scopes=scopes,
        ttl_seconds=policy.session_ttl_seconds,
    )
    
    logger.info(
        f"Created session {session.session_id} for {email} "
        f"(type={request.client_type}, ttl={policy.session_ttl_seconds}s)"
    )
    
    return HandshakeResponse(
        session_id=session.session_id,
        expires_in=policy.session_ttl_seconds,
        scopes=policy.scopes,
        user_email=email,
    )


@router.post("/revoke-session")
async def revoke_session(session_id: str) -> dict:
    """
    Revoke an ephemeral session.
    
    This endpoint allows clients to explicitly revoke their session
    (e.g., on logout). The session will no longer be valid for signing.
    
    Note: This endpoint doesn't require authentication - knowing the
    session_id is sufficient to revoke it (similar to a logout token).
    """
    if ephemeral_registry.revoke_session(session_id):
        logger.info(f"Session {session_id} revoked")
        return {"status": "revoked", "session_id": session_id}
    else:
        return {"status": "not_found", "session_id": session_id}


@router.get("/session-info/{session_id}")
async def get_session_info(session_id: str) -> dict:
    """
    Get information about a session (for debugging).
    
    Returns basic info about the session without sensitive data.
    """
    session = ephemeral_registry.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired",
        )
    
    return {
        "session_id": session.session_id,
        "client_type": session.client_type,
        "user_email": session.user_email,
        "scopes": [str(s) for s in session.scopes],
        "created_at": session.created_at,
        "expires_at": session.expires_at,
        "remaining_seconds": session.remaining_seconds,
    }


# Import for signed auth /me endpoint
from backend.api.signed_auth import get_auth_context, AuthContext, HEADER_SIGNATURE
from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Optional Bearer token extractor
optional_bearer = HTTPBearer(auto_error=False)


@router.get("/me")
async def get_signed_user_info(
    request: Request,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer),
):
    """
    Get current authenticated user's information.
    
    Supports BOTH:
    - Signed authentication (X-Client-Id, X-Signature headers) for new OAuth flow
    - JWT Bearer token for legacy/fallback
    
    For V0 portal users, returns LabMember info from the database.
    """
    auth: Optional[AuthContext] = None
    user_email: Optional[str] = None
    client_id: Optional[str] = None
    
    # Try signed auth first (check for X-Signature header)
    if request.headers.get(HEADER_SIGNATURE):
        try:
            auth = await get_auth_context(request, api_key=None)
            user_email = auth.user_email
            client_id = auth.client_id
        except HTTPException:
            # Signed auth failed, try JWT below
            pass
    
    # Fall back to JWT Bearer token
    if not user_email and bearer_token:
        try:
            payload = decode_jwt_token(bearer_token.credentials)
            user_email = payload.get("email", "").lower()
            client_id = "jwt-user"
        except HTTPException:
            pass
    
    # If neither auth method worked, return 401
    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    
    # For V0 portal clients (or JWT users), look up LabMember info
    if client_id and (client_id.startswith("v0-portal") or client_id == "jwt-user"):
        from backend.core.database import get_db
        from backend.core.database.models import LabMember
        
        db = next(get_db())
        try:
            member = db.query(LabMember).filter(
                LabMember.email == user_email
            ).first()
            
            if member:
                return {
                    "id": str(member.id),
                    "email": member.email,
                    "full_name": member.full_name,
                    "role": member.role,
                    "can_review": member.can_review,
                    "is_active": member.is_active,
                    "avatar_url": member.avatar_url,
                    "last_login_at": member.last_login_at.isoformat() if member.last_login_at else None,
                }
            else:
                # User is authenticated but not in LabMember database
                # Return basic info - user may need to be added by admin
                return {
                    "email": user_email,
                    "can_review": False,
                    "is_active": False,
                    "role": "pending",
                }
        finally:
            db.close()
    
    # For other clients (web-ui, scripts), return basic session info
    scopes = [str(s) for s in auth.scopes] if auth else []
    return {
        "email": user_email,
        "client_type": client_id.split("-")[0] if client_id and "-" in client_id else client_id,
        "scopes": scopes,
        "is_active": True,
        "can_review": True,  # Assume web-ui users have access
    }


# =============================================================================
# Multi-Client OAuth Flow
# =============================================================================
# These endpoints allow any client type to use the backend for OAuth,
# without needing Google credentials on the client side.

import secrets
import httpx
from urllib.parse import urlencode
from starlette.responses import RedirectResponse

# OAuth state store (in-memory, with TTL and size limits)
_oauth_states: dict = {}  # state -> {"client_type": str, "redirect_uri": str, "client_state": str, "expires": float}
_MAX_OAUTH_STATES = 1000  # Maximum number of states to store (prevent memory exhaustion)


def _cleanup_expired_states() -> None:
    """Remove expired states and enforce size limit."""
    import time
    current_time = time.time()
    
    # Remove expired states
    expired = [s for s, data in _oauth_states.items() if data["expires"] < current_time]
    for s in expired:
        del _oauth_states[s]
    
    # If still over limit, remove oldest states
    if len(_oauth_states) > _MAX_OAUTH_STATES:
        sorted_states = sorted(_oauth_states.items(), key=lambda x: x[1]["expires"])
        to_remove = len(_oauth_states) - _MAX_OAUTH_STATES
        for state, _ in sorted_states[:to_remove]:
            del _oauth_states[state]


def _generate_oauth_state(client_type: str, redirect_uri: str, client_state: str = "") -> str:
    """Generate and store OAuth state for CSRF protection."""
    import time
    
    # Cleanup before adding new state
    _cleanup_expired_states()
    
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = {
        "client_type": client_type,
        "redirect_uri": redirect_uri,
        "client_state": client_state,  # Pass-through state from client for their CSRF
        "expires": time.time() + 600,  # 10 minute expiry
    }
    return state


def _verify_oauth_state(state: str) -> Optional[dict]:
    """Verify and consume OAuth state. Returns state data or None."""
    import time
    data = _oauth_states.pop(state, None)
    if data and data["expires"] > time.time():
        return data
    return None


@router.get("/oauth/init/{client_type}")
async def oauth_init_for_client(
    client_type: str,
    redirect_uri: Optional[str] = None,
    state: Optional[str] = None,  # Client's CSRF state (passed through)
):
    """
    Initiate OAuth flow for a specific client type.
    
    Each client type can have different Google credentials and policies.
    The client specifies where to redirect after OAuth completes.
    
    Args:
        client_type: "web-ui" or "v0-portal"
        redirect_uri: Where to redirect after OAuth (client's callback URL)
        state: Client's CSRF state (will be passed back unchanged)
    
    Returns:
        Redirect to Google OAuth
    """
    policies = load_oauth_policies()
    
    if client_type not in policies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown client type: {client_type}. Valid: {list(policies.keys())}",
        )
    
    policy = policies[client_type]
    
    if not policy.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google OAuth not configured for {client_type}",
        )
    
    # Validate redirect_uri (security: prevent open redirect)
    if redirect_uri:
        # Only allow HTTPS or localhost for development
        if not (redirect_uri.startswith("https://") or 
                redirect_uri.startswith("http://localhost") or
                redirect_uri.startswith("http://127.0.0.1")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="redirect_uri must use HTTPS (or localhost for development)",
            )
    
    # Get Google OAuth credentials from environment
    google_client_id = policy.google_client_id
    
    # Backend's callback URL
    backend_callback = os.getenv("BACKEND_PUBLIC_URL")
    if not backend_callback:
        raise HTTPException(status_code=500, detail="BACKEND_PUBLIC_URL not configured")
    backend_callback = f"{backend_callback.rstrip('/')}/auth/oauth/callback"
    
    # Generate backend state (includes client's state for pass-through)
    backend_state = _generate_oauth_state(client_type, redirect_uri or "", state or "")
    
    # Build Google OAuth URL
    params = {
        "client_id": google_client_id,
        "redirect_uri": backend_callback,
        "response_type": "code",
        "scope": "openid email profile",
        "state": backend_state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    
    google_auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    
    logger.info(f"OAuth init for {client_type}, redirecting to Google")
    return RedirectResponse(url=google_auth_url)


@router.get("/oauth/callback")
async def oauth_callback_for_client(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    """
    Handle OAuth callback from Google for multi-client flow.
    
    1. Verifies state (CSRF protection)
    2. Exchanges code for tokens
    3. Verifies user is allowed by policy
    4. Creates internal JWT token
    5. Redirects to client's callback URL with token
    """
    if error:
        logger.error(f"OAuth error from Google: {error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth error: {error}",
        )
    
    if not code or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing code or state parameter",
        )
    
    # Verify and consume state
    state_data = _verify_oauth_state(state)
    if not state_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state parameter",
        )
    
    client_type = state_data["client_type"]
    client_redirect_uri = state_data["redirect_uri"]
    client_state = state_data.get("client_state", "")  # Client's CSRF state to pass back
    
    policies = load_oauth_policies()
    policy = policies.get(client_type)
    
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown client type: {client_type}",
        )
    
    # Get credentials for this client type
    google_client_id = policy.google_client_id
    google_client_secret_env = f"GOOGLE_CLIENT_SECRET_{client_type.upper().replace('-', '_')}"
    google_client_secret = os.getenv(google_client_secret_env)
    
    if not google_client_secret:
        # Try fallback naming
        if client_type == "web-ui":
            google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET_WEB_UI")
        elif client_type == "v0-portal":
            google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET_V0_PORTAL")
    
    if not google_client_id or not google_client_secret:
        missing = []
        if not google_client_id:
            missing.append(f"GOOGLE_CLIENT_ID (expected env: {policy._google_client_id_env})")
        if not google_client_secret:
            missing.append(f"GOOGLE_CLIENT_SECRET (expected env: {google_client_secret_env})")
        logger.error(f"OAuth credentials missing for {client_type}: {missing}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth credentials not configured for {client_type}: missing {', '.join(missing)}",
        )
    
    # Backend's callback URL (where Google redirected to)
    backend_callback = os.getenv("BACKEND_PUBLIC_URL")
    if not backend_callback:
        raise HTTPException(status_code=500, detail="BACKEND_PUBLIC_URL not configured")
    backend_callback = f"{backend_callback.rstrip('/')}/auth/oauth/callback"
    
    # Exchange code for tokens
    try:
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": google_client_id,
                    "client_secret": google_client_secret,
                    "redirect_uri": backend_callback,
                    "grant_type": "authorization_code",
                },
            )
            token_response.raise_for_status()
            tokens = token_response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Token exchange failed: {e.response.text}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to exchange authorization code",
        )
    
    id_token_str = tokens.get("id_token")
    if not id_token_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No id_token in Google response",
        )
    
    # Verify id_token
    claims = verify_google_id_token(id_token_str, google_client_id)
    email = claims.get("email", "").lower()
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No email in token",
        )
    
    # Check if user is allowed by policy
    if not policy.is_user_allowed(email):
        logger.warning(f"User {email} not allowed for {client_type}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User {email} is not authorized for {client_type}",
        )
    
    # Create a short-lived JWT for the handshake
    # (Different from the review_auth JWT - this is just for handshake)
    from backend.api.review_auth import create_jwt_token
    
    jwt_token = create_jwt_token(
        user_id=email,  # Use email as user_id for handshake
        email=email,
        role="user",
    )
    
    # Build redirect URL to client
    if client_redirect_uri:
        redirect_params = {
            "token": jwt_token,
            "email": email,
            "client_type": client_type,
        }
        # Include client's CSRF state if provided
        if client_state:
            redirect_params["state"] = client_state
        
        redirect_url = f"{client_redirect_uri}?{urlencode(redirect_params)}"
    else:
        # No redirect URI - just return the token
        result = {
            "token": jwt_token,
            "email": email,
            "client_type": client_type,
        }
        if client_state:
            result["state"] = client_state
        return result
    
    logger.info(f"OAuth complete for {email} ({client_type}), redirecting to client")
    return RedirectResponse(url=redirect_url)
