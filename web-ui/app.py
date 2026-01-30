#!/usr/bin/env python3
"""
Mail-Done Web UI
Local web interface for email processing system

This is a lightweight web UI that proxies requests to the Backend API.
It provides a user-friendly interface for:
- Semantic search
- Cost overview
- Triggering inbox processing

Security features:
- Rate limiting to prevent brute-force attacks
- Secure session handling
- CSP and security headers
- Input validation
"""
import os
import asyncio
import subprocess
import shutil
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from collections import defaultdict
import imaplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import make_msgid, formatdate

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends, Request, Cookie, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import httpx
from dotenv import load_dotenv
import sys
import secrets
import hashlib
import traceback

# Import shared auth library for OAuth and signing
try:
    from shared_auth import (
        generate_keypair,
        sign_request,
        public_key_to_base64,
        session_store,
    )
    from shared_auth.handshake import HandshakeError
    OAUTH_AVAILABLE = True
except ImportError as e:
    OAUTH_AVAILABLE = False
    print(f"⚠️  OAuth not available (missing dependencies): {e}")

# Add email-processor to path for database access
email_processor_path = Path(__file__).parent.parent / "email-processor"
if str(email_processor_path) not in sys.path:
    sys.path.insert(0, str(email_processor_path))

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# =============================================================================
# Security Configuration
# =============================================================================

# Environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG_MODE = ENVIRONMENT in ("development", "dev") or os.getenv("DEBUG", "").lower() == "true"

# Platform detection - disable macOS features in Docker/Cloud
IS_MACOS = os.uname().sysname == "Darwin" if hasattr(os, 'uname') else False
ENABLE_MACOS_FEATURES = IS_MACOS and os.getenv("ENABLE_MACOS_FEATURES", "true").lower() == "true"

# Configure logging
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webui")

# Reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def secure_log(message: str, level: str = "info", sensitive: bool = False):
    """Log messages with sensitivity awareness"""
    if sensitive and not DEBUG_MODE:
        return  # Don't log sensitive info in production
    getattr(logger, level)(message)

# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    In-memory rate limiter to prevent brute-force attacks.
    
    Implements sliding window rate limiting with progressive lockout
    for failed authentication attempts.
    """
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.failed_auth: Dict[str, List[float]] = defaultdict(list)
        self.lockouts: Dict[str, float] = {}  # IP -> lockout expiry time
        
        # Configuration
        self.requests_per_minute = int(os.getenv("RATE_LIMIT_RPM", "60"))
        self.auth_attempts_per_minute = int(os.getenv("RATE_LIMIT_AUTH_PM", "5"))
        self.lockout_duration = int(os.getenv("RATE_LIMIT_LOCKOUT_SECONDS", "300"))  # 5 minutes
    
    def _clean_old_entries(self, entries: List[float], window_seconds: float = 60) -> List[float]:
        """Remove entries older than the window"""
        now = time.time()
        cutoff = now - window_seconds
        return [t for t in entries if t > cutoff]
    
    def is_locked_out(self, ip: str) -> bool:
        """Check if IP is currently locked out"""
        if ip in self.lockouts:
            if time.time() < self.lockouts[ip]:
                return True
            else:
                del self.lockouts[ip]
        return False
    
    def get_lockout_remaining(self, ip: str) -> int:
        """Get remaining lockout time in seconds"""
        if ip in self.lockouts:
            remaining = int(self.lockouts[ip] - time.time())
            return max(0, remaining)
        return 0
    
    def check_rate_limit(self, ip: str) -> bool:
        """Check if request is within rate limits. Returns True if allowed."""
        if self.is_locked_out(ip):
            return False
        
        now = time.time()
        self.requests[ip] = self._clean_old_entries(self.requests[ip])
        
        if len(self.requests[ip]) >= self.requests_per_minute:
            return False
        
        self.requests[ip].append(now)
        return True
    
    def record_auth_failure(self, ip: str):
        """Record a failed authentication attempt"""
        now = time.time()
        self.failed_auth[ip] = self._clean_old_entries(self.failed_auth[ip])
        self.failed_auth[ip].append(now)
        
        # Progressive lockout
        failures = len(self.failed_auth[ip])
        if failures >= self.auth_attempts_per_minute:
            # Lock out for progressively longer
            multiplier = min(failures // self.auth_attempts_per_minute, 5)
            lockout_time = self.lockout_duration * multiplier
            self.lockouts[ip] = now + lockout_time
            secure_log(f"IP {ip} locked out for {lockout_time}s after {failures} failed attempts", "warning")
    
    def record_auth_success(self, ip: str):
        """Clear failed auth attempts on successful login"""
        if ip in self.failed_auth:
            del self.failed_auth[ip]
        if ip in self.lockouts:
            del self.lockouts[ip]

# Global rate limiter instance
rate_limiter = RateLimiter()

# =============================================================================
# Security Helper Functions
# =============================================================================

def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies"""
    # Check for forwarded headers (Backend, nginx, etc.)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Take the first IP in the chain
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    
    # Fall back to direct client
    return request.client.host if request.client else "unknown"

def sanitize_error_message(error: Exception) -> str:
    """Sanitize error messages to prevent information leakage"""
    error_str = str(error)
    
    # Remove file paths
    if "/Users/" in error_str or "/home/" in error_str or "/app/" in error_str:
        error_str = "Internal server error"
    
    # Remove stack traces
    if "Traceback" in error_str or "File \"" in error_str:
        error_str = "Internal server error"
    
    # Truncate long errors
    if len(error_str) > 200:
        error_str = error_str[:200] + "..."
    
    return error_str

def log_exception(context: str, e: Exception):
    """Log exception with full traceback in debug mode, sanitized in production"""
    if DEBUG_MODE:
        logger.exception(f"{context}: {e}")
    else:
        secure_log(f"{context}: {sanitize_error_message(e)}", "error")

# Try to import database modules (may not be available if running standalone)
# Web-UI can run in zero-knowledge mode (API proxy only) without database access
try:
    from backend.core.database import get_db, init_db
    from backend.core.database.models import Email, EmailLocationHistory, CrossAccountMove
    from backend.core.database.repository import EmailRepository
    from backend.core.accounts.manager import AccountManager
    from backend.core.email.cross_account_move import CrossAccountMoveService
    from backend.core.email.imap_monitor import IMAPMonitor
    DATABASE_AVAILABLE = True
except (ImportError, ValueError) as e:
    # ValueError: raised when DB_ENCRYPTION_KEY not set (zero-knowledge mode)
    # ImportError: raised when backend modules not in path
    print(f"ℹ️  Running in zero-knowledge mode (no local database access)")
    print(f"   Reason: {e}")
    print("   All requests will proxy to Backend API")
    DATABASE_AVAILABLE = False

# Configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# OAuth Configuration
# Web-UI now delegates OAuth to the backend - no Google credentials needed here!
# The backend handles Google OAuth using per-client-type credentials.

# Get the base URL for OAuth redirect
WEB_UI_BASE_URL = os.getenv("WEB_UI_BASE_URL", "http://localhost:8080")

# OAuth is now always enabled (backend handles it)
OAUTH_ENABLED = OAUTH_AVAILABLE

if OAUTH_ENABLED:
    print("✅ OAuth enabled (zero-knowledge mode - backend handles Google OAuth)")
    print(f"   Backend URL: {BACKEND_API_URL}")
    print(f"   Web-UI URL: {WEB_UI_BASE_URL}")
else:
    print("⚠️  OAuth libraries not available - signing disabled")

# Check that we have a backend URL configured
if not BACKEND_API_URL:
    print("")
    print("=" * 70)
    print("🚨 FATAL ERROR: Backend URL not configured!")
    print("=" * 70)
    print("The web UI requires the backend URL for OAuth and API calls.")
    print("")
    print("Set this environment variable:")
    print("  BACKEND_API_URL=http://localhost:8000")
    print("=" * 70)
    sys.exit(1)



def get_session_from_cookie(request: Request) -> Optional[str]:
    """Extract session ID from cookie."""
    return request.cookies.get("webui_session")


def get_oauth_session(request: Request):
    """
    Get OAuth session from cookie. Returns session or None.
    Use this as a dependency to get the session for signed requests.
    """
    session_id = get_session_from_cookie(request)
    if session_id and OAUTH_AVAILABLE:
        session = session_store.get(session_id)
        if session:
            return session
    return None


async def verify_credentials(request: Request, session = Depends(get_oauth_session)) -> str:
    """
    Verify authentication via OAuth session.

    OAuth is REQUIRED for zero-knowledge mode (signed API requests).
    Returns user email if valid, raises HTTPException if invalid.
    """
    client_ip = get_client_ip(request)
    
    # Check if IP is locked out
    if rate_limiter.is_locked_out(client_ip):
        remaining = rate_limiter.get_lockout_remaining(client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {remaining} seconds.",
            headers={"Retry-After": str(remaining)},
        )
    
    # Check rate limit
    if not rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down.",
            headers={"Retry-After": "60"},
        )
    
    # Check OAuth session
    if session:
        request.state.oauth_session = session
        return session.user_email
    
    # No valid session
    rate_limiter.record_auth_failure(client_ip)
    secure_log(f"Failed auth attempt from {client_ip}", "warning")
    
    raise HTTPException(
        status_code=401,
        detail="Authentication required. Please login via OAuth.",
        headers={"WWW-Authenticate": "Bearer"},
    )

# Create FastAPI app
app = FastAPI(
    title="Mail-Done Web UI",
    description="Web interface for email processing system",
    version="1.0.0"
)

# ============================================================================
# Middleware Configuration (Applied in REVERSE order - last added runs first)
# ============================================================================

# Security Headers Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        
        # Content Security Policy - prevents XSS attacks
        # Build CSP policy dynamically to include Backend URL
        csp_parts = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",  # unsafe-inline for compatibility with UI frameworks
            "style-src 'self' 'unsafe-inline'",   # unsafe-inline needed for inline styles  
            f"img-src 'self' data: {BACKEND_API_URL}",  # Restrict to self, data URIs, and Backend
            f"connect-src 'self' {BACKEND_API_URL}",
            "font-src 'self' data:",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ]
        
        # Only add upgrade-insecure-requests on HTTPS
        if request.url.scheme == "https":
            csp_parts.append("upgrade-insecure-requests")
        
        response.headers["Content-Security-Policy"] = "; ".join(csp_parts)
        
        # Additional security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Only add HSTS if running on HTTPS (Backend)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response

# CORS Configuration - restrict to specific origins for security
ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# Add development ports only if in development mode
if DEBUG_MODE:
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",  # For frontend dev
        "http://127.0.0.1:3000",
    ])
    secure_log("Development mode: Added port 3000 to CORS", "info")

# Allow Backend URL for CORS
if BACKEND_API_URL:
    ALLOWED_ORIGINS.append(BACKEND_API_URL)

# Add self URL if running with a public domain
BACKEND_PUBLIC_DOMAIN = os.getenv("BACKEND_PUBLIC_DOMAIN")
if BACKEND_PUBLIC_DOMAIN:
    public_url = f"https://{BACKEND_PUBLIC_DOMAIN}"
    ALLOWED_ORIGINS.append(public_url)
    secure_log(f"Added public URL to CORS: {public_url}", "info")

# Custom allowed origin from environment
CUSTOM_ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN")
if CUSTOM_ALLOWED_ORIGIN:
    ALLOWED_ORIGINS.append(CUSTOM_ALLOWED_ORIGIN)
    secure_log(f"Added custom allowed origin: {CUSTOM_ALLOWED_ORIGIN}", "info")

secure_log(f"CORS allowed origins: {ALLOWED_ORIGINS}", "info")

# Add middleware in order (last added runs first, so add in reverse):
# 1. Security headers (runs last, after CORS)
app.add_middleware(SecurityHeadersMiddleware)

# 2. CORS (runs first, before security headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")



def get_signed_headers(session, method: str, path: str, body: bytes = b"") -> Dict[str, str]:
    """
    Generate signed headers for a request using OAuth session.
    
    Args:
        session: OAuth session with private_key
        method: HTTP method
        path: Request path
        body: Request body bytes
        
    Returns:
        Dict with signing headers
    """
    if not session or not OAUTH_AVAILABLE:
        return {}
    
    return sign_request(
        private_key=session.private_key,
        client_id=session.session_id,
        method=method,
        path=path,
        body=body,
    )


async def make_signed_request(
    session,
    method: str,
    path: str,
    params: dict = None,
    json_data: dict = None,
) -> httpx.Response:
    """
    Make a signed request to the Backend API using OAuth session.

    Requires valid OAuth session - no fallback to API key.
    """
    if not session:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please login via OAuth."
        )

    if not OAUTH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OAuth libraries not available. Install cryptography package."
        )

    # Build URL with params using httpx to ensure signing matches what's actually sent
    base_url = f"{BACKEND_API_URL.rstrip('/')}{path}"
    
    body = b""
    if json_data:
        import json
        body = json.dumps(json_data).encode("utf-8")

    # Build the actual URL httpx will use, then extract path+query for signing
    temp_url = httpx.URL(base_url, params=params)
    # Get the path with query string as httpx will send it
    signed_path = temp_url.raw_path.decode("utf-8")
    
    logger.debug(f"Signing request: method={method}, path={signed_path}")

    headers = get_signed_headers(session, method, signed_path, body)
    headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=30.0) as client:
        return await client.request(
            method=method,
            url=base_url,
            params=params,
            content=body if body else None,
            headers=headers,
        )

# Global processing state
processing_state = {
    "is_running": False,
    "status": "idle",
    "started_at": None,
    "progress": None,
    "error": None
}


# ============================================================================
# Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # keyword, semantic, hybrid
    limit: int = 20
    category: Optional[str] = None
    similarity_threshold: float = 0.6


class ProcessInboxRequest(BaseModel):
    limit: Optional[int] = 100
    new_only: bool = True
    dry_run: bool = False
    use_ai: bool = True
    generate_embeddings: bool = True


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Verify Backend API connection and log security status"""
    # Log security configuration
    secure_log(f"Environment: {ENVIRONMENT}", "info")
    secure_log(f"Debug mode: {DEBUG_MODE}", "info")
    secure_log(f"Platform: {'macOS' if IS_MACOS else 'Linux/Other'}", "info")
    secure_log(f"macOS features enabled: {ENABLE_MACOS_FEATURES}", "info")
    secure_log(f"Rate limit: {rate_limiter.requests_per_minute} req/min", "info")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_API_URL}/health")
            if response.status_code == 200:
                secure_log(f"Connected to Backend API", "info")
            else:
                secure_log(f"Backend API returned status {response.status_code}", "warning")
    except Exception as e:
        secure_log(f"Could not connect to Backend API: {sanitize_error_message(e)}", "warning")
        secure_log("Make sure BACKEND_API_URL is set correctly in .env", "info")


# ============================================================================
# Frontend
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface (frontend handles auth check via /auth/status)"""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return html_file.read_text()
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mail-Done Web UI</title>
    </head>
    <body>
        <h1>Mail-Done Web UI</h1>
        <p>Loading...</p>
        <p>If you see this, the static files haven't been created yet.</p>
    </body>
    </html>
    """


# ============================================================================
# OAuth Authentication Routes
# ============================================================================

@app.get("/auth/login")
async def oauth_login(request: Request):
    """
    Initiate OAuth login by redirecting to backend's OAuth flow.
    
    The backend handles Google OAuth - web-ui doesn't need Google credentials.
    Includes CSRF protection via state parameter.
    """
    from urllib.parse import quote, urlencode
    
    # Generate CSRF state
    state = secrets.token_urlsafe(32)
    
    # Build redirect URI for after OAuth completes
    callback_uri = f"{WEB_UI_BASE_URL.rstrip('/')}/auth/callback"
    
    # Build backend OAuth URL with redirect_uri and state
    params = urlencode({
        "redirect_uri": callback_uri,
        "state": state,  # Pass state to backend for pass-through
    })
    backend_oauth_url = f"{BACKEND_API_URL.rstrip('/')}/auth/oauth/init/web-ui?{params}"
    
    secure_log(f"Redirecting to backend OAuth (state={state[:10]}...)", "info")
    
    # Create response with redirect
    response = RedirectResponse(url=backend_oauth_url, status_code=303)
    
    # Determine if we should use secure cookies (only for HTTPS)
    is_secure = WEB_UI_BASE_URL.startswith("https://")
    
    # Store state in HTTP-only cookie for CSRF verification
    response.set_cookie(
        key="oauth_state",
        value=state,
        httponly=True,
        secure=is_secure,  # Only secure for HTTPS
        samesite="lax",
        max_age=600,  # 10 minutes
    )
    
    return response


@app.get("/auth/callback")
async def oauth_callback(
    request: Request,
    token: str = Query(None),  # JWT from backend
    email: str = Query(None),
    client_type: str = Query(None),
    state: str = Query(None),  # CSRF state from backend
    error: str = Query(None),
):
    """
    Handle OAuth callback from backend.
    
    The backend handles Google OAuth and redirects here with a JWT token.
    We use that token to perform the handshake and create a signed session.
    Includes CSRF verification via state parameter.
    """
    # Check for OAuth error
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    
    # Verify CSRF state
    stored_state = request.cookies.get("oauth_state")
    if not state or state != stored_state:
        secure_log(f"CSRF state mismatch - possible attack", "error")
        raise HTTPException(status_code=400, detail="Invalid state parameter - possible CSRF attack")
    
    if not token:
        raise HTTPException(status_code=400, detail="No token received from backend")
    
    secure_log(f"OAuth callback received for {email} (state verified)", "info")
    
    try:
        # Generate ephemeral keypair
        private_key, public_key = generate_keypair()
        
        # Import the JWT handshake function
        from shared_auth.handshake import do_handshake_with_jwt
        
        # Perform handshake with backend using the JWT token
        result = await do_handshake_with_jwt(
            backend_url=BACKEND_API_URL,
            jwt_token=token,
            public_key=public_key,
            client_type="web-ui",
        )
        
        # Store session
        session_store.store(
            session_id=result["session_id"],
            private_key=private_key,
            user_email=result.get("user_email", "unknown"),
            expires_in=result.get("expires_in", 3600),
            scopes=result.get("scopes", []),
        )
        
        # Redirect to home with session cookie
        # Determine if we should use secure cookies (only for HTTPS)
        is_secure = WEB_UI_BASE_URL.startswith("https://")
        
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            key="webui_session",
            value=result["session_id"],
            httponly=True,
            secure=is_secure,
            samesite="lax",
            max_age=result.get("expires_in", 3600),
        )
        
        # Clear the OAuth state cookie
        response.delete_cookie("oauth_state")

        # Clear rate limiter lockout on successful login
        client_ip = get_client_ip(request)
        rate_limiter.record_auth_success(client_ip)

        secure_log(f"OAuth login successful: {result.get('user_email', 'unknown')}", "info")
        return response
        
    except HandshakeError as e:
        secure_log(f"OAuth handshake failed: {e}", "error")
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except Exception as e:
        secure_log(f"OAuth callback error: {e}", "error")
        raise HTTPException(status_code=500, detail="Authentication failed")


@app.get("/auth/logout")
async def oauth_logout(request: Request):
    """
    Logout and clear session.
    """
    session_id = get_session_from_cookie(request)
    if session_id and OAUTH_AVAILABLE:
        session_store.delete(session_id)
    
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("webui_session")
    return response


@app.get("/auth/status")
async def auth_status(request: Request):
    """
    Get current authentication status.
    """
    session_id = get_session_from_cookie(request)
    
    if session_id and OAUTH_AVAILABLE:
        session = session_store.get(session_id)
        if session:
            return {
                "authenticated": True,
                "method": "oauth",
                "user_email": session.user_email,
                "expires_in": int(session.remaining_seconds),
                "scopes": session.scopes,
            }
    
    return {
        "authenticated": False,
        "oauth_enabled": OAUTH_ENABLED,
    }


# ============================================================================
# API Endpoints - Semantic Search (Proxy to Backend)
# ============================================================================

@app.post("/api/search")
async def search_emails(
    search_req: SearchRequest,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """
    Semantic search across emails (proxied to Backend API)
    
    Supports:
    - keyword: Traditional text search
    - semantic: Vector similarity search
    - hybrid: Best of both worlds
    
    Requires authentication.
    """
    try:
        response = await make_signed_request(
            session, "GET", "/api/search",
            params={
                "q": search_req.query,
                "mode": search_req.mode,
                "page_size": search_req.limit,
                "category": search_req.category,
                "similarity_threshold": search_req.similarity_threshold,
                "exclude_spam": "true"
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/search/simple")
async def simple_search(
    q: str = Query(..., description="Search query"),
    mode: str = Query("hybrid", description="Search mode"),
    limit: int = Query(20, ge=1, le=100),
    similarity_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Similarity threshold"),
    exclude_handled: bool = Query(False, description="Exclude handled emails"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Simple GET endpoint for search (easier to test)"""
    try:
        params = {
            "q": q,
            "mode": mode,
            "page_size": limit,
            "similarity_threshold": similarity_threshold,
            "exclude_spam": "true"
        }
        if exclude_handled:
            params["exclude_handled"] = "true"
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        
        response = await make_signed_request(session, "GET", "/api/search", params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ============================================================================
# API Endpoints - Cost Overview (New endpoint, adds to Backend API)
# ============================================================================

@app.get("/api/costs/overview")
async def get_cost_overview(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """
    Get comprehensive cost overview with detailed breakdown and projections
    """
    try:
        response = await make_signed_request(
            session, "GET", "/api/costs/overview", params={"days": days}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cost overview: {str(e)}")


@app.get("/api/costs/summary")
async def get_cost_summary(session = Depends(get_oauth_session), _user: str = Depends(verify_credentials)):
    """
    Quick cost summary for dashboard
    """
    try:
        response = await make_signed_request(session, "GET", "/api/costs/summary")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cost summary: {str(e)}")


# ============================================================================
# API Endpoints - Inbox Processing
# ============================================================================

async def run_inbox_processing(
    limit: Optional[int],
    new_only: bool,
    dry_run: bool,
    use_ai: bool,
    generate_embeddings: bool
):
    """Background task to run inbox processing"""
    global processing_state
    
    try:
        processing_state["is_running"] = True
        processing_state["status"] = "running"
        processing_state["started_at"] = datetime.utcnow().isoformat()
        processing_state["error"] = None
        
        # Build command
        cmd = [
            "poetry", "run", "python", "process_inbox.py"
        ]
        
        if dry_run:
            cmd.append("--dry-run")
        
        if new_only:
            cmd.append("--new-only")
        elif limit:
            cmd.extend(["--limit", str(limit)])
        
        if not use_ai:
            cmd.append("--skip-ai")
        
        if not generate_embeddings:
            cmd.append("--skip-embeddings")
        
        # Run in email-processor directory
        working_dir = Path(__file__).parent.parent / "email-processor"
        
        # Run the command
        processing_state["status"] = f"Running: {' '.join(cmd)}"
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            processing_state["status"] = "completed"
            processing_state["progress"] = stdout.decode()[-500:]  # Last 500 chars
        else:
            processing_state["status"] = "failed"
            processing_state["error"] = stderr.decode()[-500:]
        
    except Exception as e:
        processing_state["status"] = "error"
        processing_state["error"] = str(e)
    
    finally:
        processing_state["is_running"] = False


@app.post("/api/process/trigger", dependencies=[Depends(verify_credentials)])
async def trigger_inbox_processing(
    request: ProcessInboxRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger inbox processing
    
    Runs process_inbox.py in the background
    """
    global processing_state
    
    if processing_state["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="Processing already in progress. Please wait for it to complete."
        )
    
    # Add background task
    background_tasks.add_task(
        run_inbox_processing,
        request.limit,
        request.new_only,
        request.dry_run,
        request.use_ai,
        request.generate_embeddings
    )
    
    return {
        "status": "started",
        "message": "Inbox processing started in background",
        "check_status_at": "/api/process/status"
    }


@app.get("/api/process/status", dependencies=[Depends(verify_credentials)])
async def get_processing_status():
    """Get current processing status (requires authentication)"""
    return processing_state


# ============================================================================
# API Endpoints - Stats (Proxy to Backend)
# ============================================================================

@app.get("/api/stats")
async def get_stats(session = Depends(get_oauth_session), _user: str = Depends(verify_credentials)):
    """Get general statistics from Backend API (requires authentication)"""
    try:
        response = await make_signed_request(session, "GET", "/api/stats")
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Backend authentication failed. Please login again."
            )
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Backend API error ({response.status_code}): {response.text[:200]}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ============================================================================
# API Endpoints - Emails (Proxy to Backend)
# ============================================================================

@app.get("/api/emails")
async def get_emails(
    category: Optional[str] = None,
    account: Optional[str] = Query(None, description="Filter by account nickname"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("date", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order"),
    exclude_handled: bool = Query(False, description="Exclude handled emails"),
    exclude_spam: bool = Query(True, description="Exclude spam-marked emails"),
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Get emails list (with optional account filtering)"""
    # If database available and account filter requested, use direct DB access
    if DATABASE_AVAILABLE and account:
        try:
            init_db()
            db_session = next(get_db())
            
            # Build query
            query = db_session.query(Email)
            
            # Filter by account
            query = query.filter(Email.account_id == account)
            
            # Apply other filters
            if exclude_handled:
                # Assuming handled emails have a flag or status
                pass  # Add handling logic if needed
            
            if exclude_spam:
                from sqlalchemy import or_, is_
                from backend.core.database.models import EmailMetadata
                query = query.outerjoin(EmailMetadata).filter(
                    or_(
                        EmailMetadata.email_id.is_(None),
                        EmailMetadata.ai_category != 'spam'
                    )
                )
            
            # Get total count
            total = query.count()
            
            # Apply sorting
            if sort_by == "date":
                order_col = Email.date
            elif sort_by == "subject":
                order_col = Email.subject
            else:
                order_col = Email.date
            
            if sort_order == "desc":
                query = query.order_by(order_col.desc())
            else:
                query = query.order_by(order_col.asc())
            
            # Apply pagination
            offset = (page - 1) * page_size
            emails = query.offset(offset).limit(page_size).all()
            
            # Format response
            result = []
            for email in emails:
                metadata = email.email_metadata
                result.append({
                    'id': str(email.id),
                    'account_id': email.account_id,
                    'message_id': email.message_id,
                    'uid': email.uid,
                    'folder': email.folder,
                    'from_address': email.from_address,
                    'from_name': email.from_name,
                    'subject': email.subject,
                    'date': email.date.isoformat() if email.date else None,
                    'category': metadata.ai_category if metadata else None,
                    'urgency': metadata.ai_urgency if metadata else None,
                    'vip_level': metadata.vip_level if metadata else None,
                    'intended_color': metadata.intended_color if metadata else None,
                    'needs_reply': metadata.needs_reply if metadata else False,
                    'is_seen': email.is_seen,
                    'has_attachments': email.has_attachments
                })
            
            db_session.close()
            
            return {
                'emails': result,
                'total': total,
                'page': page,
                'page_size': page_size
            }
        except Exception as e:
            log_exception("Database query failed", e)
            # Fall through to Backend API proxy
    
    # Fallback to Backend API proxy
    try:
        params = {
            "page": page,
            "page_size": page_size
        }
        if category:
            params["category"] = category
        if account:
            params["account"] = account
        if sort_by:
            params["sort_by"] = sort_by
        if sort_order:
            params["sort_order"] = sort_order
        if exclude_handled:
            params["exclude_handled"] = "true"
        if exclude_spam:
            params["exclude_spam"] = "true"
        
        response = await make_signed_request(session, "GET", "/api/emails", params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        log_exception("Failed to get emails", e)
        raise HTTPException(status_code=500, detail=f"Failed to get emails: {sanitize_error_message(e)}")


class UpdateMetadataRequest(BaseModel):
    user_notes: Optional[str] = None
    project_tags: Optional[List[str]] = None
    awaiting_reply: Optional[bool] = None
    needs_reply: Optional[bool] = None


@app.put("/api/emails/{email_id}/metadata")
async def update_email_metadata(
    email_id: str,
    updates: UpdateMetadataRequest,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Update email metadata (proxy to Backend API, requires authentication)"""
    try:
        response = await make_signed_request(
            session, "PUT", f"/api/emails/{email_id}/metadata",
            json_data=updates.model_dump(exclude_unset=True)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        log_exception("Failed to update metadata", e)
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {sanitize_error_message(e)}")


@app.delete("/api/emails/{email_id}")
async def delete_email(
    email_id: str,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Delete email from database (proxy to Backend API, requires authentication)"""
    try:
        response = await make_signed_request(session, "DELETE", f"/api/emails/{email_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete email: {str(e)}")


# ============================================================================
# IMAP Actions - Proxy to Backend API
# ============================================================================

@app.post("/api/emails/{email_id}/delete")
async def delete_email_to_trash(
    email_id: str,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Move email to trash (proxy to Backend API, requires authentication)"""
    try:
        response = await make_signed_request(session, "POST", f"/api/emails/{email_id}/delete")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete email: {str(e)}")


@app.post("/api/emails/{email_id}/archive")
async def archive_email_action(
    email_id: str,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Archive email (proxy to Backend API, requires authentication)"""
    try:
        response = await make_signed_request(session, "POST", f"/api/emails/{email_id}/archive")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to archive email: {str(e)}")


@app.post("/api/emails/{email_id}/mark-spam")
async def mark_email_as_spam(
    email_id: str,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Mark email as spam (proxy to Backend API, requires authentication)"""
    try:
        response = await make_signed_request(session, "POST", f"/api/emails/{email_id}/mark-spam")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark as spam: {str(e)}")


@app.post("/api/emails/{email_id}/mark-handled")
async def mark_email_as_handled(
    email_id: str,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Mark email as handled (proxy to Backend API, requires authentication)"""
    try:
        response = await make_signed_request(session, "POST", f"/api/emails/{email_id}/mark-handled")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark as handled: {str(e)}")


@app.get("/api/emails/{email_id}/folder-history")
async def get_folder_history(
    email_id: str,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Get email folder history (proxy to Backend API, requires authentication)"""
    try:
        response = await make_signed_request(session, "GET", f"/api/emails/{email_id}/folder-history")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get folder history: {str(e)}")


# ============================================================================
# Mail.app Integration - Open Reply with Prefilled Text
# ============================================================================

class OpenReplyRequest(BaseModel):
    reply_text: Optional[str] = None


@app.post("/api/emails/{email_id}/create-draft-reply")
async def create_draft_reply(
    email_id: str,
    body: OpenReplyRequest,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """
    Create a draft reply via IMAP and open it in Mail.app.
    This is more reliable than AppleScript manipulation.
    
    Requires IMAP credentials to be configured locally.
    Requires authentication.
    
    NOTE: This endpoint is disabled in Docker/Cloud environments.
    """
    # Security: Disable macOS features in Docker/Cloud
    if not ENABLE_MACOS_FEATURES:
        raise HTTPException(
            status_code=501,
            detail="Mail.app integration is disabled. This feature requires running the web UI locally on macOS. "
                   "Set ENABLE_MACOS_FEATURES=true on a macOS system to enable."
        )
    
    try:
        # Check for IMAP credentials
        imap_host = os.getenv('IMAP_HOST')
        imap_user = os.getenv('IMAP_USERNAME')
        imap_pass = os.getenv('IMAP_PASSWORD')
        
        if not all([imap_host, imap_user, imap_pass]):
            raise HTTPException(
                status_code=501,
                detail="IMAP not configured. Set IMAP_HOST, IMAP_USERNAME, IMAP_PASSWORD in .env file to use draft creation."
            )
        
        # Get email details from Backend API
        response = await make_signed_request(session, "GET", f"/api/emails/{email_id}")
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Email not found in database")
        
        email_data = response.json()
        message_id = email_data.get('message_id', '').strip('<>')
        subject = email_data.get('subject', '')
        from_addr = email_data.get('from_address', '')
        references = email_data.get('references', '')
        
        # Build reply
        reply_text = body.reply_text or "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n\nBest regards,"
        reply_subject = subject if subject.startswith('Re:') else f'Re: {subject}'
        
        # Create MIME message
        msg = MIMEMultipart('alternative')
        msg['To'] = from_addr
        msg['Subject'] = reply_subject
        msg['Date'] = formatdate(localtime=True)
        draft_message_id = make_msgid()
        msg['Message-ID'] = draft_message_id
        
        # Add threading headers
        if message_id:
            msg['In-Reply-To'] = f'<{message_id}>'
            if references:
                msg['References'] = f"{references} <{message_id}>"
            else:
                msg['References'] = f'<{message_id}>'
        
        # Add body
        msg.attach(MIMEText(reply_text, 'plain'))
        
        # Connect to IMAP and save draft
        imap = imaplib.IMAP4_SSL(imap_host)
        imap.login(imap_user, imap_pass)
        
        # Append to Drafts folder
        imap.append('Drafts', '\\Draft', None, msg.as_bytes())
        imap.logout()
        
        secure_log(f"Draft created with message-id: {draft_message_id}", "debug", sensitive=True)
        
        # Give Mail.app a moment to sync the draft
        await asyncio.sleep(0.5)
        
        # Open the draft in Mail.app
        draft_message_id_clean = draft_message_id.strip('<>')
        applescript = f'''
        set msgURL to "message://%3C{draft_message_id_clean}%3E"
        open location msgURL
        delay 0.3
        tell application "Mail" to activate
        return "success"
        '''
        
        result = subprocess.run(
            ['/usr/bin/osascript', '-e', applescript],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Draft reply created and opened in Mail.app",
                "draft_message_id": draft_message_id
            }
        else:
            return {
                "status": "partial_success",
                "message": f"Draft created in Drafts folder. Open Mail.app to see it.",
                "draft_message_id": draft_message_id
            }
    
    except HTTPException:
        raise
    except Exception as e:
        log_exception("Failed to create draft", e)
        raise HTTPException(status_code=500, detail=f"Failed to create draft: {sanitize_error_message(e)}")


@app.post("/api/emails/{email_id}/open-reply")
async def open_reply_in_mail(
    email_id: str,
    body: OpenReplyRequest,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """
    Open a reply window in Mail.app with optional prefilled text.
    This uses AppleScript to automate Mail.app.
    
    Requires:
    - macOS with Mail.app
    - Email must exist in Mail.app (indexed by message_id)
    - Authentication
    
    NOTE: This endpoint is disabled in Docker/Cloud environments.
    
    Args:
        email_id: The email database ID
        request.reply_text: Optional text to prefill in the reply
    
    Returns:
        Success status and details
    """
    # Security: Disable macOS features in Docker/Cloud
    if not ENABLE_MACOS_FEATURES:
        raise HTTPException(
            status_code=501,
            detail="Mail.app integration is disabled. This feature requires running the web UI locally on macOS. "
                   "Run locally with: cd web-ui && python3.11 app.py"
        )
    
    try:
        # Get email details from Backend API
        response = await make_signed_request(session, "GET", f"/api/emails/{email_id}")
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Email not found in database")
        
        email_data = response.json()
        message_id = email_data.get('message_id')
        
        if not message_id:
            raise HTTPException(
                status_code=400, 
                detail="Email has no message_id. Cannot open in Mail.app"
            )
        
        # Clean message_id (remove < > brackets if present)
        original_message_id = message_id
        message_id = message_id.strip().strip('<>').strip()
        
        # Log for debugging (only in debug mode)
        secure_log(f"Original message_id: {repr(original_message_id)}", "debug", sensitive=True)
        secure_log(f"Cleaned message_id: {repr(message_id)}", "debug", sensitive=True)
        
        # Get reply text or use empty string
        reply_text = body.reply_text or ""
        
        # For clipboard paste, keep the actual newline characters
        reply_text_for_typing = (reply_text
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r\n", "\n")
        )
        
        # Also escape for AppleScript string
        reply_text_escaped = (reply_text
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r"))
        
        # AppleScript to find message and open reply window
        applescript = f'''
        set msgURL to "message://%3C{message_id}%3E"
        open location msgURL
        
        tell application "Mail" to activate
        delay 0.3
        
        tell application "System Events"
            tell process "Mail"
                keystroke "r" using {{command down, shift down}}
            end tell
        end tell
        
        delay 1.2
        
        tell application "System Events"
            tell process "Mail"
                set frontmost to true
                key code 126 using command down
                delay 0.1
            end tell
        end tell
        
        set the clipboard to "{reply_text_for_typing}"
        
        tell application "System Events"
            tell process "Mail"
                keystroke "v" using command down
                keystroke return
                keystroke return
            end tell
        end tell
        
        return "success"
        '''
        
        # Check if running on macOS
        platform = os.uname().sysname if hasattr(os, 'uname') else "unknown"
        if platform != "Darwin":
            raise HTTPException(
                status_code=501,
                detail=f"Mail.app reply feature requires macOS. Server is running on {platform}."
            )
        
        # Find osascript
        osascript_path = shutil.which('osascript')
        if not osascript_path:
            for path in ['/usr/bin/osascript', '/bin/osascript']:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    osascript_path = path
                    break
        
        if not osascript_path:
            raise FileNotFoundError("osascript not found")
        
        secure_log(f"Executing AppleScript (first 100 chars): {applescript[:100]}", "debug", sensitive=True)
        
        result = subprocess.run(
            [osascript_path, '-e', applescript],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        secure_log(f"AppleScript return code: {result.returncode}", "debug", sensitive=True)
        if result.stderr:
            secure_log(f"AppleScript stderr: {result.stderr[:100]}", "debug", sensitive=True)
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Reply window opened in Mail.app",
                "email_id": email_id,
                "message_id": message_id,
                "prefilled_text_length": len(reply_text)
            }
        else:
            error_msg = result.stderr.strip()
            secure_log(f"AppleScript failed: {error_msg[:100]}", "error", sensitive=True)
            raise HTTPException(
                status_code=500,
                detail="Mail.app error. Check that Mail.app is running and accessible."
            )
    
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500, 
            detail="AppleScript timeout. Mail.app may not be responding."
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"osascript not found. This feature requires macOS. Error: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        log_exception("Failed to open reply", e)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to open reply: {sanitize_error_message(e)}"
        )


@app.get("/api/emails/{email_id}")
async def get_single_email(
    email_id: str,
    session = Depends(get_oauth_session),
    _user: str = Depends(verify_credentials)
):
    """Get a single email by ID with location history (requires authentication)"""
    # If database available, use direct DB access to include location_history
    if DATABASE_AVAILABLE:
        try:
            init_db()
            db_session = next(get_db())
            
            email = db_session.query(Email).filter(Email.id == email_id).first()
            if not email:
                db_session.close()
                raise HTTPException(status_code=404, detail="Email not found")
            
            metadata = email.email_metadata
            
            # Get location history
            history = db_session.query(EmailLocationHistory).filter(
                EmailLocationHistory.email_id == email.id
            ).order_by(EmailLocationHistory.moved_at.desc()).all()
            
            location_history = [{
                'from_account': h.from_account_id,
                'from_folder': h.from_folder,
                'to_account': h.to_account_id,
                'to_folder': h.to_folder,
                'moved_at': h.moved_at.isoformat() if h.moved_at else None,
                'moved_by': h.moved_by,
                'move_reason': h.move_reason,
                'is_cross_account': h.is_cross_account
            } for h in history]
            
            # Decrypt body if needed
            body_text = email.body_text
            body_markdown = email.body_markdown
            
            db_session.close()
            
            return {
                'id': str(email.id),
                'account_id': email.account_id,
                'message_id': email.message_id,
                'uid': email.uid,
                'folder': email.folder,
                'from_address': email.from_address,
                'from_name': email.from_name,
                'to_addresses': email.to_addresses,
                'cc_addresses': email.cc_addresses,
                'subject': email.subject,
                'date': email.date.isoformat() if email.date else None,
                'body_text': body_text,
                'body_markdown': body_markdown,
                'has_attachments': email.has_attachments,
                'attachment_info': email.attachment_info,
                'metadata': {
                    'category': metadata.ai_category if metadata else None,
                    'subcategory': metadata.ai_subcategory if metadata else None,
                    'urgency': metadata.ai_urgency if metadata else None,
                    'vip_level': metadata.vip_level if metadata else None,
                    'summary': metadata.ai_summary if metadata else None,
                    'needs_reply': metadata.needs_reply if metadata else False,
                    'reply_deadline': metadata.reply_deadline.isoformat() if metadata and metadata.reply_deadline else None,
                    'answer_options': metadata.answer_options if metadata else None
                },
                'location_history': location_history
            }
        except HTTPException:
            raise
        except Exception as e:
            log_exception("Database query for single email failed", e)
            # Fall through to Backend API proxy
    
    # Fallback to Backend API proxy
    try:
        response = await make_signed_request(session, "GET", f"/api/emails/{email_id}")
        
        if response.status_code == 200:
            email_data = response.json()
            
            # Debug: log answer_options if present
            if DEBUG_MODE:
                metadata = email_data.get('email_metadata', {})
                answer_options = metadata.get('answer_options')
                secure_log(f"Email {email_id} has {len(answer_options) if answer_options else 0} answer options", "debug")
            
            return email_data
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend API error: {response.text}"
            )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to backend API. Please try again later."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get email: {str(e)}")




# ============================================================================
# Multi-Account API Endpoints
# ============================================================================

@app.get("/api/accounts", dependencies=[Depends(verify_credentials)])
async def get_accounts():
    """Get list of all configured accounts (requires authentication)"""
    # In zero-knowledge mode, return empty accounts (no local config access)
    if not DATABASE_AVAILABLE:
        return {'accounts': [], 'default_account': None}
    
    try:
        account_manager = AccountManager()
        accounts = []
        
        for nickname in account_manager.list_accounts():
            info = account_manager.get_account_display_info(nickname)
            accounts.append({
                'nickname': nickname,
                'display_name': info['display_name'],
                'email': info['email'],
                'host': info['host'],
                'is_default': nickname == account_manager.default_account,
                'allow_moves_to': info.get('allow_moves_to', [])
            })
        
        return {'accounts': accounts, 'default_account': account_manager.default_account}
    except Exception as e:
        log_exception("Failed to get accounts", e)
        raise HTTPException(status_code=500, detail=f"Failed to get accounts: {sanitize_error_message(e)}")


class MoveToAccountRequest(BaseModel):
    target_account: str
    target_folder: str = "INBOX"


@app.post("/api/emails/{email_id}/move-to-account", dependencies=[Depends(verify_credentials)])
async def move_email_to_account(email_id: str, request: MoveToAccountRequest):
    """Move email to different account (UI-triggered cross-account move, requires authentication)"""
    if not DATABASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Database access required for cross-account moves"
        )
    
    try:
        # Initialize database and account manager
        init_db()
        account_manager = AccountManager()
        db_session = next(get_db())
        
        # Get email from database
        email = db_session.query(Email).filter(Email.id == email_id).first()
        if not email:
            db_session.close()
            raise HTTPException(status_code=404, detail="Email not found")
        
        # Check if move is allowed
        if not account_manager.can_move_between(email.account_id, request.target_account):
            db_session.close()
            raise HTTPException(
                status_code=403,
                detail=f"Moves from {email.account_id} to {request.target_account} not allowed"
            )
        
        # Check for concurrent move (race condition prevention)
        existing_move = db_session.query(CrossAccountMove).filter(
            CrossAccountMove.message_id == email.message_id,
            CrossAccountMove.status.in_(['pending', 'retrying', 'in_progress'])
        ).first()
        
        if existing_move:
            db_session.close()
            raise HTTPException(
                status_code=409,
                detail=f"Move already in progress for this email (initiated by: {existing_move.initiated_by})"
            )
        
        # Create move record (don't commit yet - wait for move result)
        move_record = CrossAccountMove(
            email_id=email.id,
            message_id=email.message_id or '',
            from_account_id=email.account_id,
            from_folder=email.folder or 'INBOX',
            to_account_id=request.target_account,
            to_folder=request.target_folder,
            move_method='pending',
            status='pending',
            initiated_by='ui'
        )
        db_session.add(move_record)
        # Don't commit yet - wait for move to succeed
        
        # Execute move asynchronously
        try:
            from_config = account_manager.get_imap_config(email.account_id)
            from_imap = IMAPMonitor(from_config)
            from_imap.connect()
            
            cross_account_service = CrossAccountMoveService(
                account_manager,
                dry_run=False
            )
            
            success, error, move_method = await cross_account_service.move_email(
                email_uid=str(email.uid) if email.uid else '',
                message_id=email.message_id or '',
                from_account=email.account_id,
                from_folder=email.folder or 'INBOX',
                to_account=request.target_account,
                to_folder=request.target_folder,
                from_imap=from_imap
            )
            
            from_imap.disconnect()
            
            if success:
                # Update move record
                move_record.status = 'completed'
                move_record.completed_at = datetime.utcnow()
                move_record.move_method = move_method or 'unknown'
                
                # Update email record location
                repo = EmailRepository(db_session)
                repo.track_location_change(
                    email,
                    new_folder=request.target_folder,
                    new_account=request.target_account,
                    moved_by='ui',
                    move_reason='User-initiated move from UI'
                )
                
                # Commit transaction only after successful move
                db_session.commit()
                db_session.close()
                
                return {
                    'success': True,
                    'message': f'Email moved successfully to {request.target_account}:{request.target_folder}'
                }
            else:
                move_record.status = 'failed'
                move_record.error_message = error
                move_record.retry_count = 0
                move_record.next_retry_at = datetime.utcnow()
                
                # Commit failed move record for retry tracking
                db_session.commit()
                db_session.close()
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to move email: {error}"
                )
                
        except HTTPException:
            # Re-raise HTTP exceptions
            db_session.rollback()
            db_session.close()
            raise
        except Exception as e:
            move_record.status = 'failed'
            move_record.error_message = str(e)
            db_session.commit()  # Commit failed record for tracking
            db_session.close()
            raise HTTPException(
                status_code=500,
                detail=f"Error executing move: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        log_exception("Failed to move email", e)
        raise HTTPException(status_code=500, detail=f"Failed to move email: {sanitize_error_message(e)}")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint - checks Backend API connectivity.
    
    Returns minimal info in production to avoid information leakage.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_API_URL}/health", timeout=5.0)
            
            if response.status_code == 200:
                # In production, don't expose internal API details
                if DEBUG_MODE:
                    backend_health = response.json()
                    return {
                        "status": "healthy",
                        "web_ui": "running",
                        "backend_api": backend_health,
                        "macos_features": ENABLE_MACOS_FEATURES
                    }
                else:
                    return {"status": "healthy"}
            else:
                if DEBUG_MODE:
                    return {
                        "status": "degraded",
                        "web_ui": "running",
                        "backend_api": "unavailable",
                        "backend_status_code": response.status_code
                    }
                else:
                    return {"status": "degraded"}
    except Exception as e:
        secure_log(f"Health check failed: {e}", "error", sensitive=True)
        if DEBUG_MODE:
            return {
                "status": "degraded",
                "web_ui": "running",
                "backend_api": "unreachable",
                "error": sanitize_error_message(e)
            }
        else:
            return {"status": "degraded"}


@app.get("/api/diagnostics/macos", dependencies=[Depends(verify_credentials)])
async def macos_diagnostics():
    """
    Diagnostic endpoint to check macOS/osascript availability (requires authentication).
    
    Returns reduced information in production mode.
    """
    diagnostics = {
        "platform": os.uname().sysname if hasattr(os, 'uname') else "unknown",
        "macos_features_enabled": ENABLE_MACOS_FEATURES,
        "is_macos": IS_MACOS,
        "debug_mode": DEBUG_MODE
    }
    
    # Only include detailed info in debug mode
    if DEBUG_MODE:
        diagnostics["osascript_which"] = shutil.which('osascript')
        diagnostics["osascript_locations"] = {}
        
        # Check common locations
        for path in ['/usr/bin/osascript', '/bin/osascript', '/usr/local/bin/osascript']:
            diagnostics["osascript_locations"][path] = {
                "exists": os.path.isfile(path),
                "executable": os.access(path, os.X_OK) if os.path.isfile(path) else False
            }
        
        # Try to run osascript
        try:
            osascript_path = shutil.which('osascript') or '/usr/bin/osascript'
            result = subprocess.run(
                [osascript_path, '-e', 'return "test"'],
                capture_output=True,
                text=True,
                timeout=5
            )
            diagnostics["osascript_test"] = {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }
        except Exception as e:
            diagnostics["osascript_test"] = {
                "success": False,
                "error": sanitize_error_message(e)
            }
    
    return diagnostics


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WEB_UI_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

