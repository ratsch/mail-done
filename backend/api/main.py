"""
FastAPI Backend for Mail-Done Email Processing System

Phase 2: AI Classification + Database + Web Dashboard
"""
from fastapi import FastAPI, Depends, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import os
import uuid

from backend.api.auth import verify_api_key
from backend.api.routes import emails, stats, tracking, replies, search, applemail, costs, debug, attachments, imap, documents
from backend.api.routes import review_auth, review_applications, review_admin, review_notifications, review_stats, collections
from backend.api.routes import oauth_handshake
from backend.api.review_middleware import RateLimitMiddleware, AuditLogMiddleware, start_rate_limit_cleanup_thread
from backend.api.rate_limiting import GeneralRateLimitMiddleware, cleanup_rate_limiter
from backend.api.auditing import AuditingMiddleware
from backend.api.security_monitor import SecurityMonitorMiddleware, get_security_router
from backend.core.database import init_db
from backend.core.config import get_settings
import logging
import threading

settings = get_settings()
logger = logging.getLogger(__name__)
error_logger = logging.getLogger("api.errors")


# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response


# Create FastAPI app
app = FastAPI(
    title="Mail-Done API",
    description="Email Processing System with AI Classification",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Sensitive patterns to redact from error logs
def _sanitize_error_message(message: str) -> str:
    """
    Scrub potential secrets from exception messages before logging.
    
    Prevents accidental credential leakage in error logs from:
    - Database connection strings with passwords
    - IMAP/SMTP passwords
    - API keys
    - Encryption keys
    """
    import re
    sanitized = message
    
    # Redact database URLs with passwords (postgresql://user:password@host)
    sanitized = re.sub(
        r'(postgresql|postgres|mysql|sqlite)://[^:]+:[^@]+@',
        r'\1://[USER]:[REDACTED]@',
        sanitized,
        flags=re.IGNORECASE
    )
    
    # Redact specific sensitive env vars if they appear in error messages
    sensitive_patterns = [
        (r'(IMAP_PASSWORD|SMTP_PASSWORD|DB_ENCRYPTION_KEY|API_KEY|JWT_SECRET|GOOGLE_CLIENT_SECRET_V0_PORTAL)[=:\s]+[^\s,;]+', r'\1=[REDACTED]'),
        (r'(password|passwd|secret|token|key)["\']?\s*[=:]\s*["\']?[^"\'\s,;]+', r'\1=[REDACTED]'),
    ]
    for pattern, replacement in sensitive_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    # Redact anything that looks like a Fernet key (base64, 44 chars)
    sanitized = re.sub(r'[A-Za-z0-9_-]{43}=', '[REDACTED_KEY]', sanitized)
    
    # Redact anything that looks like a JWT (3 base64 segments separated by dots)
    sanitized = re.sub(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '[REDACTED_JWT]', sanitized)
    
    return sanitized


# Global exception handler for safe error messages
@app.exception_handler(Exception)
async def global_exception_handler(request: FastAPIRequest, exc: Exception):
    """
    Global exception handler that logs detailed errors internally
    but returns safe generic messages to clients.
    
    Security: Sanitizes error messages to prevent credential leakage in logs.
    """
    error_id = str(uuid.uuid4())
    
    # Sanitize exception message before logging to prevent credential leakage
    sanitized_message = _sanitize_error_message(str(exc))
    
    # Log sanitized error details internally with error ID
    error_logger.error(
        f"Error {error_id}: {type(exc).__name__}: {sanitized_message}",
        exc_info=True,
        extra={
            "error_id": error_id,
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else None
        }
    )
    
    # Return generic error message to client
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred",
            "error_id": error_id,
            "message": "The error has been logged. If you need assistance, reference this error ID."
        }
    )


# Startup event to initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    try:
        init_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        # Continue anyway - some endpoints don't need database
    
    # Initialize signed auth system
    try:
        from backend.api.signed_auth import init_auth_system
        from backend.core.signing.persistence import load_sessions_on_startup
        init_auth_system()
        load_sessions_on_startup()
        logger.info("✅ Signed auth system initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize signed auth: {e}")
        # Continue - legacy API_KEY auth will still work
    
    # Start rate limit cleanup thread
    try:
        start_rate_limit_cleanup_thread()
        logger.info("✅ Review rate limit cleanup started")
    except Exception as e:
        logger.error(f"❌ Failed to start review rate limit cleanup: {e}")
    
    # Start general rate limiter cleanup thread
    try:
        def cleanup_loop():
            import time
            while True:
                time.sleep(300)  # Every 5 minutes
                cleanup_rate_limiter()
                logger.debug("Rate limiter cleanup complete")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info("✅ General rate limiter cleanup started")
    except Exception as e:
        logger.error(f"❌ Failed to start rate limiter cleanup: {e}")


# Shutdown event to save sessions
@app.on_event("shutdown")
async def shutdown_event():
    """Save ephemeral sessions on graceful shutdown"""
    try:
        from backend.api.signed_auth import shutdown_auth_system
        shutdown_auth_system()
        logger.info("✅ Signed auth system shutdown complete")
    except Exception as e:
        logger.error(f"❌ Failed to shutdown signed auth: {e}")

# Security headers middleware (add first - outermost)
app.add_middleware(SecurityHeadersMiddleware)

# Security monitor (detects suspicious activity, can lock API)
app.add_middleware(SecurityMonitorMiddleware)

# General auditing (logs all requests)
app.add_middleware(AuditingMiddleware)

# General rate limiting (protects all endpoints)
app.add_middleware(GeneralRateLimitMiddleware)

# Review system middleware (additional limits for review endpoints)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuditLogMiddleware)

# CORS middleware
# Use centralized configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,  # From settings
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type", 
        "X-API-Key", 
        "Authorization",
        # Signed auth headers
        "X-Client-Id",
        "X-Timestamp",
        "X-Nonce",
        "X-Signature",
    ],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Include routers
app.include_router(emails.router)
app.include_router(stats.router)
app.include_router(tracking.router)
app.include_router(replies.router)
app.include_router(search.router)
app.include_router(applemail.router)  # Apple Mail integration
app.include_router(costs.router)  # Cost tracking
app.include_router(debug.router)  # Debug endpoints
app.include_router(attachments.router)  # Attachment downloads
app.include_router(imap.router)  # Direct IMAP access
app.include_router(documents.router)  # Document indexing (Phase 1)

# Signed auth endpoints (OAuth handshake) - MUST be before review_auth for /auth/me priority
# The /auth/me endpoint in oauth_handshake now handles both signed auth and JWT Bearer
app.include_router(oauth_handshake.router)

# Review system routers (review_auth.router has a duplicate /auth/me but won't be reached)
app.include_router(review_auth.router)
app.include_router(review_applications.router)
app.include_router(review_admin.router)
app.include_router(review_notifications.router)
app.include_router(review_stats.router)
app.include_router(collections.router)  # Application collections

# Security endpoints (status, unlock)
app.include_router(get_security_router())

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Mail-Done API",
        "version": "2.0.0",
        "status": "running",
        "phase": "Phase 2",
        "features": [
            "Email processing",
            "AI classification (43 categories)",
            "Database storage (PostgreSQL)",
            "Response tracking (Phase 3)",
            "Reply generation (Phase 3)",
            "Advanced search with pgvector (Phase 3)",
            "Apple Mail integration (color sync)",
            "Cost tracking and analytics",
            "Web dashboard"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)"""
    from sqlalchemy import text
    from backend.core.database import get_db
    
    health = {
        "status": "healthy",
        "version": "2.0.0",
        "database": "unknown",
        "checks": {}
    }
    
    # Check database connectivity
    try:
        db = next(get_db())
        db.execute(text("SELECT 1"))
        health["database"] = "connected"
        health["checks"]["database"] = "ok"
    except Exception as e:
        health["database"] = "disconnected"
        health["checks"]["database"] = f"error: {str(e)}"
        health["status"] = "degraded"  # Still running, but with issues
    
    return health

# API endpoints are now in backend/api/routes/
# - GET /api/emails - List emails with pagination and filters
# - GET /api/emails/{id} - Email details with full metadata
# - PUT /api/emails/{id}/metadata - Update user notes and tags
# - DELETE /api/emails/{id} - Delete email from database
# - GET /api/emails/folders/list - List all folders
# - GET /api/stats - System statistics
# - GET /api/stats/senders - List all senders
# - GET /api/stats/senders/{email} - Sender statistics
# - GET /api/stats/categories/breakdown - Category breakdown

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

