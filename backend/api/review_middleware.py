"""
Middleware for Lab Application Review System

Rate limiting and audit logging
"""
import time
import threading
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.core.database import get_db
from backend.core.database.models import SecurityLog, AuditLog, SystemSettings
from backend.api.review_auth import get_current_user, decode_jwt_token
import logging

logger = logging.getLogger(__name__)

# In-memory rate limit storage (token bucket implementation)
# Structure: {key: {"tokens": int, "last_refill": float, "limit": int, "refill_rate": float}}
rate_limit_store: Dict[str, Dict[str, any]] = {}
rate_limit_lock = threading.Lock()  # Thread-safe lock for rate limit store

# Settings cache to reduce DB queries (refreshed every 5 minutes)
_settings_cache: Dict[str, tuple] = {}  # {setting_key: (value, timestamp)}
_settings_cache_lock = threading.Lock()
_settings_cache_ttl = 300  # 5 minutes


def get_rate_limit_key(user_id: str, endpoint: str, limit_type: str) -> str:
    """Generate rate limit key."""
    return f"{user_id}:{endpoint}:{limit_type}"


def get_cached_setting(setting_key: str, db: Session, default_value: any) -> any:
    """
    Get setting from cache or database.
    Reduces DB queries by caching settings for 5 minutes.
    """
    now = time.time()
    
    with _settings_cache_lock:
        if setting_key in _settings_cache:
            value, timestamp = _settings_cache[setting_key]
            if now - timestamp < _settings_cache_ttl:
                return value
            # Cache expired, remove it
            del _settings_cache[setting_key]
    
    # Fetch from database
    try:
        setting = db.query(SystemSettings).filter(SystemSettings.key == setting_key).first()
        if setting and isinstance(setting.value, dict):
            value = setting.value.get("value", default_value)
        else:
            value = default_value
        
        # Update cache
        with _settings_cache_lock:
            _settings_cache[setting_key] = (value, now)
        
        return value
    except Exception as e:
        logger.warning(f"Failed to fetch setting {setting_key}: {e}, using default {default_value}")
        return default_value


def check_rate_limit(user_id: str, endpoint: str, limit_type: str, db: Session, user_role: Optional[str] = None) -> Tuple[bool, Optional[dict]]:
    """
    Check if user has exceeded rate limit using token bucket algorithm with burst handling.
    
    Rate limit configuration:
    - "requests": General API requests (per minute)
      - Default: 60/min (members), 120/min (admins)
      - Burst: 10 tokens (allows 10 rapid requests)
      - Configurable via "rate_limit_requests_per_minute" and "rate_limit_requests_window_seconds" settings
    - "reviews": Review submissions (per hour)
      - Default: 50/hour (members), 100/hour (admins)
      - Burst: 5 tokens (allows 5 rapid reviews)
      - Configurable via "rate_limit_reviews_per_hour" and "rate_limit_reviews_window_seconds" settings
    
    Returns:
        (is_allowed: bool, info: Optional[dict]) - info contains limit details if exceeded
    """
    # Determine limit type and window configuration
    if limit_type == "reviews":
        limit_setting_key = "rate_limit_reviews_per_hour"
        window_setting_key = "rate_limit_reviews_window_seconds"
        default_window = 3600  # 1 hour
        default_limit = 100 if user_role == "admin" else 50
        default_burst = 5  # Allow 5 rapid reviews
    elif limit_type == "requests":
        limit_setting_key = "rate_limit_requests_per_minute"
        window_setting_key = "rate_limit_requests_window_seconds"
        default_window = 60  # 1 minute
        default_limit = 120 if user_role == "admin" else 60
        default_burst = 10  # Allow 10 rapid requests
    else:
        # Unknown limit type, allow (fail open for safety)
        logger.warning(f"Unknown rate limit type: {limit_type}, allowing request")
        return True, None
    
    # Get limit and window from database settings (with caching)
    try:
        limit = get_cached_setting(limit_setting_key, db, default_limit)
        window_seconds = get_cached_setting(window_setting_key, db, default_window)
        # Ensure window_seconds is an integer
        window_seconds = int(window_seconds) if isinstance(window_seconds, (int, float)) else default_window
    except Exception as e:
        # If database query fails, use defaults (fail open for availability)
        logger.warning(f"Failed to fetch rate limit settings: {e}, using defaults")
        limit = default_limit
        window_seconds = default_window
    
    # Calculate refill rate (tokens per second)
    refill_rate = limit / window_seconds
    
    # Use burst as a percentage of limit (allows for natural bursts)
    burst_tokens = max(default_burst, int(limit * 0.1))  # At least default_burst, or 10% of limit
    
    # Token bucket algorithm implementation
    now = time.time()
    key = get_rate_limit_key(user_id, endpoint, limit_type)
    
    # All operations inside lock to ensure atomicity and prevent race conditions
    with rate_limit_lock:
        if key not in rate_limit_store:
            # Initialize bucket with full tokens (allows burst)
            rate_limit_store[key] = {
                "tokens": limit + burst_tokens,  # Start with limit + burst
                "last_refill": now,
                "limit": limit,
                "refill_rate": refill_rate,
                "max_tokens": limit + burst_tokens
            }
            # Consume one token
            rate_limit_store[key]["tokens"] -= 1
            return True, None
        
        store = rate_limit_store[key]
        
        # Refill tokens based on time elapsed (token bucket algorithm)
        time_elapsed = now - store["last_refill"]
        tokens_to_add = time_elapsed * refill_rate
        
        if tokens_to_add > 0:
            # Refill tokens, but don't exceed max
            store["tokens"] = min(store["max_tokens"], store["tokens"] + tokens_to_add)
            store["last_refill"] = now
        
        # Check if we have tokens available
        if store["tokens"] >= 1:
            # Consume one token
            store["tokens"] -= 1
            current_tokens = store["tokens"]
            is_allowed = True
        else:
            # No tokens available
            current_tokens = store["tokens"]
            is_allowed = False
    
    # Outside lock: log violation if exceeded (don't hold lock during DB call)
    if not is_allowed:
        info = {
            "limit": limit,
            "current_tokens": current_tokens,
            "limit_type": limit_type,
            "window_seconds": window_seconds,
            "retry_after": int((1 - current_tokens) / refill_rate) if refill_rate > 0 else window_seconds
        }
        
        # Log violation (non-blocking)
        try:
            log_security_event(
                db=db,
                user_id=user_id,
                event_type="rate_limit",
                endpoint=endpoint,
                details=info
            )
        except Exception as e:
            # Log error but don't fail the rate limit check
            logger.error(f"Failed to log rate limit violation for user {user_id} on {endpoint}: {e}", exc_info=True)
        
        return False, info
    
    return True, None


def cleanup_old_rate_limit_entries(max_age_seconds: int = 3600):
    """
    Clean up old rate limit entries that haven't been accessed recently.
    This prevents memory leaks in long-running processes.
    
    Args:
        max_age_seconds: Remove entries older than this (default: 1 hour)
    """
    now = time.time()
    keys_to_remove = []
    
    with rate_limit_lock:
        for key, store in rate_limit_store.items():
            # Remove entries that haven't been refilled in max_age_seconds
            if now - store.get("last_refill", 0) > max_age_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del rate_limit_store[key]
    
    if keys_to_remove:
        logger.debug(f"Cleaned up {len(keys_to_remove)} old rate limit entries")


# Background cleanup thread (runs every 30 minutes)
_cleanup_thread: Optional[threading.Thread] = None
_cleanup_running = threading.Event()


def start_rate_limit_cleanup_thread():
    """Start background thread to clean up old rate limit entries."""
    global _cleanup_thread
    
    if _cleanup_thread is not None and _cleanup_thread.is_alive():
        return  # Already running
    
    def cleanup_loop():
        while True:
            try:
                cleanup_old_rate_limit_entries(max_age_seconds=3600)
            except Exception as e:
                logger.error(f"Error in rate limit cleanup thread: {e}", exc_info=True)
            
            # Wait 30 minutes before next cleanup (or exit if event is set)
            if _cleanup_running.wait(1800):  # Returns True if event is set (should exit)
                break
    
    _cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True, name="RateLimitCleanup")
    _cleanup_thread.start()
    logger.info("Rate limit cleanup thread started")


def log_security_event(
    db: Session,
    user_id: Optional[str],
    event_type: str,
    endpoint: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[dict] = None
):
    """
    Log security event to database with improved error handling.
    
    This function handles various error scenarios gracefully:
    - Invalid UUID formats
    - Database connection issues
    - Constraint violations
    - Transaction failures
    """
    try:
        from uuid import UUID
        from sqlalchemy.exc import SQLAlchemyError, IntegrityError
        
        # Convert user_id string to UUID if provided
        user_id_uuid = None
        if user_id:
            try:
                user_id_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
            except (ValueError, AttributeError, TypeError) as e:
                # Invalid UUID format, skip logging but don't fail
                logger.warning(f"Invalid user_id format for security log: {user_id} ({type(e).__name__})")
                return
        
        # Validate required fields
        if not event_type:
            logger.warning("Cannot log security event: event_type is required")
            return
        
        security_log = SecurityLog(
            user_id=user_id_uuid,
            event_type=event_type,
            endpoint=endpoint,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        db.add(security_log)
        db.commit()
        
    except IntegrityError as e:
        # Database constraint violation (e.g., foreign key, unique constraint)
        logger.error(f"Database integrity error while logging security event: {e}", exc_info=True)
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Failed to rollback after integrity error: {rollback_error}")
    except SQLAlchemyError as e:
        # General database error
        logger.error(f"Database error while logging security event: {e}", exc_info=True)
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Failed to rollback after database error: {rollback_error}")
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error while logging security event: {type(e).__name__}: {e}", exc_info=True)
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Failed to rollback after unexpected error: {rollback_error}")


def log_audit_event(
    db: Session,
    user_id: str,
    email_id: str,
    action_type: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    action_details: Optional[dict] = None
):
    """
    Log user action to audit log.
    
    Throttles view actions to avoid excessive entries (once per user per application per hour).
    """
    from uuid import UUID
    
    # Convert string UUIDs to UUID objects
    try:
        user_id_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
        email_id_uuid = UUID(email_id) if isinstance(email_id, str) else email_id
    except (ValueError, AttributeError):
        logger.warning(f"Invalid UUID format for audit log: user_id={user_id}, email_id={email_id}")
        return
    
    # Throttle view actions
    if action_type == "view":
        # Check if view logged in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_view = db.query(AuditLog).filter(
            AuditLog.user_id == user_id_uuid,
            AuditLog.email_id == email_id_uuid,
            AuditLog.action_type == "view",
            AuditLog.created_at > one_hour_ago
        ).first()
        
        if recent_view:
            return  # Skip logging duplicate view
    
    try:
        audit_log = AuditLog(
            user_id=user_id_uuid,
            email_id=email_id_uuid,
            action_type=action_type,
            ip_address=ip_address,
            user_agent=user_agent,
            action_details=action_details or {}
        )
        db.add(audit_log)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")
        db.rollback()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for review endpoints."""
    
    async def dispatch(self, request: Request, call_next):
        # Temporarily disable rate limiting for debugging
        return await call_next(request)
        
        # Original code
        # # Only apply to review endpoints
        # if not request.url.path.startswith("/auth") and not request.url.path.startswith("/applications"):
        #     return await call_next(request)
        
        # # Skip for health checks and public endpoints
        # if request.url.path in ["/health", "/"]:
        #     return await call_next(request)
        
        # # Get user from token if available
        # user_id = None
        # try:
        #     auth_header = request.headers.get("Authorization")
        #     if auth_header and auth_header.startswith("Bearer "):
        #         token = auth_header.split(" ")[1]
        #         payload = decode_jwt_token(token)
        #         user_id = payload.get("sub")
        # except Exception:
        #     pass  # Not authenticated, will be handled by endpoint
        
        # if user_id:
        #     # Check rate limit
        #     try:
        #         db = next(get_db())
        #         try:
        #             if not check_rate_limit(user_id, request.url.path, "requests", db):
        #                 return Response(
        #                     content='{"detail": "Rate limit exceeded"}',
        #                     status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        #                     headers={"Retry-After": "60"},
        #                     media_type="application/json"
        #                 )
        #         finally:
        #             db.close()
        #     except Exception as e:
        #         # If database not available (e.g., in tests), skip rate limiting
        #         logger.debug(f"Rate limit check skipped: {e}")
        
        # response = await call_next(request)
        # return response


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Audit logging middleware for review actions."""
    
    async def dispatch(self, request: Request, call_next):
        # Only apply to review endpoints
        if not request.url.path.startswith("/applications"):
            return await call_next(request)
        
        # Get user and email_id from request
        user_id = None
        email_id = None
        
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                payload = decode_jwt_token(token)
                user_id = payload.get("sub")
        except Exception:
            pass
        
        # Extract email_id from path (skip endpoints that don't have email_id like /available-tags)
        if "/applications/" in request.url.path:
            parts = request.url.path.split("/applications/")
            if len(parts) > 1:
                email_id_str = parts[1].split("/")[0]
                # Skip known endpoints that don't have email_id
                if email_id_str not in ["available-tags"]:
                    try:
                        # Validate it's a UUID format before using it
                        UUID(email_id_str)
                        email_id = email_id_str
                    except (ValueError, AttributeError):
                        # Not a valid UUID, skip audit logging for this endpoint
                        pass
        
        response = await call_next(request)
        
        # Log audit event after request (if successful)
        if user_id and email_id and response.status_code < 400:
            action_type = None
            if request.method == "GET":
                action_type = "view"
            elif request.method == "POST" and "/review" in request.url.path:
                action_type = "review_submit"
            elif request.method == "PUT" and "/review" in request.url.path:
                action_type = "review_update"
            elif request.method == "DELETE" and "/review" in request.url.path:
                action_type = "review_delete"
            elif request.method == "POST" and "/decision" in request.url.path:
                action_type = "decision_make"
            elif request.method == "PUT" and "/decision" in request.url.path:
                action_type = "decision_update"
            
            if action_type:
                try:
                    db = next(get_db())
                    try:
                        log_audit_event(
                            db=db,
                            user_id=user_id,
                            email_id=email_id,
                            action_type=action_type,
                            ip_address=request.client.host if request.client else None,
                            user_agent=request.headers.get("user-agent")
                        )
                    finally:
                        db.close()
                except Exception as e:
                    # If database not available (e.g., in tests), skip audit logging
                    logger.debug(f"Audit log skipped: {e}")
        
        return response

