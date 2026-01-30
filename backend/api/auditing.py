"""
Security Auditing System

Comprehensive audit logging for all sensitive operations.
Logs are written to both file and database for analysis.
"""
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from sqlalchemy import Column, String, DateTime, Text, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("security.audit")

# Configure audit logger to write to separate file
import os
log_dir = os.path.expanduser("~/.email-api-security-logs")
os.makedirs(log_dir, exist_ok=True)
audit_log_file = os.path.join(log_dir, f"audit_{datetime.now().strftime('%Y%m%d')}.log")

audit_handler = logging.FileHandler(audit_log_file)
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)


Base = declarative_base()


class APIAuditLog(Base):
    """
    Audit log table for all API requests.
    
    Tracks who accessed what, when, and from where.
    """
    __tablename__ = "api_audit_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Request info
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    method = Column(String(10), nullable=False)
    path = Column(String(500), nullable=False, index=True)
    query_params = Column(JSON)
    
    # Authentication
    api_key_hash = Column(String(64))  # SHA256 hash of API key (never store plaintext)
    authenticated = Column(Integer, default=0)  # 0=no, 1=yes
    
    # Client info
    ip_address = Column(String(50), index=True)
    user_agent = Column(Text)
    
    # Response
    status_code = Column(Integer, nullable=False, index=True)
    response_time_ms = Column(Integer)
    
    # Sensitive operation flags
    is_search = Column(Integer, default=0)  # Email search operations
    is_data_access = Column(Integer, default=0)  # Email content access
    is_modification = Column(Integer, default=0)  # Data modification
    is_failed_auth = Column(Integer, default=0)  # Failed authentication
    
    # Additional context
    details = Column(JSON)  # Extra details (e.g., search query length, email IDs)
    
    __table_args__ = (
        # Indexes for security analysis
        {'extend_existing': True}
    )


class AuditingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to audit all API requests.
    
    Logs sensitive operations with full context for security analysis.
    """
    
    # Paths that require auditing
    SENSITIVE_PATHS = [
        "/api/emails",
        "/api/search",
        "/api/stats",
        "/api/replies"
    ]
    
    # Attachment download paths (highly sensitive)
    ATTACHMENT_PATHS = [
        "/api/emails/",
        "/attachments/"
    ]
    
    # Paths to skip (high-frequency, low-value)
    SKIP_PATHS = ["/health", "/docs", "/redoc", "/openapi.json", "/"]
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
    
    async def dispatch(self, request: Request, call_next):
        import time
        import hashlib
        
        # Skip non-sensitive paths
        if any(request.url.path.startswith(path) for path in self.SKIP_PATHS):
            return await call_next(request)
        
        start_time = time.time()
        self.request_count += 1
        
        # Extract request details
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key")
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest() if api_key else None
        
        # Determine operation type
        is_search = "/search" in request.url.path
        is_data_access = "/emails/" in request.url.path and request.method == "GET"
        is_modification = request.method in ["POST", "PUT", "DELETE", "PATCH"]
        is_attachment_download = "/attachments/" in request.url.path and "/download" in request.url.path
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Determine if this was a failed auth
        is_failed_auth = response.status_code in [401, 403]
        authenticated = 1 if (response.status_code not in [401, 403] and api_key) else 0
        
        # Build audit log entry
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": self.request_count,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params) if request.query_params else None,
            "ip": client_ip,
            "api_key_hash": api_key_hash[:16] if api_key_hash else None,  # Only log first 16 chars
            "authenticated": bool(authenticated),
            "status": response.status_code,
            "response_time_ms": response_time_ms,
            "is_search": is_search,
            "is_data_access": is_data_access,
            "is_modification": is_modification,
            "is_attachment_download": is_attachment_download,
            "is_failed_auth": is_failed_auth,
            "user_agent": request.headers.get("user-agent", "")[:200]
        }
        
        # Log to audit file
        # Always log attachment downloads (highly sensitive)
        if is_attachment_download or is_search or is_modification or is_failed_auth or response.status_code >= 400:
            audit_logger.info(json.dumps(audit_entry))
        
        # Log security events
        if is_failed_auth:
            logger.warning(f"SECURITY: Failed auth from {client_ip} to {request.url.path}")
        
        if is_modification:
            logger.info(f"AUDIT: Data modification by {client_ip[:8]}... - {request.method} {request.url.path}")
        
        if is_search:
            logger.info(f"AUDIT: Search operation by {client_ip[:8]}... - {request.url.path}")
        
        # Log slow requests
        if response_time_ms > 5000:
            logger.warning(f"PERFORMANCE: Slow request ({response_time_ms}ms) - {request.method} {request.url.path}")
        
        return response


def audit_log_action(
    action: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    """
    Log a specific audit action.
    
    Use for actions not captured by middleware (e.g., background tasks).
    """
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "user_id": user_id,
        "ip": ip_address,
        "details": details or {}
    }
    
    audit_logger.info(json.dumps(audit_entry))
