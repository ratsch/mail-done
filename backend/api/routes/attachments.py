"""
Attachment API Routes

Endpoints for listing and downloading email attachments from IMAP.
"""
import os
import time
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from uuid import UUID
import io

from backend.api.auth import verify_api_key
from backend.api.auditing import audit_log_action
from backend.core.database import get_db
from backend.core.database.models import Email
from backend.core.accounts.manager import AccountManager
from backend.core.email.attachment_extractor import (
    AttachmentExtractor,
    AttachmentNotFoundError,
    AttachmentIndexError,
    EmailNotFoundError,
    FolderNotFoundError,
    IMAPConnectionError
)
from backend.core.config import get_settings
from starlette.requests import Request
import hashlib
from collections import defaultdict
from typing import List as ListType

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/emails", tags=["attachments"])

# Feature flag check
ENABLE_ATTACHMENT_API = os.getenv("ENABLE_ATTACHMENT_API", "false").lower() == "true"
ATTACHMENT_MAX_SIZE_MB = int(os.getenv("ATTACHMENT_MAX_SIZE_MB", "50"))
ATTACHMENT_DOWNLOAD_TIMEOUT = int(os.getenv("ATTACHMENT_DOWNLOAD_TIMEOUT", "120"))
ATTACHMENT_RATE_LIMIT_PER_MINUTE = int(os.getenv("ATTACHMENT_RATE_LIMIT_PER_MINUTE", "25"))

# Simple in-memory rate limiter for attachment downloads
# Key: api_key_hash, Value: list of timestamps
_attachment_rate_limiter: Dict[str, ListType[float]] = defaultdict(list)


def check_feature_enabled():
    """Check if attachment API is enabled"""
    if not ENABLE_ATTACHMENT_API:
        raise HTTPException(
            status_code=403,
            detail="Attachment downloads not enabled. Set ENABLE_ATTACHMENT_API=true"
        )


def check_rate_limit(api_key: str, client_ip: str) -> None:
    """
    Check if attachment download rate limit is exceeded.
    
    Raises HTTPException(429) if limit exceeded.
    """
    if not api_key:
        return  # No API key = no rate limiting (will fail auth anyway)
    
    # Hash API key for rate limiting
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    # Clean old entries (older than 1 minute)
    current_time = time.time()
    _attachment_rate_limiter[api_key_hash] = [
        ts for ts in _attachment_rate_limiter[api_key_hash]
        if current_time - ts < 60
    ]
    
    # Remove empty keys to prevent memory leak over time
    if not _attachment_rate_limiter[api_key_hash]:
        del _attachment_rate_limiter[api_key_hash]
    
    # Check limit (re-check if key exists after cleanup)
    request_count = len(_attachment_rate_limiter.get(api_key_hash, []))
    if request_count >= ATTACHMENT_RATE_LIMIT_PER_MINUTE:
        logger.warning(
            f"Rate limit exceeded for attachment downloads: "
            f"{request_count} requests in last minute "
            f"(limit: {ATTACHMENT_RATE_LIMIT_PER_MINUTE})"
        )
        audit_log_action(
            action="attachment_download_rate_limited",
            details={
                "api_key_hash": api_key_hash,
                "requests_in_minute": request_count,
                "limit": ATTACHMENT_RATE_LIMIT_PER_MINUTE
            },
            ip_address=client_ip
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: Maximum {ATTACHMENT_RATE_LIMIT_PER_MINUTE} downloads per minute"
        )
    
    # Record this request
    _attachment_rate_limiter[api_key_hash].append(current_time)


@router.get("/{email_id}/attachments", dependencies=[Depends(verify_api_key)])
async def list_attachments(
    email_id: UUID,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    List attachments for an email (from database metadata).
    
    **Security:** Feature must be enabled (ENABLE_ATTACHMENT_API=true)
    
    Returns attachment metadata including filename, content_type, and size.
    Does not download attachments - use download endpoint for that.
    
    **Returns:**
    ```json
    [
        {
            "index": 0,
            "filename": "document.pdf",
            "content_type": "application/pdf",
            "size": 1048576
        }
    ]
    ```
    """
    # Check feature flag
    check_feature_enabled()
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Get attachment info from database
    attachment_info = email.attachment_info or []
    
    # Format response
    return [
        {
            "index": idx,
            "filename": att.get("filename", f"attachment_{idx}"),
            "content_type": att.get("content_type", "application/octet-stream"),
            "size": att.get("size", 0)
        }
        for idx, att in enumerate(attachment_info)
    ]


@router.get("/{email_id}/attachments/{index}/download", dependencies=[Depends(verify_api_key)])
async def download_attachment(
    email_id: UUID,
    index: int,
    request: Request,
    db: Session = Depends(get_db)
) -> StreamingResponse:
    """
    Download specific attachment binary from IMAP.
    
    **Security:**
    - Feature must be enabled (ENABLE_ATTACHMENT_API=true)
    - Size limit enforced (default: 50MB)
    - Rate limiting applies (via middleware)
    
    **Returns:**
    - Binary file stream with appropriate Content-Type
    - Content-Disposition header with original filename
    
    **Errors:**
    - 403: Feature disabled
    - 404: Email not found
    - 400: Attachment index out of range
    - 410: Email no longer available on IMAP server
    - 413: Attachment exceeds size limit
    - 503: IMAP server unavailable
    """
    # Check feature flag
    check_feature_enabled()
    
    # Get email from database
    email = db.query(Email).filter(Email.id == email_id).first()
    
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Validate attachment index
    attachment_info = email.attachment_info or []
    if index < 0 or index >= len(attachment_info):
        raise HTTPException(
            status_code=400,
            detail=f"Attachment index {index} out of range (found {len(attachment_info)} attachments)"
        )
    
    # Check size limit
    attachment_meta = attachment_info[index]
    size_bytes = attachment_meta.get("size", 0)
    max_size_bytes = ATTACHMENT_MAX_SIZE_MB * 1024 * 1024
    
    if size_bytes > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Attachment exceeds size limit ({ATTACHMENT_MAX_SIZE_MB}MB)"
        )
    
    # Extract attachment from IMAP
    # Note: AccountManager is lightweight (just reads config), but could be cached if needed
    account_manager = AccountManager()
    extractor = AttachmentExtractor(account_manager)
    
    # Security: Verify account access (defense-in-depth)
    # Check if account restrictions are configured
    allowed_accounts_env = os.getenv("ALLOWED_ACCOUNTS", "")
    if allowed_accounts_env:
        allowed_accounts = [acc.strip() for acc in allowed_accounts_env.split(",") if acc.strip()]
        if allowed_accounts and email.account_id not in allowed_accounts:
            logger.warning(f"Attachment download denied: account {email.account_id} not in allowed list")
            audit_log_action(
                action="attachment_download_denied",
                details={
                    "email_id": str(email_id),
                    "account_id": email.account_id,
                    "attachment_index": index,
                    "reason": "account_not_allowed"
                },
                ip_address=request.client.host if request.client else None
            )
            raise HTTPException(
                status_code=403,
                detail=f"Access denied to attachments from account '{email.account_id}'"
            )
    
    # Get client IP for audit logging
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    api_key = request.headers.get("X-API-Key", "")
    check_rate_limit(api_key, client_ip)
    
    # Callback to update database if email location changed
    def update_email_location(new_folder: str, new_uid: str) -> None:
        """Update email location in database when found in different folder."""
        try:
            email.folder = new_folder
            email.uid = new_uid
            db.commit()
            logger.info(f"Updated email {email_id} location: {new_folder}/{new_uid}")
            audit_log_action(
                action="email_location_updated",
                details={
                    "email_id": str(email_id),
                    "account_id": email.account_id,
                    "new_folder": new_folder,
                    "new_uid": new_uid,
                    "reason": "attachment_download_fallback"
                },
                ip_address=client_ip
            )
        except Exception as e:
            logger.error(f"Failed to update email location: {e}")
            db.rollback()
    
    try:
        # Get attachment binary (with Message-ID fallback for moved emails)
        content, filename, content_type = extractor.get_attachment(
            account_id=email.account_id,
            folder=email.folder,
            uid=email.uid,
            attachment_index=index,
            timeout=ATTACHMENT_DOWNLOAD_TIMEOUT,
            message_id=email.message_id,  # For fallback lookup
            on_location_update=update_email_location  # Update DB if moved
        )
        
        # Verify size matches (safety check)
        actual_size = len(content)
        if actual_size > max_size_bytes:
            audit_log_action(
                action="attachment_download_failed",
                details={
                    "email_id": str(email_id),
                    "account_id": email.account_id,
                    "attachment_index": index,
                    "filename": filename,
                    "size_bytes": actual_size,
                    "reason": "size_limit_exceeded"
                },
                ip_address=client_ip
            )
            raise HTTPException(
                status_code=413,
                detail=f"Downloaded attachment exceeds size limit ({ATTACHMENT_MAX_SIZE_MB}MB)"
            )
        
        # Audit log successful download
        audit_log_action(
            action="attachment_download",
            details={
                "email_id": str(email_id),
                "account_id": email.account_id,
                "attachment_index": index,
                "filename": filename,
                "content_type": content_type,
                "size_bytes": actual_size,
                "folder": email.folder
            },
            ip_address=client_ip
        )
        logger.info(f"Attachment downloaded: {filename} ({actual_size} bytes) from email {email_id}")
        
        # Sanitize filename for Content-Disposition header (prevent header injection)
        # Use RFC 5987 encoding for non-ASCII characters
        from urllib.parse import quote
        safe_filename = filename.replace('"', '\\"').replace('\n', '').replace('\r', '')
        if any(ord(c) > 127 for c in safe_filename):
            # Non-ASCII characters - use RFC 5987 encoding
            content_disposition = f"attachment; filename*=UTF-8''{quote(safe_filename)}"
        else:
            # ASCII only - use simple format
            content_disposition = f'attachment; filename="{safe_filename}"'
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(content),
            media_type=content_type,
            headers={
                "Content-Disposition": content_disposition,
                "Content-Length": str(len(content))
            }
        )
        
    except EmailNotFoundError as e:
        logger.warning(f"Email not found on IMAP: {email_id} - {e}")
        audit_log_action(
            action="attachment_download_failed",
            details={
                "email_id": str(email_id),
                "account_id": email.account_id if email else None,
                "attachment_index": index,
                "reason": "email_not_found_on_imap"
            },
            ip_address=client_ip
        )
        raise HTTPException(
            status_code=410,
            detail="Email no longer available on server"
        )
    except FolderNotFoundError as e:
        logger.warning(f"Folder not found: {email.folder} - {e}")
        raise HTTPException(
            status_code=410,
            detail=f"Email folder '{email.folder}' no longer exists on server"
        )
    except AttachmentIndexError as e:
        logger.warning(f"Attachment index error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except AttachmentNotFoundError as e:
        logger.warning(f"Attachment not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Attachment {index} not found: {e}"
        )
    except IMAPConnectionError as e:
        logger.error(f"IMAP connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="IMAP server unavailable. Please try again later."
        )
    except Exception as e:
        logger.error(f"Unexpected error downloading attachment: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to download attachment"
        )
