"""
IMAP Direct Access API endpoints

Provides direct access to IMAP server operations like:
- Listing folders
- Listing emails in a folder (without processing/storing)

Supports multi-account configuration via AccountManager.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import logging
from datetime import datetime

from backend.api.auth import verify_api_key
from backend.core.accounts.manager import AccountManager
from backend.core.email.imap_monitor import IMAPMonitor, IMAPConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/imap", tags=["imap"])

# Cache AccountManager instance (lightweight, just reads config)
_account_manager: Optional[AccountManager] = None


def get_account_manager() -> AccountManager:
    """Get or create AccountManager singleton."""
    global _account_manager
    if _account_manager is None:
        try:
            _account_manager = AccountManager()
        except Exception as e:
            logger.error(f"Failed to initialize AccountManager: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Account configuration error: {str(e)}"
            )
    return _account_manager


# Response schemas
class IMAPFolderResponse(BaseModel):
    """Response for folder listing"""
    folders: List[str]
    total: int
    account: str


class IMAPMessageSummary(BaseModel):
    """Summary of an email message from IMAP"""
    uid: str
    message_id: Optional[str] = None
    subject: Optional[str] = None
    from_address: Optional[str] = None
    from_name: Optional[str] = None
    date: Optional[str] = None
    is_seen: bool = False
    is_flagged: bool = False
    has_attachments: bool = False


class IMAPFolderMessagesResponse(BaseModel):
    """Response for folder messages listing"""
    folder: str
    messages: List[IMAPMessageSummary]
    total: int
    account: str


class IMAPFolderStatusResponse(BaseModel):
    """Response for folder status (message counts)"""
    folder: str
    total: int
    unseen: int
    recent: int
    uidnext: int
    uidvalidity: int
    account: str


def _get_imap_config(account: Optional[str] = None) -> tuple[IMAPConfig, str]:
    """
    Get IMAP configuration for an account.
    
    Args:
        account: Account nickname (e.g., 'work', 'personal'). 
                 If None, uses default account.
    
    Returns:
        Tuple of (IMAPConfig, account_nickname)
    """
    manager = get_account_manager()
    
    try:
        # Get the account (uses default if None)
        account_nickname = account or manager.default_account
        imap_config = manager.get_imap_config(account_nickname)
        
        # Validate credentials are present
        # For OAuth2 accounts, we need username + oauth2 credentials (refresh_token or client_secret)
        # For password accounts, we need username + password
        sanitized = account_nickname.upper().replace('-', '_').replace(' ', '_')

        if not imap_config.username:
            raise HTTPException(
                status_code=503,
                detail=f"IMAP username not configured for account '{account_nickname}'. "
                       f"Set IMAP_USERNAME_{sanitized}."
            )

        if imap_config.auth_type == "oauth2":
            # OAuth2 authentication - check for refresh token or client secret
            if not imap_config.oauth2_refresh_token and not imap_config.oauth2_client_secret:
                raise HTTPException(
                    status_code=503,
                    detail=f"OAuth2 credentials not configured for account '{account_nickname}'. "
                           f"Set OAUTH2_REFRESH_TOKEN_{sanitized} or OAUTH2_CLIENT_SECRET_{sanitized}."
                )
        else:
            # Password authentication
            if not imap_config.password:
                raise HTTPException(
                    status_code=503,
                    detail=f"IMAP password not configured for account '{account_nickname}'. "
                           f"Set IMAP_PASSWORD_{sanitized}."
                )
        
        return imap_config, account_nickname
        
    except ValueError as e:
        # Account not found
        available = ', '.join(manager.list_accounts())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown account: '{account}'. Available accounts: {available}"
        )


@router.get("/folders", response_model=IMAPFolderResponse, dependencies=[Depends(verify_api_key)])
async def list_imap_folders(
    account: Optional[str] = Query(None, description="Account nickname (e.g., 'work', 'personal'). Uses default if not specified.")
):
    """
    List all folders on the IMAP server.
    
    **Parameters:**
    - account: Account nickname (e.g., 'work', 'personal'). Uses default account if not specified.
    
    **Returns:**
    - List of folder names
    - Total count
    - Account used
    
    **Example folders:**
    - INBOX
    - Sent
    - Drafts
    - Trash
    - Archive
    - Custom/Subfolder
    """
    try:
        config, account_used = _get_imap_config(account)
        
        with IMAPMonitor(config) as monitor:
            folders = monitor.get_folder_list()
        
        return IMAPFolderResponse(
            folders=sorted(folders),
            total=len(folders),
            account=account_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list IMAP folders: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list folders: {str(e)}"
        )


@router.get("/folders/{folder_path:path}/status", response_model=IMAPFolderStatusResponse, dependencies=[Depends(verify_api_key)])
async def get_folder_status(
    folder_path: str,
    account: Optional[str] = Query(None, description="Account nickname (e.g., 'work', 'personal'). Uses default if not specified.")
):
    """
    Get folder status (message counts) using IMAP STATUS command.
    
    This is the most efficient way to get folder statistics without
    fetching any messages.
    
    **Parameters:**
    - folder_path: Folder name (e.g., "INBOX", "Sent", "Archive/2024", "Old Sent Messages/2016")
    - account: Account nickname (e.g., 'work', 'personal'). Uses default if not specified.
    
    **Returns:**
    - total: Total number of messages in folder
    - unseen: Number of unread messages
    - recent: Number of recent messages (new since last check)
    - uidnext: Next UID that will be assigned
    - uidvalidity: UID validity value
    - account: Account used
    
    **Note:** This uses the IMAP STATUS command which is very efficient and
    does not require downloading any message data.
    """
    try:
        config, account_used = _get_imap_config(account)
        
        with IMAPMonitor(config) as monitor:
            status = monitor.get_folder_status(folder_path)
        
        return IMAPFolderStatusResponse(
            folder=folder_path,
            total=status['total'],
            unseen=status['unseen'],
            recent=status['recent'],
            uidnext=status['uidnext'],
            uidvalidity=status['uidvalidity'],
            account=account_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get folder status for {folder_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get folder status: {str(e)}"
        )


@router.get("/folders/{folder_path:path}/messages", response_model=IMAPFolderMessagesResponse, dependencies=[Depends(verify_api_key)])
async def list_folder_messages(
    folder_path: str,
    limit: int = Query(50, ge=1, le=500, description="Maximum number of messages to return"),
    since_date: Optional[str] = Query(None, description="Only messages since this date (YYYY-MM-DD)"),
    include_headers: bool = Query(True, description="Include full headers (subject, from, date)"),
    account: Optional[str] = Query(None, description="Account nickname (e.g., 'work', 'personal'). Uses default if not specified.")
):
    """
    List emails in a specific IMAP folder.
    
    **Parameters:**
    - folder_path: Folder name (e.g., "INBOX", "Sent", "Archive/2024")
    - limit: Maximum messages to return (default: 50, max: 500)
    - since_date: Filter messages since date (YYYY-MM-DD format)
    - include_headers: If true, fetch full headers (slower but more info)
    - account: Account nickname (e.g., 'work', 'personal'). Uses default if not specified.
    
    **Returns:**
    - List of message summaries with UID, subject, from, date, flags
    - Messages are ordered by date (newest first)
    - Account used
    
    **Note:** This fetches directly from IMAP, not from the local database.
    Use this to see what's currently on the server.
    """
    try:
        config, account_used = _get_imap_config(account)
        
        with IMAPMonitor(config) as monitor:
            if include_headers:
                # Fetch messages with full headers
                messages = _fetch_messages_with_headers(monitor, folder_path, limit, since_date)
            else:
                # Just get message IDs (fast)
                message_ids = monitor.get_message_ids_with_headers(
                    folder=folder_path,
                    limit=limit,
                    since_date=since_date
                )
                messages = [
                    IMAPMessageSummary(
                        uid=msg['uid'],
                        message_id=msg.get('message_id')
                    )
                    for msg in message_ids
                ]
        
        return IMAPFolderMessagesResponse(
            folder=folder_path,
            messages=messages,
            total=len(messages),
            account=account_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list messages in folder {folder_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list folder messages: {str(e)}"
        )


def _fetch_messages_with_headers(
    monitor: IMAPMonitor,
    folder: str,
    limit: int,
    since_date: Optional[str]
) -> List[IMAPMessageSummary]:
    """
    Fetch messages from IMAP with full headers parsed.
    
    This fetches envelope/header data without downloading full body content.
    """
    import email
    from email.utils import parseaddr, parsedate_to_datetime
    from datetime import datetime, timezone
    
    if not monitor.client:
        monitor.connect()
    
    try:
        monitor.client.select_folder(folder)
        monitor.current_folder = folder
        
        # Build search criteria
        if since_date:
            date_obj = datetime.strptime(since_date, '%Y-%m-%d')
            utc_date = date_obj.replace(tzinfo=timezone.utc).date()
            search_criteria = [('SINCE', utc_date)]
        else:
            search_criteria = ['ALL']
        
        # Search for messages
        message_ids = monitor.client.search(search_criteria)
        
        # Sort by most recent first (highest UID first)
        message_ids = sorted(message_ids, reverse=True)
        
        # Apply limit
        if limit:
            message_ids = message_ids[:limit]
        
        if not message_ids:
            return []
        
        # Fetch headers and flags for all messages at once (batch for efficiency)
        # ENVELOPE contains: date, subject, from, sender, reply-to, to, cc, bcc, in-reply-to, message-id
        # FLAGS contains: \Seen, \Flagged, etc.
        # BODYSTRUCTURE tells us if there are attachments
        fetch_data = monitor.client.fetch(
            message_ids,
            ['ENVELOPE', 'FLAGS', 'BODYSTRUCTURE']
        )
        
        messages = []
        for msg_id in message_ids:
            if msg_id not in fetch_data:
                continue
            
            data = fetch_data[msg_id]
            envelope = data.get(b'ENVELOPE')
            flags = data.get(b'FLAGS', ())
            body_structure = data.get(b'BODYSTRUCTURE')
            
            if not envelope:
                continue
            
            # Parse envelope data
            # Envelope format: (date, subject, from, sender, reply-to, to, cc, bcc, in-reply-to, message-id)
            msg_date = envelope.date
            subject = envelope.subject
            from_list = envelope.from_
            message_id = envelope.message_id
            
            # Decode subject if bytes
            if isinstance(subject, bytes):
                subject = subject.decode('utf-8', errors='replace')
            if isinstance(message_id, bytes):
                message_id = message_id.decode('utf-8', errors='replace')
            
            # Parse from address
            from_name = None
            from_address = None
            if from_list and len(from_list) > 0:
                from_data = from_list[0]
                # from_data is (name, route, mailbox, host)
                if hasattr(from_data, 'name') and from_data.name:
                    from_name = from_data.name
                    if isinstance(from_name, bytes):
                        from_name = from_name.decode('utf-8', errors='replace')
                if hasattr(from_data, 'mailbox') and hasattr(from_data, 'host'):
                    mailbox = from_data.mailbox
                    host = from_data.host
                    if isinstance(mailbox, bytes):
                        mailbox = mailbox.decode('utf-8', errors='replace')
                    if isinstance(host, bytes):
                        host = host.decode('utf-8', errors='replace')
                    if mailbox and host:
                        from_address = f"{mailbox}@{host}"
            
            # Parse date
            date_str = None
            if msg_date:
                if isinstance(msg_date, datetime):
                    date_str = msg_date.isoformat()
                elif isinstance(msg_date, bytes):
                    date_str = msg_date.decode('utf-8', errors='replace')
                else:
                    date_str = str(msg_date)
            
            # Check flags
            is_seen = b'\\Seen' in flags or '\\Seen' in flags
            is_flagged = b'\\Flagged' in flags or '\\Flagged' in flags
            
            # Check for attachments in body structure
            has_attachments = _has_attachments(body_structure)
            
            messages.append(IMAPMessageSummary(
                uid=str(msg_id),
                message_id=message_id,
                subject=subject,
                from_address=from_address,
                from_name=from_name,
                date=date_str,
                is_seen=is_seen,
                is_flagged=is_flagged,
                has_attachments=has_attachments
            ))
        
        return messages
        
    except Exception as e:
        logger.error(f"Error fetching messages with headers from {folder}: {e}", exc_info=True)
        raise


def _has_attachments(body_structure) -> bool:
    """
    Check if a BODYSTRUCTURE indicates attachments.
    
    Attachments are typically parts with:
    - Content-Disposition: attachment
    - Or non-text/html parts that aren't inline
    """
    if not body_structure:
        return False
    
    # Handle tuple structure from imapclient
    if isinstance(body_structure, (list, tuple)):
        # Multipart message
        if len(body_structure) > 0:
            # Check if it's a multipart (first element is another list/tuple)
            first = body_structure[0]
            if isinstance(first, (list, tuple)):
                # It's multipart - check each part
                for part in body_structure:
                    if isinstance(part, (list, tuple)):
                        if _has_attachments(part):
                            return True
                    elif isinstance(part, bytes):
                        # Check for "attachment" in disposition
                        if b'attachment' in part.lower() if isinstance(part, bytes) else 'attachment' in str(part).lower():
                            return True
            else:
                # Single part - check the content type and disposition
                # Body structure for single part: (type, subtype, params, id, description, encoding, size, ...)
                if len(body_structure) >= 2:
                    content_type = body_structure[0]
                    if isinstance(content_type, bytes):
                        content_type = content_type.decode('utf-8', errors='replace').lower()
                    else:
                        content_type = str(content_type).lower()
                    
                    # Check if it's not text/html (likely an attachment)
                    if content_type not in ('text', b'text'):
                        # Could be image, application, etc. - likely attachment
                        # But we need to check disposition to be sure
                        # For now, just check if there's content-disposition
                        for item in body_structure:
                            if isinstance(item, (list, tuple)):
                                for sub in item:
                                    if isinstance(sub, bytes) and b'attachment' in sub.lower():
                                        return True
                                    elif isinstance(sub, str) and 'attachment' in sub.lower():
                                        return True
    
    return False
