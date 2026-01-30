"""
IMAP client following inbox-zero patterns.
Handles multi-account email fetching with sequential processing.
"""
# Apply Python 3.14 compatibility fix before importing IMAPClient
from backend.core.email.imap_compat import apply_imapclient_python314_fix
apply_imapclient_python314_fix()

from imapclient import IMAPClient
import email
from typing import Iterator, List, Optional, Dict
import logging
from dataclasses import dataclass
import time

from .models import Email

logger = logging.getLogger(__name__)


@dataclass
class IMAPConfig:
    """IMAP connection configuration"""
    host: str
    username: str
    port: int = 993
    use_ssl: bool = True
    folder: str = 'INBOX'
    
    # Authentication type: "password" or "oauth2"
    auth_type: str = "password"
    
    # For password authentication
    password: Optional[str] = None
    
    # For OAuth2 authentication (Microsoft 365 / Outlook)
    oauth2_tenant_id: Optional[str] = None
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None  # For app-only auth
    oauth2_refresh_token: Optional[str] = None  # For delegated auth


class IMAPMonitor:
    """Python version maintaining inbox-zero's approach"""
    
    def __init__(self, 
                 config: IMAPConfig, 
                 timeout: int = 30,
                 safe_move: bool = False):
        """
        Initialize IMAP monitor.
        
        Args:
            config: IMAP connection configuration
            timeout: Network timeout in seconds (default: 30)
            safe_move: Enable post-copy verification (default: False)
        """
        self.config = config
        self.timeout = timeout
        self.safe_move = safe_move
        self.client: Optional[IMAPClient] = None
        self.current_folder: Optional[str] = None
        self._last_activity = time.time()
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5  # Increased for servers with frequent outages
        
    def is_connected(self) -> bool:
        """Check if IMAP connection is alive using NOOP command"""
        if not self.client:
            return False
        try:
            self.client.noop()
            self._last_activity = time.time()
            return True
        except Exception:
            return False
    
    def ensure_connected(self) -> bool:
        """Ensure connection is alive, reconnect if needed"""
        if self.is_connected():
            return True
        
        logger.warning("IMAP connection lost, attempting reconnect...")
        return self.reconnect()
    
    def keep_alive(self) -> bool:
        """
        Send NOOP to keep connection alive during long operations.
        Call this periodically (every 30-60s) during AI processing.
        If NOOP fails, attempts to reconnect automatically.
        """
        if not self.client:
            return self.reconnect()
        try:
            self.client.noop()
            self._last_activity = time.time()
            return True
        except Exception as e:
            logger.warning(f"Keep-alive NOOP failed: {e}, attempting reconnect...")
            return self.reconnect()
    
    def reconnect(self) -> bool:
        """
        Reconnect to IMAP server, reselecting the current folder if any.
        Uses exponential backoff for retries.
        
        Returns:
            True if reconnection succeeded, False otherwise
        """
        saved_folder = self.current_folder
        
        # Close old connection gracefully
        try:
            if self.client:
                self.client.logout()
        except Exception:
            pass
        
        self.client = None
        self.current_folder = None
        
        # Reconnect with retries (connect() handles exponential backoff)
        try:
            self.connect()  # Uses built-in retry logic
            
            # Re-select folder if we had one selected
            if saved_folder:
                self.client.select_folder(saved_folder)
                self.current_folder = saved_folder
                logger.info(f"Reconnected and reselected folder: {saved_folder}")
            else:
                logger.info("Reconnected to IMAP server")
            
            return True
            
        except Exception as e:
            logger.error(f"Reconnect failed after all retries: {e}")
            return False
        
    def connect(self, max_retries: int = None) -> IMAPClient:
        """
        Establish IMAP connection with timeout protection and retry logic.
        
        Args:
            max_retries: Maximum connection attempts (default: self._max_reconnect_attempts)
        
        Returns:
            IMAPClient instance
            
        Raises:
            Exception: If all retry attempts fail
        """
        if max_retries is None:
            max_retries = self._max_reconnect_attempts
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 5s, 10s, 20s, 40s...
                    wait_time = 5 * (2 ** (attempt - 1))
                    logger.info(f"Retry {attempt}/{max_retries-1}: waiting {wait_time}s before reconnecting...")
                    time.sleep(wait_time)
                
                logger.info(f"Connecting to IMAP server {self.config.host}:{self.config.port} (timeout: {self.timeout}s, attempt {attempt+1}/{max_retries})")
                
                client = IMAPClient(
                    host=self.config.host,
                    port=self.config.port,
                    ssl=self.config.use_ssl,
                    timeout=self.timeout  # Prevent network hangs
                )
                
                # Login based on auth type
                if self.config.auth_type == "oauth2":
                    # OAuth2 authentication (Microsoft 365 / Outlook)
                    access_token = self._get_oauth2_token()
                    client.oauth2_login(self.config.username, access_token)
                    logger.info(f"Successfully logged in via OAuth2 as {self.config.username}")
                else:
                    # Standard password authentication
                    client.login(self.config.username, self.config.password)
                    logger.info(f"Successfully logged in as {self.config.username}")
                
                self.client = client
                self._reconnect_attempts = 0  # Reset on successful connect
                return client
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check if this is a retryable error (timeout, network issue)
                is_retryable = any(pattern in error_msg for pattern in [
                    'timed out', 'timeout', 'connection reset', 'connection refused',
                    'broken pipe', 'network', 'temporary', 'unavailable', 'eof'
                ])
                
                if is_retryable and attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt+1} failed (retryable): {e}")
                    continue
                elif not is_retryable:
                    # Non-retryable error (auth failure, etc.) - fail immediately
                    logger.error(f"Connection failed (non-retryable): {e}")
                    raise
                else:
                    # Last attempt failed
                    logger.error(f"All {max_retries} connection attempts failed")
        
        # All retries exhausted
        raise last_error or Exception(f"Failed to connect after {max_retries} attempts")
    
    def _get_oauth2_token(self) -> str:
        """Get OAuth2 access token for IMAP authentication"""
        from backend.core.auth.oauth2_provider import OAuth2Provider, OAuth2Config
        
        if not self.config.oauth2_client_id or not self.config.oauth2_tenant_id:
            raise ValueError(
                "OAuth2 authentication requires oauth2_client_id and oauth2_tenant_id"
            )
        
        if not self.config.oauth2_refresh_token and not self.config.oauth2_client_secret:
            raise ValueError(
                "OAuth2 authentication requires either oauth2_refresh_token (delegated) "
                "or oauth2_client_secret (app-only)"
            )
        
        oauth_config = OAuth2Config(
            tenant_id=self.config.oauth2_tenant_id,
            client_id=self.config.oauth2_client_id,
            client_secret=self.config.oauth2_client_secret,
            refresh_token=self.config.oauth2_refresh_token
        )
        
        provider = OAuth2Provider(oauth_config)
        return provider.get_access_token(self.config.username)
    
    def disconnect(self):
        """Close IMAP connection"""
        if self.client:
            try:
                self.client.logout()
                logger.info("Disconnected from IMAP server")
            except Exception as e:
                logger.warning(f"Error during logout: {e}")
            finally:
                self.client = None
    
    def fetch_new_emails(self, limit: Optional[int] = None) -> Iterator[tuple[str, bytes]]:
        """
        Generator pattern like inbox-zero.
        Yields (uid, raw_email_bytes) tuples.
        
        Args:
            limit: Maximum number of emails to fetch (None for all)
        """
        if not self.client:
            self.connect()
        
        try:
            # Select folder
            self.client.select_folder(self.config.folder)
            self.current_folder = self.config.folder
            logger.info(f"Selected folder: {self.config.folder}")
            
            # Search pattern from inbox-zero (unseen emails)
            messages = self.client.search(['UNSEEN'])
            logger.info(f"Found {len(messages)} unseen emails")
            
            # Apply limit if specified
            if limit:
                messages = messages[:limit]
            
            # Fetch emails sequentially
            for msg_id in messages:
                try:
                    # Fetch RFC822 message without marking as read (PEEK)
                    fetch_data = self.client.fetch([msg_id], ['BODY.PEEK[]', 'FLAGS'])
                    
                    if msg_id not in fetch_data:
                        logger.warning(f"No data returned for message {msg_id}")
                        continue
                    
                    msg_data = fetch_data[msg_id]
                    # BODY.PEEK[] returns the email body with key b'BODY[]'
                    raw_email = msg_data.get(b'BODY[]', msg_data.get(b'RFC822'))
                    
                    # Convert msg_id to string UID
                    uid = str(msg_id)
                    
                    yield (uid, raw_email)
                    
                except Exception as e:
                    logger.error(f"Error fetching message {msg_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error during email fetch: {e}")
            raise
    
    def get_message_ids_with_headers(self, folder: str = 'INBOX', limit: Optional[int] = None, 
                                     since_date: Optional[str] = None, since_uid: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Efficiently fetch all message UIDs with their Message-ID headers (without full email body).
        This is MUCH faster than fetching full emails when you just need to check which ones to process.
        
        Args:
            folder: IMAP folder name
            limit: Maximum number of emails to fetch
            since_date: Only fetch emails since this date (YYYY-MM-DD format)
            since_uid: Only fetch emails with UID greater than this (for incremental sync)
            
        Returns:
            List of dicts: [{'uid': '123', 'message_id': '<abc@example.com>'}, ...]
        """
        if not self.client:
            self.connect()
        
        try:
            self.client.select_folder(folder)
            self.current_folder = folder
            logger.info(f"Selected folder: {folder}")
            
            # Build search criteria
            if since_uid:
                search_criteria = ['ALL']
                logger.info(f"Searching for emails with UID > {since_uid} (will filter after search)")
            elif since_date:
                # Parse date string and ensure UTC timezone for consistent behavior
                from datetime import datetime, timezone
                date_obj = datetime.strptime(since_date, '%Y-%m-%d')
                # Convert to UTC date (IMAP SINCE uses date part only, but we ensure UTC interpretation)
                utc_date = date_obj.replace(tzinfo=timezone.utc).date()
                search_criteria = [('SINCE', utc_date)]
                logger.info(f"Searching for emails since {utc_date} (UTC)")
            else:
                search_criteria = ['ALL']
            
            # Search for messages
            messages = self.client.search(search_criteria)
            logger.info(f"Found {len(messages)} total emails in {folder}")
            
            # Filter by UID if since_uid is specified
            if since_uid:
                messages = [msg_id for msg_id in messages if msg_id > since_uid]
                logger.info(f"Filtered to {len(messages)} emails with UID > {since_uid}")
            
            # Sort by most recent first (reverse the message IDs)
            messages = list(reversed(messages))
            
            # Apply limit if specified
            if limit:
                messages = messages[:limit]
                logger.info(f"Limited to {limit} most recent emails")
            
            # Fetch ONLY the Message-ID header (super fast, no body content)
            # Batch in chunks to avoid overwhelming the connection
            result = []
            if messages:
                batch_size = 500  # Fetch 500 at a time to avoid connection issues
                for i in range(0, len(messages), batch_size):
                    batch = messages[i:i + batch_size]
                    
                    try:
                        # Batch fetch for efficiency
                        fetch_data = self.client.fetch(batch, ['BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)]'])
                        
                        for msg_id in batch:
                            if msg_id not in fetch_data:
                                continue
                            
                            # Extract Message-ID from header
                            raw_data = fetch_data[msg_id]
                            header_data = raw_data.get(b'BODY[HEADER.FIELDS (MESSAGE-ID)]', b'')
                            
                            if header_data:
                                header_str = header_data.decode('utf-8', errors='ignore').strip()
                                # Parse Message-ID: <value>
                                if ':' in header_str:
                                    parts = header_str.split(':', 1)
                                    if len(parts) == 2:
                                        # Keep angle brackets for consistency with database storage (RFC 5322 format)
                                        message_id = parts[1].strip()
                                        if message_id:
                                            result.append({
                                                'uid': str(msg_id),
                                                'message_id': message_id
                                            })
                        
                        if (i + batch_size) < len(messages):
                            logger.debug(f"Fetched {len(result)} of {len(messages)} message IDs...")
                    
                    except Exception as e:
                        logger.error(f"Error fetching batch {i}-{i+batch_size}: {e}")
                        # Continue with next batch
                        continue
            
            logger.info(f"Retrieved {len(result)} message IDs (with headers only)")
            return result
                    
        except Exception as e:
            logger.error(f"Error fetching message IDs: {e}")
            raise
    
    def fetch_all_emails(self, folder: str = 'INBOX', limit: Optional[int] = None, since_date: Optional[str] = None, since_uid: Optional[int] = None) -> Iterator[tuple[str, bytes]]:
        """
        Fetch all emails from a folder (not just unseen).
        
        Args:
            folder: IMAP folder name
            limit: Maximum number of emails to fetch
            since_date: Only fetch emails since this date (YYYY-MM-DD format)
            since_uid: Only fetch emails with UID greater than this (for incremental sync)
        """
        if not self.client:
            self.connect()
        
        try:
            self.client.select_folder(folder)
            self.current_folder = folder
            logger.info(f"Selected folder: {folder}")
            
            # Build search criteria
            if since_uid:
                # Fetch only UIDs greater than since_uid (incremental sync)
                # Get all messages first, then filter by UID (IMAPClient doesn't support UID range in search)
                search_criteria = ['ALL']
                logger.info(f"Searching for emails with UID > {since_uid} (will filter after search)")
            elif since_date:
                # IMAPClient accepts datetime objects for SINCE
                # Parse date string and ensure UTC timezone for consistent behavior
                from datetime import datetime, timezone
                date_obj = datetime.strptime(since_date, '%Y-%m-%d')
                # Convert to UTC date (IMAP SINCE uses date part only, but we ensure UTC interpretation)
                # RFC 3501: SINCE date is interpreted in server's timezone, but we normalize to UTC
                utc_date = date_obj.replace(tzinfo=timezone.utc).date()
                search_criteria = [('SINCE', utc_date)]
                logger.info(f"Searching for emails since {utc_date} (UTC)")
            else:
                search_criteria = ['ALL']
            
            # Search for messages
            messages = self.client.search(search_criteria)
            logger.info(f"Found {len(messages)} total emails in {folder}")
            
            # Filter by UID if since_uid is specified
            if since_uid:
                messages = [msg_id for msg_id in messages if msg_id > since_uid]
                logger.info(f"Filtered to {len(messages)} emails with UID > {since_uid}")
            
            # Sort by most recent first (reverse the message IDs)
            messages = list(reversed(messages))
            
            # Apply limit if specified
            if limit:
                messages = messages[:limit]  # Get first N (which are most recent)
                logger.info(f"Limited to {limit} most recent emails")
            
            for msg_id in messages:
                try:
                    # Use BODY.PEEK[] to avoid marking emails as read
                    fetch_data = self.client.fetch([msg_id], ['BODY.PEEK[]'])
                    self._last_activity = time.time()
                    
                    if msg_id not in fetch_data:
                        continue
                    
                    # BODY.PEEK[] returns the email body with key b'BODY[]'
                    raw_email = fetch_data[msg_id].get(b'BODY[]', fetch_data[msg_id].get(b'RFC822'))
                    uid = str(msg_id)
                    
                    yield (uid, raw_email)
                    
                except Exception as e:
                    # Check if this is a timeout/connection error that we can recover from
                    error_msg = str(e).lower()
                    is_timeout_error = any(pattern in error_msg for pattern in [
                        'timed out', 'timeout', 'cannot read', 'connection reset',
                        'broken pipe', 'connection refused', 'eof', 'disconnected'
                    ])
                    
                    if is_timeout_error:
                        logger.warning(f"Connection error fetching UID {msg_id}: {e}")
                        logger.info("Attempting reconnect...")
                        if self.reconnect():
                            # Retry this message after reconnect
                            try:
                                fetch_data = self.client.fetch([msg_id], ['BODY.PEEK[]'])
                                self._last_activity = time.time()
                                if msg_id in fetch_data:
                                    raw_email = fetch_data[msg_id].get(b'BODY[]', fetch_data[msg_id].get(b'RFC822'))
                                    yield (str(msg_id), raw_email)
                                logger.info(f"Successfully fetched UID {msg_id} after reconnect")
                                continue
                            except Exception as retry_error:
                                logger.error(f"Retry failed for UID {msg_id}: {retry_error}")
                        else:
                            logger.error(f"Reconnect failed after 3 attempts, skipping remaining emails")
                            return
                    else:
                        logger.error(f"Error fetching UID {msg_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error during email fetch: {e}")
            raise
    
    def mark_as_read(self, uid: str) -> bool:
        """Mark email as read"""
        if not self.client:
            raise RuntimeError("Not connected to IMAP server")
        
        try:
            msg_id = int(uid)
            self.client.add_flags([msg_id], ['\\Seen'])
            logger.debug(f"Marked message {uid} as read")
            return True
        except Exception as e:
            logger.error(f"Failed to mark message {uid} as read: {e}")
            return False
    
    def create_folder(self, folder: str) -> bool:
        """
        Create a folder if it doesn't exist.
        Handles nested folders (e.g., "Notifications/Sentry").
        
        Args:
            folder: Folder name (can include / for nested)
            
        Returns:
            True if folder exists or was created
        """
        if not self.client:
            raise RuntimeError("Not connected to IMAP server")
        
        try:
            # Check if folder exists
            existing_folders = [f[2] for f in self.client.list_folders()]
            
            if folder in existing_folders:
                return True  # Already exists
            
            # Create folder (imapclient handles nested folders)
            self.client.create_folder(folder)
            logger.info(f"Created folder: {folder}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create folder {folder}: {e}")
            return False
    
    def move_to_folder(self, uid: str, folder: str, create_if_missing: bool = False) -> bool:
        """
        Move email to another folder with optional post-copy verification.
        
        Args:
            uid: Email UID to move
            folder: Destination folder path
            create_if_missing: If True, create folder if it doesn't exist (default: False for safety)
            
        Returns:
            True if move succeeded (or verified if safe_move enabled)
        """
        if not self.client:
            raise RuntimeError("Not connected to IMAP server")
        
        # Defensive: Ensure current_folder is set (should always be set during normal workflow)
        if not self.current_folder:
            logger.warning(f"current_folder is None before move - selecting default folder {self.config.folder}")
            self.client.select_folder(self.config.folder)
            self.current_folder = self.config.folder
        
        try:
            # Check if folder exists
            existing_folders = [f[2] for f in self.client.list_folders()]
            if folder not in existing_folders:
                if create_if_missing:
                    # Create folder if allowed
                    logger.info(f"Creating folder '{folder}' (create_if_missing=True)")
                    self.create_folder(folder)
                else:
                    # Safety: do not auto-create
                    logger.error(f"Cannot move UID {uid} to '{folder}' - folder does not exist (safety: auto-create disabled)")
                    return False
            
            msg_id = int(uid)
            
            # Fetch Message-ID for verification (if safe_move enabled)
            message_id_header = None
            if self.safe_move:
                try:
                    fetch_data = self.client.fetch([msg_id], ['BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)]'])
                    
                    if msg_id in fetch_data:
                        # Get the header data
                        # Note: Server returns 'BODY[...]' even if we request 'BODY.PEEK[...]'
                        raw_data = fetch_data[msg_id]
                        header_data = raw_data.get(b'BODY[HEADER.FIELDS (MESSAGE-ID)]', b'')
                        
                        if not header_data:
                            logger.warning(f"Header data is empty for UID {uid}. Available keys: {list(raw_data.keys())}")
                            message_id_header = None
                        else:
                            # Parse Message-ID from header
                            # Format can be: "Message-ID: <id>" or "Message-Id: <id>" (case varies)
                            header_str = header_data.decode('utf-8', errors='ignore').strip()
                            
                            # Case-insensitive search for Message-ID header
                            if header_str and ':' in header_str:
                                # Split on first colon to get the value
                                parts = header_str.split(':', 1)
                                if len(parts) == 2:
                                    message_id_header = parts[1].strip()
                                    # Remove angle brackets and whitespace
                                    message_id_header = message_id_header.strip('<> \r\n')
                                    if message_id_header:
                                        logger.debug(f"✓ Fetched Message-ID for verification: {message_id_header}")
                                    else:
                                        logger.warning(f"Message-ID header exists but value is empty for UID {uid}")
                                        message_id_header = None
                                else:
                                    logger.warning(f"Could not parse Message-ID (no colon found) for UID {uid}")
                                    message_id_header = None
                            else:
                                logger.warning(f"Message-ID header not found in email UID {uid} (header_str: {repr(header_str[:50])})")
                                message_id_header = None
                    else:
                        logger.warning(f"UID {uid} not in fetch results")
                except Exception as e:
                    logger.warning(f"Error fetching Message-ID for UID {uid}: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    # Continue without verification if Message-ID unavailable
                    message_id_header = None
            
            # Store original folder before copy
            original_folder = self.current_folder
            
            # Copy to destination folder
            logger.debug(f"Copying UID {uid} to {folder}")
            self.client.copy([msg_id], folder)
            
            # Verify copy if safe_move enabled and Message-ID available
            if self.safe_move and message_id_header:
                logger.debug(f"Verifying copy for UID {uid} in {folder}...")
                verified = False
                
                for attempt in range(3):
                    try:
                        # Switch to target folder
                        self.client.select_folder(folder)
                        self.current_folder = folder
                        
                        # Search by Message-ID (IMAP search syntax: ['HEADER', 'MESSAGE-ID', 'value'])
                        # Note: Don't use quotes around the Message-ID value in the list syntax
                        results = self.client.search(['HEADER', 'MESSAGE-ID', message_id_header])
                        
                        if results and len(results) > 0:
                            verified = True
                            logger.debug(f"✓ Copy verified for UID {uid} in {folder} (attempt {attempt+1})")
                            break
                        else:
                            logger.warning(f"Verification attempt {attempt+1}/3 failed - email not found in {folder}")
                    
                    except Exception as verify_error:
                        logger.warning(f"Verification attempt {attempt+1}/3 error: {verify_error}")
                    
                    # Wait before retry (unless last attempt)
                    if attempt < 2:
                        time.sleep(2)
                
                # Switch back to original folder
                if original_folder:
                    self.client.select_folder(original_folder)
                    self.current_folder = original_folder
                
                if not verified:
                    logger.error(f"❌ Copy verification FAILED for UID {uid} after 3 attempts - ABORTING delete to prevent loss")
                    return False
                    
            elif self.safe_move:
                logger.warning(f"Safe-move enabled but Message-ID unavailable for UID {uid} - proceeding without verification")
            
            # Only proceed with delete if copy succeeded (and verified if safe_move)
            logger.debug(f"Flagging UID {uid} as deleted in {original_folder or self.current_folder}")
            self.client.add_flags([msg_id], ['\\Deleted'])
            
            # Expunge to actually remove from source
            logger.debug(f"Expunging deleted messages from {original_folder or self.current_folder}")
            self.client.expunge()
            
            # Only log at debug level - success will be shown in card output
            logger.debug(f"✓ Moved message {uid} to {folder}")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            is_timeout_error = any(pattern in error_msg for pattern in [
                'timed out', 'timeout', 'cannot read', 'connection reset',
                'broken pipe', 'connection refused', 'eof', 'disconnected'
            ])
            
            if is_timeout_error:
                logger.warning(f"Connection error during move of UID {uid}: {e}")
                logger.info("Attempting reconnect for future operations...")
                # Reconnect so subsequent operations can work
                # Don't retry the move itself (email might be in inconsistent state)
                self.reconnect()
            
            logger.error(f"Failed to move message {uid} to {folder}: {e}")
            return False
    
    def apply_color_label(self, uid: str, color: int) -> bool:
        """
        Apply flag to email (color intent tracked but not applied to IMAP).
        
        IMAP servers have limited support for color keywords:
        - Apple Mail's $MailFlagBit* keywords don't work on most servers
        - Custom keywords often not supported
        
        Strategy:
        1. Set standard \\Flagged flag (shows ⭐ in Mail.app)
        2. Color/priority stored in database (Phase 2)
        3. Web UI will display colors based on database metadata
        
        Args:
            uid: Email UID
            color: Color code (1-7) - recorded but not applied to IMAP
        """
        if not self.client:
            raise RuntimeError("Not connected to IMAP server")
        
        try:
            msg_id = int(uid)
            
            # Validate color code first
            color_names = {
                1: 'Red (Urgent)',
                2: 'Orange (Important)', 
                3: 'Yellow (Review)',
                4: 'Green (Waiting)',
                5: 'Blue (Action)',
                6: 'Purple (Later)',
                7: 'Gray (Info)'
            }
            
            if color not in color_names:
                logger.warning(f"Invalid color code: {color}")
                return False
            
            # Set \Flagged for all important emails
            # Color information will be stored in database (Phase 2)
            self.client.add_flags([msg_id], ['\\Flagged'])
            
            color_name = color_names[color]
            logger.info(f"Flagged message {uid} (intended: {color_name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to flag message {uid}: {e}")
            return False
    
    def save_draft(self,
                   to_address: str,
                   subject: str,
                   body: str,
                   original_message_id: Optional[str] = None,
                   original_references: Optional[str] = None,
                   drafts_folder: str = 'Drafts') -> bool:
        """
        Save an email as a draft in the Drafts folder.
        
        Args:
            to_address: Recipient email address
            subject: Email subject
            body: Email body
            original_message_id: Message-ID of original email (for threading)
            original_references: Full References header from original email (for threading chains)
            drafts_folder: Name of drafts folder (default: 'Drafts')
            
        Returns:
            True if draft saved successfully
        """
        if not self.client:
            raise RuntimeError("Not connected to IMAP server")
        
        try:
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from email.utils import make_msgid, formatdate
            
            # Create draft message
            msg = MIMEMultipart('alternative')
            msg['To'] = to_address
            msg['Subject'] = subject
            msg['Date'] = formatdate(localtime=True)
            msg['Message-ID'] = make_msgid()
            
            # Add threading headers if replying (properly construct References chain)
            if original_message_id:
                msg['In-Reply-To'] = original_message_id
                
                # Build full References chain: original References + original Message-ID
                if original_references:
                    # Append original message ID to existing chain
                    msg['References'] = f"{original_references} {original_message_id}"
                else:
                    # Start new chain
                    msg['References'] = original_message_id
            
            # Add body
            if body.startswith('<html>') or '<' in body[:100]:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Convert to bytes
            draft_bytes = msg.as_bytes()
            
            # Append to Drafts folder with \Draft flag
            self.client.append(drafts_folder, draft_bytes, flags=['\\Draft'])
            
            logger.info(f"Saved draft to {drafts_folder}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save draft: {e}")
            return False
    
    def add_custom_flag(self, uid: str, flag_name: str) -> bool:
        """
        Add a custom IMAP keyword/flag to an email (for notes/annotations).
        
        NOTE: Most IMAP servers (including yours) don't support arbitrary custom keywords.
        This method will attempt to set the keyword, but if it fails, it will just
        set \\Flagged instead. The actual label will be stored in database (Phase 2).
        
        Args:
            uid: Email UID
            flag_name: Custom flag name (e.g., 'Important', 'FollowUp')
            
        Returns:
            True if flag added successfully (or \\Flagged set as fallback)
        """
        if not self.client:
            raise RuntimeError("Not connected to IMAP server")
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                msg_id = int(uid)
                
                # Try to set custom flag (will likely fail on most servers)
                if not flag_name.startswith('$'):
                    flag_name = f'${flag_name}'
                
                try:
                    self.client.add_flags([msg_id], [flag_name])
                    logger.info(f"Added custom flag '{flag_name}' to message {uid}")
                except Exception as keyword_error:
                    # Server doesn't support custom keywords
                    # Silently fall back to \Flagged (stored in database in Phase 2)
                    self.client.add_flags([msg_id], ['\\Flagged'])
                    logger.debug(f"Custom flag '{flag_name}' not supported, using \\Flagged (will store '{flag_name}' in database)")
                
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                is_timeout_error = any(pattern in error_msg for pattern in [
                    'timed out', 'timeout', 'cannot read', 'connection reset'
                ])
                
                if is_timeout_error and attempt < max_retries - 1:
                    logger.warning(f"Connection error adding flag to {uid}, attempting reconnect...")
                    if self.reconnect():
                        continue
                
                logger.error(f"Failed to add flag to {uid}: {e}")
                return False
        
        return False
    
    def get_folder_list(self) -> List[str]:
        """Get list of all folders"""
        if not self.client:
            self.connect()
        
        try:
            folders = self.client.list_folders()
            return [folder[2] for folder in folders]
        except Exception as e:
            logger.error(f"Failed to get folder list: {e}")
            return []
    
    def get_folder_status(self, folder: str) -> Dict[str, int]:
        """
        Get folder status (message counts) using IMAP STATUS command.
        
        This is the most efficient way to get folder statistics without
        fetching any messages.
        
        Args:
            folder: Folder name (e.g., "INBOX", "Sent", "Archive/2024")
            
        Returns:
            Dict with:
            - total: Total number of messages in folder
            - unseen: Number of unread messages
            - recent: Number of recent messages (new since last check)
            - uidnext: Next UID that will be assigned
            - uidvalidity: UID validity value
        """
        if not self.client:
            self.connect()
        
        try:
            # Use IMAP STATUS command - very efficient, doesn't require SELECT
            # Request: MESSAGES (total), UNSEEN (unread), RECENT, UIDNEXT, UIDVALIDITY
            status = self.client.folder_status(
                folder,
                ['MESSAGES', 'UNSEEN', 'RECENT', 'UIDNEXT', 'UIDVALIDITY']
            )
            
            # imapclient returns a dict with bytes keys
            result = {
                'total': status.get(b'MESSAGES', 0),
                'unseen': status.get(b'UNSEEN', 0),
                'recent': status.get(b'RECENT', 0),
                'uidnext': status.get(b'UIDNEXT', 0),
                'uidvalidity': status.get(b'UIDVALIDITY', 0)
            }
            
            logger.debug(f"Folder status for '{folder}': {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get status for folder '{folder}': {e}")
            raise
    
    def __enter__(self):
        """Context manager support"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.disconnect()

