"""
Attachment Extractor - Extract attachments from IMAP emails

Handles IMAP connection, retry logic, and MIME parsing to extract binary attachments.
Includes resilient email lookup with Message-ID fallback for moved emails.
"""
import email
import email.header
import time
import logging
import re
from typing import Tuple, Optional, List, Callable
from email.message import Message
from contextlib import contextmanager
from imapclient import IMAPClient

from backend.core.accounts.manager import AccountManager
from backend.core.email.imap_monitor import IMAPConfig
from backend.core.email.attachment_detection import is_attachment_part

logger = logging.getLogger(__name__)

# Priority folders to search when email is not found in expected location
# Order matters: INBOX and Archive are most likely, others are fallbacks
PRIORITY_FOLDERS = [
    'INBOX',
    'Archive',
]

# Additional common folders to check before searching all folders
# Include variations for different mail providers
COMMON_FOLDERS = [
    # Sent folders
    'Sent',           # Personal/standard
    'Sent Items',     # Work/Exchange
    'Sent Messages',  # Apple Mail
    # Deleted/Trash folders
    'Trash',          # Personal/standard
    'Deleted Items',  # Work/Exchange
    # Other common folders
    'Drafts',
    'Junk',
    'Spam',
    # Gmail-specific
    '[Gmail]/All Mail',
    '[Gmail]/Sent Mail',
    '[Gmail]/Trash',
]


class AttachmentNotFoundError(Exception):
    """Attachment not found on IMAP server"""
    pass


class AttachmentIndexError(Exception):
    """Attachment index out of range"""
    pass


class EmailNotFoundError(Exception):
    """Email not found on IMAP server"""
    pass


class FolderNotFoundError(Exception):
    """Folder not found on IMAP server"""
    pass


class IMAPConnectionError(Exception):
    """IMAP connection failed after retries"""
    pass


class IMAPError(Exception):
    """General IMAP error"""
    pass


class AttachmentExtractor:
    """Extract attachments from IMAP emails with retry logic"""
    
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2  # Exponential backoff: 2, 4, 8 seconds
    
    def __init__(self, account_manager: AccountManager):
        """
        Initialize attachment extractor.
        
        Args:
            account_manager: AccountManager instance for getting IMAP configs
        """
        self.account_manager = account_manager
    
    @contextmanager
    def _get_imap_connection(self, config: IMAPConfig, timeout: int = 120):
        """
        Create IMAP connection context manager.
        
        Args:
            config: IMAP configuration
            timeout: Connection timeout in seconds
            
        Yields:
            IMAPClient connection
        """
        client = None
        try:
            client = IMAPClient(
                host=config.host,
                port=config.port,
                ssl=config.use_ssl,
                timeout=timeout
            )
            client.login(config.username, config.password)
            logger.debug(f"Connected to IMAP: {config.host}")
            yield client
        finally:
            if client:
                try:
                    client.logout()
                except Exception as e:
                    logger.warning(f"Error during IMAP logout: {e}")
    
    def get_attachment(
        self, 
        account_id: str,
        folder: str, 
        uid: str, 
        attachment_index: int,
        timeout: int = 120,
        message_id: Optional[str] = None,
        on_location_update: Optional[Callable[[str, str], None]] = None
    ) -> Tuple[bytes, str, str]:
        """
        Fetch specific attachment from IMAP with retry logic and fallback lookup.
        
        If email is not found at the expected folder/UID, attempts to find it by
        Message-ID header across common folders.
        
        Args:
            account_id: Account nickname (e.g., 'work', 'personal')
            folder: IMAP folder name (expected location)
            uid: Email UID (string, expected)
            attachment_index: 0-based index of attachment
            timeout: Connection timeout in seconds
            message_id: Optional Message-ID header for fallback lookup
            on_location_update: Optional callback(new_folder, new_uid) called when
                               email is found in a different location
            
        Returns:
            Tuple of (binary_content, filename, content_type)
            
        Raises:
            AttachmentNotFoundError: Attachment not found
            AttachmentIndexError: Index out of range
            EmailNotFoundError: Email not found on IMAP
            FolderNotFoundError: Folder not found
            IMAPConnectionError: Connection failed after retries
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                # Get IMAP config for account
                config = self.account_manager.get_imap_config(account_id)
                
                # Connect to IMAP with timeout
                with self._get_imap_connection(config, timeout) as conn:
                    # Try to find email - first at expected location, then by Message-ID
                    actual_folder, actual_uid, raw_email = self._find_email(
                        conn, folder, uid, message_id
                    )
                    
                    # If found in different location, call update callback
                    if (actual_folder != folder or str(actual_uid) != str(uid)) and on_location_update:
                        logger.info(
                            f"Email moved: {folder}/{uid} -> {actual_folder}/{actual_uid}"
                        )
                        on_location_update(actual_folder, str(actual_uid))
                    
                    # Parse MIME
                    msg = email.message_from_bytes(raw_email)
                    
                    # Extract attachment at index
                    return self._extract_attachment(msg, attachment_index)
                    
            except (ConnectionError, TimeoutError, OSError) as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY_BASE ** (attempt + 1)
                    logger.warning(
                        f"IMAP connection failed, retry {attempt+1}/{self.MAX_RETRIES} in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    raise IMAPConnectionError(
                        f"IMAP connection failed after {self.MAX_RETRIES} attempts: {e}"
                    )
            except (AttachmentNotFoundError, AttachmentIndexError, EmailNotFoundError, 
                    FolderNotFoundError, IMAPError) as e:
                # These errors shouldn't be retried
                raise
            except Exception as e:
                # Unexpected errors - log and re-raise
                logger.error(f"Unexpected error in get_attachment (attempt {attempt+1}): {e}", exc_info=True)
                if attempt == self.MAX_RETRIES - 1:
                    raise IMAPConnectionError(f"Failed after {self.MAX_RETRIES} attempts: {e}")
                # Otherwise retry
                delay = self.RETRY_DELAY_BASE ** (attempt + 1)
                time.sleep(delay)

    def get_all_attachments(
        self,
        account_id: str,
        folder: str,
        uid: str,
        timeout: int = 120,
        message_id: Optional[str] = None,
        on_location_update: Optional[Callable[[str, str], None]] = None
    ) -> List[Tuple[bytes, str, str]]:
        """
        Fetch all attachments from an email on IMAP with retry logic.

        Similar to get_attachment() but returns ALL attachments at once,
        avoiding multiple IMAP round-trips for emails with multiple attachments.

        Args:
            account_id: Account nickname (e.g., 'work', 'personal')
            folder: IMAP folder name (expected location)
            uid: Email UID (string, expected)
            timeout: Connection timeout in seconds
            message_id: Optional Message-ID header for fallback lookup
            on_location_update: Optional callback(new_folder, new_uid) called when
                               email is found in a different location

        Returns:
            List of (binary_content, filename, content_type) tuples for each attachment

        Raises:
            EmailNotFoundError: Email not found on IMAP
            FolderNotFoundError: Folder not found
            IMAPConnectionError: Connection failed after retries
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                # Get IMAP config for account
                config = self.account_manager.get_imap_config(account_id)

                # Connect to IMAP with timeout
                with self._get_imap_connection(config, timeout) as conn:
                    # Try to find email - first at expected location, then by Message-ID
                    actual_folder, actual_uid, raw_email = self._find_email(
                        conn, folder, uid, message_id
                    )

                    # If found in different location, call update callback
                    if (actual_folder != folder or str(actual_uid) != str(uid)) and on_location_update:
                        logger.info(
                            f"Email moved: {folder}/{uid} -> {actual_folder}/{actual_uid}"
                        )
                        on_location_update(actual_folder, str(actual_uid))

                    # Parse MIME
                    msg = email.message_from_bytes(raw_email)

                    # Extract all attachments
                    return self._extract_all_attachments(msg)

            except (ConnectionError, TimeoutError, OSError) as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY_BASE ** (attempt + 1)
                    logger.warning(
                        f"IMAP connection failed, retry {attempt+1}/{self.MAX_RETRIES} in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    raise IMAPConnectionError(
                        f"IMAP connection failed after {self.MAX_RETRIES} attempts: {e}"
                    )
            except (EmailNotFoundError, FolderNotFoundError, IMAPError) as e:
                # These errors shouldn't be retried
                raise
            except Exception as e:
                # Unexpected errors - log and re-raise
                logger.error(f"Unexpected error in get_all_attachments (attempt {attempt+1}): {e}", exc_info=True)
                if attempt == self.MAX_RETRIES - 1:
                    raise IMAPConnectionError(f"Failed after {self.MAX_RETRIES} attempts: {e}")
                # Otherwise retry
                delay = self.RETRY_DELAY_BASE ** (attempt + 1)
                time.sleep(delay)

        # Should never reach here
        return []

    def _extract_all_attachments(self, msg: Message) -> List[Tuple[bytes, str, str]]:
        """
        Extract all attachments from MIME message.

        Args:
            msg: Parsed email message

        Returns:
            List of (binary_content, filename, content_type) tuples
        """
        results = []

        # Use shared attachment detection logic
        attachments = []
        for part in msg.walk():
            if is_attachment_part(part):
                attachments.append(part)

        for idx, part in enumerate(attachments):
            try:
                # Extract and decode filename
                filename = self._decode_filename(part, idx)

                # Get content type
                content_type = part.get_content_type() or 'application/octet-stream'

                # Decode payload
                content = part.get_payload(decode=True)
                if content is None:
                    logger.warning(f"Could not decode attachment {idx}: {filename}")
                    continue

                results.append((content, filename, content_type))

            except Exception as e:
                logger.warning(f"Failed to extract attachment {idx}: {e}")
                continue

        return results

    def _find_email(
        self,
        conn: IMAPClient,
        expected_folder: str,
        expected_uid: str,
        message_id: Optional[str]
    ) -> Tuple[str, int, bytes]:
        """
        Find email on IMAP, with fallback to Message-ID search if not at expected location.
        
        Args:
            conn: Active IMAP connection
            expected_folder: Expected folder name
            expected_uid: Expected UID
            message_id: Optional Message-ID for fallback search
            
        Returns:
            Tuple of (actual_folder, actual_uid, raw_email_bytes)
            
        Raises:
            EmailNotFoundError: Email not found anywhere
            FolderNotFoundError: Expected folder not found
        """
        # Parse expected UID
        try:
            uid_int = int(expected_uid)
        except ValueError:
            raise EmailNotFoundError(f"Invalid UID format: {expected_uid}")
        
        # Step 1: Try expected folder + UID
        try:
            conn.select_folder(expected_folder, readonly=True)
            raw_email = self._fetch_by_uid(conn, uid_int)
            if raw_email:
                return expected_folder, uid_int, raw_email
            logger.debug(f"Email UID {expected_uid} not in {expected_folder}")
        except Exception as e:
            logger.debug(f"Could not check {expected_folder}: {e}")
        
        # Step 2: If we have message_id, search by it
        if message_id:
            logger.info(f"Searching for email by Message-ID: {message_id[:50]}...")
            
            # Build ordered list of folders to search:
            # 1. Expected folder (already tried above, but include for completeness)
            # 2. Priority folders (INBOX, Archive - most likely locations)
            # 3. Common folders (Sent, Drafts, etc.)
            folders_to_search = []
            searched = set()
            
            # Add expected folder first
            if expected_folder not in searched:
                folders_to_search.append(expected_folder)
                searched.add(expected_folder)
            
            # Add priority folders (INBOX, Archive)
            for f in PRIORITY_FOLDERS:
                if f not in searched:
                    folders_to_search.append(f)
                    searched.add(f)
            
            # Add common folders
            for f in COMMON_FOLDERS:
                if f not in searched:
                    folders_to_search.append(f)
                    searched.add(f)
            
            # Search priority and common folders first
            for folder in folders_to_search:
                try:
                    result = self._search_by_message_id(conn, folder, message_id)
                    if result:
                        found_uid, raw_email = result
                        logger.info(f"Found email by Message-ID in {folder} with UID {found_uid}")
                        return folder, found_uid, raw_email
                except Exception as e:
                    logger.debug(f"Could not search {folder}: {e}")
                    continue
            
            # If not found in priority/common folders, search ALL folders
            logger.info("Not found in common folders, searching all folders...")
            try:
                all_folders = conn.list_folders()
                
                for flags, delimiter, folder_name in all_folders:
                    if folder_name in searched:
                        continue  # Already searched
                    
                    try:
                        result = self._search_by_message_id(conn, folder_name, message_id)
                        if result:
                            found_uid, raw_email = result
                            logger.info(f"Found email by Message-ID in {folder_name} with UID {found_uid}")
                            return folder_name, found_uid, raw_email
                    except Exception as e:
                        logger.debug(f"Could not search {folder_name}: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Could not list all folders: {e}")
        
        # Not found anywhere
        raise EmailNotFoundError(
            f"Email not found on IMAP server. "
            f"Expected: {expected_folder}/{expected_uid}, "
            f"Message-ID: {message_id[:50] if message_id else 'N/A'}..."
        )
    
    def _fetch_by_uid(self, conn: IMAPClient, uid: int) -> Optional[bytes]:
        """
        Fetch email by UID from currently selected folder.
        
        Returns raw email bytes or None if not found.
        """
        try:
            # First check if UID exists
            flags_data = conn.fetch([uid], ['FLAGS'])
            if uid not in flags_data:
                return None
            
            # Fetch full message
            fetch_data = conn.fetch([uid], ['BODY.PEEK[]'])
            if uid not in fetch_data:
                return None
            
            msg_data = fetch_data[uid]
            raw_email = msg_data.get(b'BODY[]', msg_data.get(b'RFC822'))
            return raw_email if raw_email else None
            
        except Exception as e:
            logger.debug(f"Error fetching UID {uid}: {e}")
            return None
    
    def _search_by_message_id(
        self, 
        conn: IMAPClient, 
        folder: str, 
        message_id: str
    ) -> Optional[Tuple[int, bytes]]:
        """
        Search for email by Message-ID header in a folder.
        
        Args:
            conn: Active IMAP connection
            folder: Folder to search
            message_id: Message-ID header value
            
        Returns:
            Tuple of (uid, raw_email_bytes) or None if not found
        """
        try:
            conn.select_folder(folder, readonly=True)
        except Exception:
            return None
        
        # Search by Message-ID header
        # Some servers need the angle brackets, some don't - try both
        search_ids = [message_id]
        if message_id.startswith('<') and message_id.endswith('>'):
            search_ids.append(message_id[1:-1])  # Without brackets
        else:
            search_ids.append(f'<{message_id}>')  # With brackets
        
        for search_id in search_ids:
            try:
                result = conn.search(['HEADER', 'Message-ID', search_id])
                if result:
                    # Found! Fetch the first match
                    uid = result[0]
                    raw_email = self._fetch_by_uid(conn, uid)
                    if raw_email:
                        return uid, raw_email
            except Exception as e:
                logger.debug(f"Search error for {search_id}: {e}")
                continue
        
        return None
    
    def _extract_attachment(self, msg: Message, index: int) -> Tuple[bytes, str, str]:
        """
        Extract specific attachment from MIME message.
        
        Args:
            msg: Parsed email message
            index: 0-based index of attachment
            
        Returns:
            Tuple of (binary_content, filename, content_type)
            
        Raises:
            AttachmentIndexError: Index out of range
        """
        # Use shared attachment detection logic to ensure consistency
        # between indexing (processor.py) and download (this file)
        attachments = []
        for part in msg.walk():
            if is_attachment_part(part):
                attachments.append(part)
        
        if index >= len(attachments):
            raise AttachmentIndexError(
                f"Attachment index {index} out of range (found {len(attachments)} attachments)"
            )
        
        part = attachments[index]
        
        # Extract and decode filename properly
        filename = self._decode_filename(part, index)
        
        # Get content type
        content_type = part.get_content_type() or 'application/octet-stream'
        
        # Decode payload
        content = part.get_payload(decode=True)
        if content is None:
            raise AttachmentNotFoundError(f"Could not decode attachment {index}")
        
        return content, filename, content_type
    
    def _decode_filename(self, part: Message, index: int) -> str:
        """
        Decode attachment filename from MIME headers.
        
        Handles:
        - RFC 2047 MIME encoded-word syntax (=?charset?encoding?text?=)
        - RFC 2231 parameter value character set and language
        - Plain text filenames
        - Fallback to Content-Type name parameter
        - Default filename if nothing else works
        
        Args:
            part: MIME message part
            index: Attachment index (for fallback name)
            
        Returns:
            Decoded filename string
        """
        filename = None
        
        # Try get_filename() first (handles most cases)
        raw_filename = part.get_filename()
        
        if raw_filename:
            filename = self._decode_mime_string(raw_filename)
        
        # Fallback: Try Content-Type 'name' parameter
        if not filename or filename in ('=', ''):
            content_type_params = part.get_params(header='Content-Type') or []
            for key, value in content_type_params:
                if key.lower() == 'name':
                    filename = self._decode_mime_string(value)
                    break
        
        # Fallback: Try Content-Disposition 'filename' parameter directly
        if not filename or filename in ('=', ''):
            cd_params = part.get_params(header='Content-Disposition') or []
            for key, value in cd_params:
                if key.lower() == 'filename':
                    filename = self._decode_mime_string(value)
                    break
        
        # Final fallback: generate a name based on content type
        if not filename or filename in ('=', ''):
            content_type = part.get_content_type() or 'application/octet-stream'
            ext = self._get_extension_for_content_type(content_type)
            filename = f"attachment_{index}{ext}"
        
        # Clean up the filename
        filename = self._sanitize_filename(filename)
        
        return filename
    
    def _decode_mime_string(self, value: any) -> str:
        """
        Decode a MIME-encoded string (RFC 2047 or RFC 2231).
        
        Args:
            value: The raw value (could be str, bytes, or encoded)
            
        Returns:
            Decoded string
        """
        if value is None:
            return ''
        
        # Handle bytes
        if isinstance(value, bytes):
            try:
                value = value.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    value = value.decode('latin-1')
                except:
                    value = value.decode('utf-8', errors='replace')
        
        # Convert to string if not already
        value = str(value)
        
        # Check for RFC 2047 encoded-word syntax: =?charset?encoding?text?=
        if '=?' in value and '?=' in value:
            try:
                decoded_parts = email.header.decode_header(value)
                result_parts = []
                for part, charset in decoded_parts:
                    if isinstance(part, bytes):
                        if charset:
                            try:
                                result_parts.append(part.decode(charset))
                            except (UnicodeDecodeError, LookupError):
                                result_parts.append(part.decode('utf-8', errors='replace'))
                        else:
                            result_parts.append(part.decode('utf-8', errors='replace'))
                    else:
                        result_parts.append(str(part))
                return ''.join(result_parts)
            except Exception as e:
                logger.debug(f"Error decoding RFC 2047: {e}")
                # Fall through to return original value
        
        # Check for RFC 2231 encoding: charset'language'encoded_value
        # e.g., utf-8''My%20File.pdf
        if "'" in value:
            rfc2231_match = re.match(r"([^']*)'([^']*)'(.+)", value)
            if rfc2231_match:
                charset, language, encoded_value = rfc2231_match.groups()
                charset = charset or 'utf-8'
                try:
                    # URL-decode the value
                    from urllib.parse import unquote
                    decoded = unquote(encoded_value, encoding=charset)
                    return decoded
                except Exception as e:
                    logger.debug(f"Error decoding RFC 2231: {e}")
        
        return value
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe filesystem use.
        
        Args:
            filename: Raw filename
            
        Returns:
            Sanitized filename
        """
        # Remove path separators and other dangerous characters
        filename = re.sub(r'[/\\:*?"<>|]', '_', filename)
        
        # Remove control characters
        filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)
        
        # Collapse multiple underscores/spaces
        filename = re.sub(r'[_\s]+', '_', filename)
        
        # Remove leading/trailing whitespace and dots
        filename = filename.strip(' ._')
        
        # Ensure non-empty
        if not filename:
            filename = 'attachment'
        
        # Limit length (255 is common filesystem limit)
        if len(filename) > 200:
            name, ext = self._split_extension(filename)
            filename = name[:200 - len(ext)] + ext
        
        return filename
    
    def _split_extension(self, filename: str) -> Tuple[str, str]:
        """Split filename into name and extension."""
        if '.' in filename:
            parts = filename.rsplit('.', 1)
            if len(parts) == 2 and len(parts[1]) <= 10:
                return parts[0], '.' + parts[1]
        return filename, ''
    
    def _get_extension_for_content_type(self, content_type: str) -> str:
        """Get a file extension for a content type."""
        extensions = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.ms-excel': '.xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'application/vnd.ms-powerpoint': '.ppt',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
            'application/zip': '.zip',
            'application/x-zip-compressed': '.zip',
            'application/x-gzip': '.gz',
            'application/x-tar': '.tar',
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/svg+xml': '.svg',
            'text/plain': '.txt',
            'text/html': '.html',
            'text/csv': '.csv',
            'application/json': '.json',
            'application/xml': '.xml',
        }
        return extensions.get(content_type.lower(), '')
