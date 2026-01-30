"""
Cross-Account Move Service

Handles moving emails between different IMAP accounts with retry logic and fallback methods.
"""
from email import message_from_bytes
from typing import Optional, Tuple
import logging
from datetime import datetime

from backend.core.accounts.manager import AccountManager
from backend.core.email.imap_monitor import IMAPMonitor
from backend.core.email.retry_manager import RetryManager

logger = logging.getLogger(__name__)


class CrossAccountMoveService:
    """
    Handles moving emails between different IMAP accounts.
    
    Supports multiple methods:
    1. IMAP COPY (if same server) - fastest, preserves Message-ID
    2. Download/Upload fallback - works across any servers
    
    Includes retry logic and verification.
    """
    
    def __init__(self, 
                 account_manager: AccountManager, 
                 dry_run: bool = False,
                 max_retries: int = 3):
        """
        Initialize cross-account move service.
        
        Args:
            account_manager: AccountManager instance
            dry_run: If True, only log what would happen
            max_retries: Maximum retry attempts for failed moves
        """
        self.account_manager = account_manager
        self.dry_run = dry_run
        self.retry_manager = RetryManager(max_retries=max_retries, base_delay=2.0)
        self.stats = {
            'attempted': 0,
            'succeeded': 0,
            'failed': 0,
            'method_copy': 0,
            'method_download': 0,
            'duplicates_skipped': 0
        }
        self._last_move_method = None  # Track method used in last move
    
    async def move_email(self,
                        email_uid: str,
                        message_id: str,
                        from_account: str,
                        from_folder: str,
                        to_account: str,
                        to_folder: str,
                        from_imap: IMAPMonitor) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Move email between accounts with retry logic.
        
        Args:
            email_uid: UID of email in source account
            message_id: Message-ID header value
            from_account: Source account nickname
            from_folder: Source folder name
            to_account: Target account nickname
            to_folder: Target folder name
            from_imap: IMAPMonitor instance for source account (already connected)
        
        Returns:
            (success: bool, error_message: Optional[str])
        """
        self.stats['attempted'] += 1
        
        # Validate move is allowed
        if not self.account_manager.can_move_between(from_account, to_account):
            error = f"Moves not allowed from {from_account} to {to_account}"
            logger.error(error)
            self.stats['failed'] += 1
            return False, error
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would move email {email_uid} ({message_id}) from "
                       f"{from_account}:{from_folder} to {to_account}:{to_folder}")
            return True, None, 'dry_run'
        
        # Reset last move method
        self._last_move_method = None
        
        # Use retry manager to execute move with automatic retries
        # Note: RetryManager expects (success, error) tuple, so we wrap the result
        async def wrapped_move():
            success, error, method = await self._execute_move(
                email_uid, message_id, from_account, from_folder,
                to_account, to_folder, from_imap
            )
            if success:
                self._last_move_method = method
            return success, error
        
        success, error = await self.retry_manager.execute_with_retry(
            wrapped_move,
            f"move_email({from_account}→{to_account})",
        )
        
        if success:
            self.stats['succeeded'] += 1
            return success, error, self._last_move_method
        else:
            self.stats['failed'] += 1
            logger.error(f"Failed to move email {email_uid} after retries: {error}")
            return success, error, None
    
    async def _execute_move(self,
                           email_uid: str,
                           message_id: str,
                           from_account: str,
                           from_folder: str,
                           to_account: str,
                           to_folder: str,
                           from_imap: IMAPMonitor) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Execute the actual move operation (called by retry manager).
        
        Returns:
            (success: bool, error_message: Optional[str], move_method: Optional[str])
        """
        # Try IMAP COPY first (if same server)
        success, error, method = await self._try_imap_copy(
            email_uid, from_account, from_folder, 
            to_account, to_folder, from_imap, message_id
        )
        
        if success:
            self.stats['method_copy'] += 1
            return True, None, 'imap_copy'
        
        # Different servers is expected, not a warning - just use download/upload
        if "Different IMAP servers" in str(error):
            logger.debug(f"Using download/upload method (different servers)")
        else:
            logger.warning(f"IMAP COPY failed: {error}, trying download/upload method")
        
        # Fallback to download/upload
        success, error, method = await self._try_download_upload(
            email_uid, from_account, from_folder,
            to_account, to_folder, from_imap, message_id
        )
        
        if success:
            self.stats['method_download'] += 1
            return True, None, 'download_upload'
        
        return success, error, None
    
    async def _try_imap_copy(self, 
                            uid: str,
                            from_account: str,
                            from_folder: str,
                            to_account: str,
                            to_folder: str,
                            from_imap: IMAPMonitor,
                            expected_message_id: str) -> Tuple[bool, Optional[str], str]:
        """
        Try to use IMAP APPEND (works if both accounts on same server).
        
        Note: Standard IMAP COPY doesn't work across accounts, so we use APPEND.
        """
        from_config = self.account_manager.get_account(from_account)
        to_config = self.account_manager.get_account(to_account)
        
        # Check if same server (APPEND works across accounts on same server)
        if from_config.imap_host != to_config.imap_host:
            return False, "Different IMAP servers (use download/upload)", "none"
        
        to_imap = None
        try:
            # Create connection to target account
            to_imap_config = self.account_manager.get_imap_config(to_account)
            to_imap = IMAPMonitor(to_imap_config)
            to_imap.connect()
            
            # Ensure target folder exists
            to_imap.create_folder(to_folder)
            
            # Get email data from source
            from_imap.client.select_folder(from_folder)
            
            # Validate UID is numeric
            try:
                uid_int = int(uid)
            except (ValueError, TypeError):
                return False, f"Invalid UID format: {uid} (must be numeric)", "none"
            
            msg_data = from_imap.client.fetch([uid_int], ['RFC822', 'FLAGS'])
            
            # Handle UIDVALIDITY changes: if UID not found, try looking up by Message-ID
            if uid_int not in msg_data:
                logger.warning(f"UID {uid} not found, attempting Message-ID lookup for {expected_message_id}")
                from_imap.client.select_folder(from_folder)
                search_results = from_imap.client.search(['HEADER', 'Message-ID', expected_message_id])
                if search_results:
                    new_uid = search_results[0]
                    logger.info(f"Found email with new UID: {new_uid} (was {uid})")
                    # Retry fetch with new UID
                    msg_data = from_imap.client.fetch([new_uid], ['RFC822', 'FLAGS'])
                    if new_uid not in msg_data:
                        return False, f"Failed to fetch email after UID lookup", "none"
                    uid_int = new_uid  # Update for later use
                else:
                    return False, f"Failed to fetch email {uid} from source and Message-ID lookup failed", "none"
            
            email_data = msg_data[uid_int]
            raw_email = email_data[b'RFC822']
            flags = email_data.get(b'FLAGS', [])
            
            # Parse to verify Message-ID
            msg = message_from_bytes(raw_email)
            message_id = msg.get('Message-ID', '')
            if message_id != expected_message_id:
                logger.warning(f"Message-ID mismatch: expected {expected_message_id}, got {message_id}")
            
            # Check if already exists in target
            to_imap.client.select_folder(to_folder)
            existing = to_imap.client.search(['HEADER', 'Message-ID', message_id])
            
            if existing:
                logger.info(f"Email already exists in {to_account}:{to_folder}, skipping copy")
                self.stats['duplicates_skipped'] += 1
                # Still delete from source
                from_imap.client.set_flags([int(uid)], [b'\\Deleted'])
                from_imap.client.expunge()
                return True, None
            
            # Append to target account
            to_imap.client.append(to_folder, raw_email, flags=flags)
            
            # Verify it arrived
            verify = to_imap.client.search(['HEADER', 'Message-ID', message_id])
            if not verify:
                return False, "Failed to verify copy in target account", "none"
            
            logger.info(f"Successfully copied email {uid} to {to_account}:{to_folder}")
            
            # Delete from source (uid_int already validated above)
            from_imap.client.set_flags([uid_int], [b'\\Deleted'])
            from_imap.client.expunge()
            
            return True, None, "imap_copy"
            
        except Exception as e:
            logger.error(f"IMAP COPY failed: {e}")
            return False, str(e), "none"
        finally:
            if to_imap:
                to_imap.disconnect()
    
    async def _try_download_upload(self,
                                  uid: str,
                                  from_account: str,
                                  from_folder: str,
                                  to_account: str,
                                  to_folder: str,
                                  from_imap: IMAPMonitor,
                                  expected_message_id: str) -> Tuple[bool, Optional[str], str]:
        """
        Download from source and upload to target (works across any servers).
        """
        to_imap = None
        try:
            # Download from source
            from_imap.client.select_folder(from_folder)
            
            # Validate UID is numeric
            try:
                uid_int = int(uid)
            except (ValueError, TypeError):
                return False, f"Invalid UID format: {uid} (must be numeric)", "none"
            
            msg_data = from_imap.client.fetch([uid_int], ['RFC822', 'FLAGS', 'INTERNALDATE'])
            
            # Handle UIDVALIDITY changes: if UID not found, try looking up by Message-ID
            if uid_int not in msg_data:
                logger.warning(f"UID {uid} not found, attempting Message-ID lookup for {expected_message_id}")
                from_imap.client.select_folder(from_folder)
                search_results = from_imap.client.search(['HEADER', 'Message-ID', expected_message_id])
                if search_results:
                    new_uid = search_results[0]
                    logger.info(f"Found email with new UID: {new_uid} (was {uid})")
                    # Retry fetch with new UID
                    msg_data = from_imap.client.fetch([new_uid], ['RFC822', 'FLAGS', 'INTERNALDATE'])
                    if new_uid not in msg_data:
                        return False, f"Failed to fetch email after UID lookup", "none"
                    uid_int = new_uid  # Update for later use
                else:
                    return False, f"Failed to fetch email {uid} from source and Message-ID lookup failed", "none"
            
            email_data = msg_data[uid_int]
            raw_email = email_data[b'RFC822']
            flags = email_data.get(b'FLAGS', [])
            internal_date = email_data.get(b'INTERNALDATE')
            
            # Parse to verify Message-ID
            msg = message_from_bytes(raw_email)
            message_id = msg.get('Message-ID', '')
            if message_id != expected_message_id:
                logger.warning(f"Message-ID mismatch: expected {expected_message_id}, got {message_id}")
            
            # Connect to target
            to_imap_config = self.account_manager.get_imap_config(to_account)
            to_imap = IMAPMonitor(to_imap_config)
            to_imap.connect()
            
            # Ensure target folder exists
            to_imap.create_folder(to_folder)
            
            # Check if already exists
            to_imap.client.select_folder(to_folder)
            existing = to_imap.client.search(['HEADER', 'Message-ID', message_id])
            
            if existing:
                logger.info(f"Email already exists in {to_account}:{to_folder}, skipping upload")
                self.stats['duplicates_skipped'] += 1
            else:
                # Upload to target with original date and flags
                to_imap.client.append(
                    to_folder, 
                    raw_email,
                    flags=flags,
                    msg_time=internal_date
                )
                
                # Verify
                verify = to_imap.client.search(['HEADER', 'Message-ID', message_id])
                if not verify:
                    return False, "Failed to verify upload to target account", "none"
                
                logger.info(f"Successfully uploaded email {uid} to {to_account}:{to_folder}")
            
            # Delete from source (uid_int already validated above)
            from_imap.client.set_flags([uid_int], [b'\\Deleted'])
            from_imap.client.expunge()
            
            return True, None, "download_upload"
            
        except Exception as e:
            logger.error(f"Download/upload failed: {e}")
            return False, str(e), "none"
        finally:
            if to_imap:
                to_imap.disconnect()
    
    def get_stats(self) -> dict:
        """Get movement statistics"""
        return self.stats.copy()

