"""
IMAP Actions Service

Handles IMAP operations triggered by UI actions:
- Move to spam/trash/archive folders
- Mark as read/unread
- Clear/set flags
- Auto-create folders if missing
"""
import imaplib
import logging
from typing import Optional, Tuple
from contextlib import contextmanager

from backend.core.config import get_settings
from backend.core.database.models import Email, EmailMetadata, EmailLocationHistory
from backend.core.database.repository import EmailRepository

logger = logging.getLogger(__name__)


class IMAPActionError(Exception):
    """Custom exception for IMAP action failures"""
    pass


class IMAPActionService:
    """Service for performing IMAP operations from UI actions"""
    
    def __init__(self):
        self.settings = get_settings()
    
    @contextmanager
    def _get_imap_connection(self):
        """
        Create IMAP connection context manager.
        
        Yields:
            imaplib.IMAP4_SSL connection
            
        Raises:
            IMAPActionError: If connection fails
        """
        connection = None
        try:
            # Connect to IMAP server
            if self.settings.imap_use_ssl:
                connection = imaplib.IMAP4_SSL(
                    self.settings.imap_host,
                    self.settings.imap_port,
                    timeout=30
                )
            else:
                connection = imaplib.IMAP4(
                    self.settings.imap_host,
                    self.settings.imap_port,
                    timeout=30
                )
            
            # Login
            connection.login(
                self.settings.imap_username,
                self.settings.imap_password
            )
            
            yield connection
            
        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP connection error: {e}")
            raise IMAPActionError(f"IMAP connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected IMAP error: {e}", exc_info=True)
            raise IMAPActionError(f"IMAP error: {str(e)}")
        finally:
            if connection:
                try:
                    connection.logout()
                except:
                    pass
    
    def _get_actual_folder(self, email: Email, repo: EmailRepository) -> str:
        """
        Get the actual current folder of an email.
        
        The email.folder field in the database might be stale if the email was moved
        by Mail.app or another client. This checks the folder history to find the
        most recent folder location.
        
        Args:
            email: Email database object
            repo: Email repository for database access
            
        Returns:
            Actual current folder name
        """
        # Check if there's folder history
        last_history = repo.db.query(EmailLocationHistory).filter(
            EmailLocationHistory.email_id == email.id
        ).order_by(EmailLocationHistory.moved_at.desc()).first()
        
        if last_history:
            actual_folder = last_history.to_folder
            # Update email.folder if it's out of sync
            if actual_folder != email.folder:
                logger.info(f"Email {email.id} folder mismatch: DB says '{email.folder}' but history says '{actual_folder}'. Using history.")
                return actual_folder
        
        # Fall back to email.folder if no history
        return email.folder or 'INBOX'
    
    def _ensure_folder_exists(self, connection: imaplib.IMAP4_SSL, folder: str) -> bool:
        """
        Check if folder exists, create if missing and auto_create_folders is True.
        
        Args:
            connection: IMAP connection
            folder: Folder name/path
            
        Returns:
            True if folder exists or was created, False otherwise
            
        Raises:
            IMAPActionError: If folder doesn't exist and can't be created
        """
        try:
            # List folder to check if it exists
            status, folders = connection.list(pattern=folder)
            
            if status == 'OK' and folders and folders[0] is not None:
                # Folder exists
                return True
            
            # Folder doesn't exist
            if not self.settings.auto_create_folders:
                raise IMAPActionError(
                    f"Folder '{folder}' does not exist. "
                    "Enable AUTO_CREATE_FOLDERS=true to auto-create."
                )
            
            # Create folder
            logger.info(f"Creating IMAP folder: {folder}")
            status, response = connection.create(folder)
            
            if status != 'OK':
                raise IMAPActionError(f"Failed to create folder '{folder}': {response}")
            
            logger.info(f"Successfully created folder: {folder}")
            return True
            
        except imaplib.IMAP4.error as e:
            raise IMAPActionError(f"Error checking/creating folder '{folder}': {str(e)}")
    
    def _move_email(
        self,
        connection: imaplib.IMAP4_SSL,
        email: Email,
        target_folder: str,
        repo: EmailRepository
    ) -> Tuple[bool, str]:
        """
        Move email to target folder via IMAP.
        
        UIDs change when emails move between folders, so we search by Message-ID
        which is permanent and unique across all folders.
        
        Args:
            connection: IMAP connection
            email: Email database object
            target_folder: Target folder name
            repo: Email repository for database access
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Ensure target folder exists
            self._ensure_folder_exists(connection, target_folder)
            
            # Get actual current folder (may have changed)
            current_folder = self._get_actual_folder(email, repo)
            status, response = connection.select(current_folder)
            
            if status != 'OK':
                return False, f"Failed to select folder '{current_folder}': {response}"
            
            # Search for email by Message-ID (more reliable than UID which changes per folder)
            if not email.message_id:
                return False, f"Email has no Message-ID, cannot locate on IMAP server"
            
            # Strip angle brackets if present
            message_id = email.message_id.strip('<>')
            
            # Search using IMAP HEADER command
            status, data = connection.search(None, 'HEADER', 'Message-ID', message_id)
            
            if status != 'OK' or not data[0]:
                logger.warning(f"Email with Message-ID {message_id} not found in {current_folder}")
                return False, f"Email not found in {current_folder}"
            
            # Get the UID of the found message
            msg_nums = data[0].split()
            if not msg_nums:
                return False, f"Email not found in {current_folder}"
            
            msg_num = msg_nums[0]
            
            # Get UID for the message
            status, uid_data = connection.fetch(msg_num, '(UID)')
            if status != 'OK':
                return False, f"Failed to get UID for message"
            
            # Parse UID from response
            import re
            uid_match = re.search(r'UID (\d+)', str(uid_data[0]))
            if not uid_match:
                return False, f"Could not parse UID from IMAP response"
            
            uid = uid_match.group(1)
            logger.info(f"Found email in {current_folder} with UID {uid}")
            
            # Copy to target folder
            status, response = connection.uid('COPY', uid, target_folder)
            
            if status != 'OK':
                return False, f"Failed to copy email to '{target_folder}': {response}"
            
            # Mark original as deleted
            status, response = connection.uid('STORE', uid, '+FLAGS', '\\Deleted')
            
            if status != 'OK':
                logger.warning(f"Failed to mark email as deleted: {response}")
            
            # Expunge to actually delete
            connection.expunge()
            
            logger.info(f"Moved email {email.id} from {current_folder} to {target_folder}")
            return True, f"Moved to {target_folder}"
            
        except Exception as e:
            logger.error(f"Error moving email: {e}", exc_info=True)
            return False, f"Move failed: {str(e)}"
    
    def _mark_as_read(
        self,
        connection: imaplib.IMAP4_SSL,
        email: Email,
        repo: EmailRepository
    ) -> Tuple[bool, str]:
        """
        Mark email as read (\\Seen flag).
        
        Uses Message-ID to find email since UID changes when email moves folders.
        
        Args:
            connection: IMAP connection
            email: Email database object
            repo: Email repository for database access
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Get actual current folder
            current_folder = self._get_actual_folder(email, repo)
            status, response = connection.select(current_folder)
            
            if status != 'OK':
                return False, f"Failed to select folder: {response}"
            
            # Find email by Message-ID
            if not email.message_id:
                return False, f"Email has no Message-ID"
            
            message_id = email.message_id.strip('<>')
            status, data = connection.search(None, 'HEADER', 'Message-ID', message_id)
            
            if status != 'OK' or not data[0]:
                return False, f"Email not found in {current_folder}"
            
            msg_nums = data[0].split()
            if not msg_nums:
                return False, f"Email not found"
            
            msg_num = msg_nums[0]
            
            # Set \\Seen flag (using message sequence number)
            status, response = connection.store(msg_num, '+FLAGS', '\\Seen')
            
            if status != 'OK':
                return False, f"Failed to mark as read: {response}"
            
            logger.info(f"Marked email {email.id} as read")
            return True, "Marked as read"
            
        except Exception as e:
            logger.error(f"Error marking as read: {e}", exc_info=True)
            return False, f"Mark read failed: {str(e)}"
    
    def _clear_flags(
        self,
        connection: imaplib.IMAP4_SSL,
        email: Email,
        repo: EmailRepository
    ) -> Tuple[bool, str]:
        """
        Clear all flags except \\Seen (remove \\Flagged, etc).
        
        Uses Message-ID to find email since UID changes when email moves folders.
        
        Args:
            connection: IMAP connection
            email: Email database object
            repo: Email repository for database access
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Get actual current folder
            current_folder = self._get_actual_folder(email, repo)
            status, response = connection.select(current_folder)
            
            if status != 'OK':
                return False, f"Failed to select folder: {response}"
            
            # Find email by Message-ID
            if not email.message_id:
                return False, f"Email has no Message-ID"
            
            message_id = email.message_id.strip('<>')
            status, data = connection.search(None, 'HEADER', 'Message-ID', message_id)
            
            if status != 'OK' or not data[0]:
                return False, f"Email not found in {current_folder}"
            
            msg_nums = data[0].split()
            if not msg_nums:
                return False, f"Email not found"
            
            msg_num = msg_nums[0]
            
            # Remove \\Flagged flag
            status, response = connection.store(msg_num, '-FLAGS', '\\Flagged')
            
            if status != 'OK':
                logger.warning(f"Failed to remove \\Flagged: {response}")
            
            logger.info(f"Cleared flags for email {email.id}")
            return True, "Flags cleared"
            
        except Exception as e:
            logger.error(f"Error clearing flags: {e}", exc_info=True)
            return False, f"Clear flags failed: {str(e)}"
    
    async def mark_as_spam(
        self,
        email: Email,
        repo: EmailRepository
    ) -> Tuple[bool, str]:
        """
        Mark email as spam: move to spam folder, mark as read.
        
        Args:
            email: Email database object
            repo: Email repository for DB updates
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            with self._get_imap_connection() as conn:
                # Move to spam folder
                success, message = self._move_email(conn, email, self.settings.folder_spam, repo)
                
                if not success:
                    raise IMAPActionError(message)
                
                # Mark as read
                read_success, read_msg = self._mark_as_read(conn, email, repo)
                if not read_success:
                    logger.warning(f"Email moved but mark read failed: {read_msg}")
            
            # Update database
            # Get or create metadata
            metadata = repo.db.query(EmailMetadata).filter(
                EmailMetadata.email_id == email.id
            ).first()
            
            if not metadata:
                metadata = EmailMetadata(email_id=email.id)
                repo.db.add(metadata)
            
            # Add user-spam tag
            if 'user-spam' not in (metadata.user_tags or []):
                tags = list(metadata.user_tags or [])
                tags.append('user-spam')
                metadata.user_tags = tags
            
            # Update email folder and flags
            email.folder = self.settings.folder_spam
            email.is_seen = True
            
            # Track folder change
            repo.track_folder_change(
                email,
                self.settings.folder_spam,
                moved_by='user',
                move_reason='Marked as spam via UI'
            )
            
            repo.db.commit()
            
            return True, f"Moved to {self.settings.folder_spam} and marked as spam"
            
        except IMAPActionError as e:
            repo.db.rollback()
            logger.error(f"Spam action failed: {e}")
            return False, str(e)
        except Exception as e:
            repo.db.rollback()
            logger.error(f"Unexpected error in mark_as_spam: {e}", exc_info=True)
            return False, f"Spam action failed: {str(e)}"
    
    async def delete_email(
        self,
        email: Email,
        repo: EmailRepository
    ) -> Tuple[bool, str]:
        """
        Delete email: move to trash folder, mark as read, mark as handled.
        
        Args:
            email: Email database object
            repo: Email repository for DB updates
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            with self._get_imap_connection() as conn:
                # Move to trash folder
                success, message = self._move_email(conn, email, self.settings.folder_trash, repo)
                
                if not success:
                    raise IMAPActionError(message)
                
                # Mark as read
                read_success, read_msg = self._mark_as_read(conn, email, repo)
                if not read_success:
                    logger.warning(f"Email moved but mark read failed: {read_msg}")
            
            # Update database
            email.folder = self.settings.folder_trash
            email.is_seen = True
            
            # Add 'handled' tag to metadata (get or create metadata)
            metadata = repo.db.query(EmailMetadata).filter(
                EmailMetadata.email_id == email.id
            ).first()
            
            if not metadata:
                metadata = EmailMetadata(email_id=email.id)
                repo.db.add(metadata)
            
            if 'handled' not in (metadata.user_tags or []):
                tags = list(metadata.user_tags or [])
                tags.append('handled')
                metadata.user_tags = tags
            
            # Track folder change
            repo.track_folder_change(
                email,
                self.settings.folder_trash,
                moved_by='user',
                move_reason='Deleted via UI'
            )
            
            repo.db.commit()
            
            return True, f"Moved to {self.settings.folder_trash}"
            
        except IMAPActionError as e:
            repo.db.rollback()
            logger.error(f"Delete action failed: {e}")
            return False, str(e)
        except Exception as e:
            repo.db.rollback()
            logger.error(f"Unexpected error in delete_email: {e}", exc_info=True)
            return False, f"Delete failed: {str(e)}"
    
    async def archive_email(
        self,
        email: Email,
        repo: EmailRepository
    ) -> Tuple[bool, str]:
        """
        Archive email: move to archive folder, mark as read.
        
        Args:
            email: Email database object
            repo: Email repository for DB updates
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            with self._get_imap_connection() as conn:
                # Move to archive folder
                success, message = self._move_email(conn, email, self.settings.folder_archive, repo)
                
                if not success:
                    raise IMAPActionError(message)
                
                # Mark as read
                read_success, read_msg = self._mark_as_read(conn, email, repo)
                if not read_success:
                    logger.warning(f"Email moved but mark read failed: {read_msg}")
            
            # Update database
            # Get or create metadata
            metadata = repo.db.query(EmailMetadata).filter(
                EmailMetadata.email_id == email.id
            ).first()
            
            if not metadata:
                metadata = EmailMetadata(email_id=email.id)
                repo.db.add(metadata)
            
            metadata.user_archived = True
            
            email.folder = self.settings.folder_archive
            email.is_seen = True
            
            # Track folder change
            repo.track_folder_change(
                email,
                self.settings.folder_archive,
                moved_by='user',
                move_reason='Archived via UI'
            )
            
            repo.db.commit()
            
            return True, f"Moved to {self.settings.folder_archive}"
            
        except IMAPActionError as e:
            repo.db.rollback()
            logger.error(f"Archive action failed: {e}")
            return False, str(e)
        except Exception as e:
            repo.db.rollback()
            logger.error(f"Unexpected error in archive_email: {e}", exc_info=True)
            return False, f"Archive failed: {str(e)}"
    
    async def mark_as_handled(
        self,
        email: Email,
        repo: EmailRepository
    ) -> Tuple[bool, str]:
        """
        Mark email as handled: mark as read, clear flags, stay in current folder.
        
        Args:
            email: Email database object
            repo: Email repository for DB updates
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            with self._get_imap_connection() as conn:
                # Mark as read
                read_success, read_msg = self._mark_as_read(conn, email, repo)
                
                if not read_success:
                    raise IMAPActionError(read_msg)
                
                # Clear flags (but keep labels/tags in metadata)
                flag_success, flag_msg = self._clear_flags(conn, email, repo)
                if not flag_success:
                    logger.warning(f"Marked read but clear flags failed: {flag_msg}")
            
            # Update database
            # Get or create metadata
            metadata = repo.db.query(EmailMetadata).filter(
                EmailMetadata.email_id == email.id
            ).first()
            
            if not metadata:
                metadata = EmailMetadata(email_id=email.id)
                repo.db.add(metadata)
            
            # Add 'handled' tag
            if 'handled' not in (metadata.user_tags or []):
                tags = list(metadata.user_tags or [])
                tags.append('handled')
                metadata.user_tags = tags
            
            # Update email flags
            email.is_seen = True
            email.is_flagged = False
            
            repo.db.commit()
            
            return True, "Marked as handled (read, flags cleared)"
            
        except IMAPActionError as e:
            repo.db.rollback()
            logger.error(f"Mark handled action failed: {e}")
            return False, str(e)
        except Exception as e:
            repo.db.rollback()
            logger.error(f"Unexpected error in mark_as_handled: {e}", exc_info=True)
            return False, f"Mark handled failed: {str(e)}"


