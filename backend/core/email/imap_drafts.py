"""
IMAP Drafts Manager

Handles saving and managing drafts in IMAP Drafts folder.
"""
import os
import logging
from typing import Optional
from datetime import datetime
from imapclient import IMAPClient
from email.mime.text import MIMEText
from email.utils import formataddr, make_msgid
from typing import List, Tuple

logger = logging.getLogger(__name__)


class IMAPDraftsManager:
    """Manage drafts in IMAP Drafts folder."""
    
    def __init__(
        self,
        imap_host: Optional[str] = None,
        imap_username: Optional[str] = None,
        imap_password: Optional[str] = None,
        drafts_folder: str = "Drafts"
    ):
        """
        Initialize IMAP drafts manager.
        
        Args:
            imap_host: IMAP server hostname
            imap_username: IMAP username
            imap_password: IMAP password
            drafts_folder: Name of drafts folder (default: "Drafts")
        """
        self.imap_host = imap_host or os.getenv('IMAP_HOST')
        self.imap_username = imap_username or os.getenv('IMAP_USERNAME')
        self.imap_password = imap_password or os.getenv('IMAP_PASSWORD')
        self.drafts_folder = drafts_folder
        
        if not all([self.imap_host, self.imap_username, self.imap_password]):
            logger.warning(
                "IMAP not fully configured. Set IMAP_HOST, IMAP_USERNAME, IMAP_PASSWORD "
                "environment variables to enable draft saving."
            )
    
    def save_draft(
        self,
        to_address: str,
        subject: str,
        body: str,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
        from_name: Optional[str] = None,
        from_email: Optional[str] = None,
        cc_addresses: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Save a draft to IMAP Drafts folder.
        
        Args:
            to_address: Recipient email
            subject: Email subject
            body: Email body
            in_reply_to: Message-ID being replied to
            references: References header
            from_name: From name
            from_email: From email address
            cc_addresses: List of CC recipients
        
        Returns:
            UID of saved draft in IMAP, or None if failed
        """
        result = self.save_draft_with_message_id(
            to_address=to_address,
            subject=subject,
            body=body,
            in_reply_to=in_reply_to,
            references=references,
            from_name=from_name,
            from_email=from_email,
            cc_addresses=cc_addresses
        )
        return result[0] if result else None
    
    def save_draft_with_message_id(
        self,
        to_address: str,
        subject: str,
        body: str,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
        from_name: Optional[str] = None,
        from_email: Optional[str] = None,
        cc_addresses: Optional[List[str]] = None
    ) -> Optional[Tuple[str, str]]:
        """
        Save a draft to IMAP Drafts folder and return both UID and Message-ID.
        
        Args:
            to_address: Recipient email
            subject: Email subject
            body: Email body
            in_reply_to: Message-ID being replied to
            references: References header
            from_name: From name
            from_email: From email address
            cc_addresses: List of CC recipients
        
        Returns:
            Tuple of (UID, Message-ID) or None if failed
        """
        if not all([self.imap_host, self.imap_username, self.imap_password]):
            logger.error("IMAP not configured. Cannot save draft.")
            return None
        
        try:
            # Create email message
            msg = MIMEText(body, 'plain', 'utf-8')
            
            # Set headers
            from_email = from_email or self.imap_username
            from_name = from_name or os.getenv('FROM_NAME', '')
            msg['From'] = formataddr((from_name, from_email))
            msg['To'] = to_address
            msg['Subject'] = subject
            msg['Date'] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S +0000')
            
            # CC recipients
            if cc_addresses:
                msg['Cc'] = ', '.join(cc_addresses)
            
            # Generate Message-ID
            message_id = make_msgid(domain=from_email.split('@')[1])
            msg['Message-ID'] = message_id
            
            # Threading headers
            if in_reply_to:
                msg['In-Reply-To'] = in_reply_to
            if references:
                msg['References'] = references
            elif in_reply_to:
                msg['References'] = in_reply_to
            
            # Connect to IMAP
            with IMAPClient(self.imap_host) as client:
                client.login(self.imap_username, self.imap_password)
                
                # Check if Drafts folder exists
                folders = client.list_folders()
                drafts_exists = any(self.drafts_folder in f[2] for f in folders)
                
                if not drafts_exists:
                    # Create Drafts folder if it doesn't exist
                    client.create_folder(self.drafts_folder)
                    logger.info(f"Created {self.drafts_folder} folder")
                
                # Select Drafts folder
                client.select_folder(self.drafts_folder)
                
                # Append message to Drafts (with \Draft flag)
                result = client.append(
                    self.drafts_folder,
                    msg.as_bytes(),
                    flags=[b'\\Draft'],
                    msg_time=datetime.utcnow()
                )
                
                # Extract UID from result
                uid = str(result)
                
                logger.info(f"Draft saved to IMAP {self.drafts_folder} folder with UID: {uid}, Message-ID: {message_id}")
                return (uid, message_id)
                
        except Exception as e:
            logger.error(f"Error saving draft to IMAP: {e}", exc_info=True)
            return None
    
    def delete_draft(self, uid: str) -> bool:
        """
        Delete a draft from IMAP Drafts folder.
        
        Args:
            uid: UID of draft to delete
        
        Returns:
            True if successful
        """
        if not all([self.imap_host, self.imap_username, self.imap_password]):
            logger.error("IMAP not configured. Cannot delete draft.")
            return False
        
        try:
            with IMAPClient(self.imap_host) as client:
                client.login(self.imap_username, self.imap_password)
                client.select_folder(self.drafts_folder)
                
                # Mark as deleted
                client.delete_messages([uid])
                # Expunge to permanently remove
                client.expunge()
                
                logger.info(f"Deleted draft UID {uid} from {self.drafts_folder}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting draft {uid}: {e}", exc_info=True)
            return False
    
    def get_draft(self, uid: str) -> Optional[dict]:
        """
        Retrieve a draft from IMAP.
        
        Args:
            uid: UID of draft
        
        Returns:
            Dict with draft content or None
        """
        if not all([self.imap_host, self.imap_username, self.imap_password]):
            logger.error("IMAP not configured. Cannot retrieve draft.")
            return None
        
        try:
            with IMAPClient(self.imap_host) as client:
                client.login(self.imap_username, self.imap_password)
                client.select_folder(self.drafts_folder)
                
                # Fetch the draft
                messages = client.fetch([uid], ['RFC822', 'FLAGS'])
                
                if not messages or uid not in messages:
                    logger.warning(f"Draft UID {uid} not found")
                    return None
                
                msg_data = messages[uid]
                raw_email = msg_data[b'RFC822']
                
                # Parse email
                from email import message_from_bytes
                msg = message_from_bytes(raw_email)
                
                return {
                    'to': msg.get('To'),
                    'subject': msg.get('Subject'),
                    'body': msg.get_payload(),
                    'message_id': msg.get('Message-ID')
                }
                
        except Exception as e:
            logger.error(f"Error retrieving draft {uid}: {e}", exc_info=True)
            return None
    
    def ensure_folder_exists(self, folder_path: str) -> bool:
        """
        Ensure a folder exists in IMAP, creating it if necessary.
        Creates parent folders recursively if needed.
        
        Args:
            folder_path: IMAP folder path (e.g., "MD/Applications/Inquiries")
            
        Returns:
            True if folder exists or was created
        """
        if not all([self.imap_host, self.imap_username, self.imap_password]):
            logger.error("IMAP not configured. Cannot ensure folder.")
            return False
        
        try:
            with IMAPClient(self.imap_host) as client:
                client.login(self.imap_username, self.imap_password)
                
                # Get list of existing folders
                folders = client.list_folders()
                existing_folders = [f[2] for f in folders]
                
                if folder_path in existing_folders:
                    logger.debug(f"Folder {folder_path} already exists")
                    return True
                
                # Create parent folders if needed
                parts = folder_path.split('/')
                for i in range(1, len(parts) + 1):
                    partial_path = '/'.join(parts[:i])
                    if partial_path not in existing_folders:
                        try:
                            client.create_folder(partial_path)
                            logger.info(f"Created IMAP folder: {partial_path}")
                            existing_folders.append(partial_path)
                        except Exception as e:
                            # Folder might already exist (race condition) or parent exists
                            if 'ALREADYEXISTS' not in str(e).upper():
                                logger.warning(f"Could not create folder {partial_path}: {e}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error ensuring folder {folder_path} exists: {e}", exc_info=True)
            return False
    
    def move_email(self, source_folder: str, uid: str, dest_folder: str) -> bool:
        """
        Move an email from one folder to another.
        
        Args:
            source_folder: Source IMAP folder
            uid: UID of email in source folder
            dest_folder: Destination IMAP folder
            
        Returns:
            True if successful
        """
        if not all([self.imap_host, self.imap_username, self.imap_password]):
            logger.error("IMAP not configured. Cannot move email.")
            return False
        
        try:
            with IMAPClient(self.imap_host) as client:
                client.login(self.imap_username, self.imap_password)
                
                # Ensure destination folder exists
                self.ensure_folder_exists(dest_folder)
                
                # Select source folder
                client.select_folder(source_folder)
                
                # Copy to destination
                client.copy([uid], dest_folder)
                
                # Delete from source
                client.delete_messages([uid])
                client.expunge()
                
                logger.info(f"Moved email UID {uid} from {source_folder} to {dest_folder}")
                return True
                
        except Exception as e:
            logger.error(f"Error moving email UID {uid}: {e}", exc_info=True)
            return False
    
    def search_sent_items_robust(
        self, 
        original_message_id: str, 
        to_address: str,
        subject_fragment: str,
        sent_folder: str = "Sent Items"
    ) -> Optional[dict]:
        """
        Robustly search for a sent reply to an inquiry.
        
        Searches by:
        1. In-Reply-To header matching original_message_id
        2. Subject + recipient + recent date as fallback
        
        Args:
            original_message_id: Message-ID of the original inquiry
            to_address: Recipient of the reply
            subject_fragment: Part of the expected subject
            sent_folder: Sent folder name
            
        Returns:
            Dict with sent message info or None if not found
        """
        if not all([self.imap_host, self.imap_username, self.imap_password]):
            logger.error("IMAP not configured. Cannot search sent items.")
            return None
        
        try:
            with IMAPClient(self.imap_host) as client:
                client.login(self.imap_username, self.imap_password)
                client.select_folder(sent_folder)
                
                # Method 1: Search by In-Reply-To header
                try:
                    results = client.search(['HEADER', 'In-Reply-To', original_message_id])
                    if results:
                        # Fetch the first match
                        msg_data = client.fetch(results[:1], ['ENVELOPE', 'RFC822.SIZE'])
                        if msg_data:
                            uid = results[0]
                            envelope = msg_data[uid][b'ENVELOPE']
                            return {
                                'uid': uid,
                                'subject': envelope.subject.decode() if envelope.subject else '',
                                'date': envelope.date,
                                'found_by': 'in_reply_to'
                            }
                except Exception as e:
                    logger.debug(f"In-Reply-To search failed: {e}")
                
                # Method 2: Search by subject + recipient + recent date
                from datetime import timedelta
                since_date = datetime.utcnow() - timedelta(days=7)
                date_str = since_date.strftime('%d-%b-%Y')
                
                try:
                    # Search for recent emails to this recipient
                    results = client.search([
                        'SINCE', date_str,
                        'TO', to_address
                    ])
                    
                    if results:
                        # Fetch subjects and check for match
                        msg_data = client.fetch(results, ['ENVELOPE'])
                        for uid, data in msg_data.items():
                            envelope = data[b'ENVELOPE']
                            subj = envelope.subject.decode() if envelope.subject else ''
                            if subject_fragment.lower() in subj.lower():
                                return {
                                    'uid': uid,
                                    'subject': subj,
                                    'date': envelope.date,
                                    'found_by': 'subject_recipient_date'
                                }
                except Exception as e:
                    logger.debug(f"Subject+recipient search failed: {e}")
                
                return None
                
        except Exception as e:
            logger.error(f"Error searching sent items: {e}", exc_info=True)
            return None
    
    def draft_exists(self, message_id: str) -> bool:
        """
        Check if a draft with the given Message-ID still exists in Drafts.
        
        Args:
            message_id: Message-ID of the draft
            
        Returns:
            True if draft exists
        """
        if not all([self.imap_host, self.imap_username, self.imap_password]):
            return False
        
        try:
            with IMAPClient(self.imap_host) as client:
                client.login(self.imap_username, self.imap_password)
                client.select_folder(self.drafts_folder)
                
                # Search by Message-ID header
                results = client.search(['HEADER', 'Message-ID', message_id])
                return len(results) > 0
                
        except Exception as e:
            logger.error(f"Error checking draft existence: {e}", exc_info=True)
            return False

