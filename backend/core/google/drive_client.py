"""
Google Drive and Sheets integration for exporting application data.

This module handles:
- Uploading files to Google Drive
- Creating Google Sheets
- Managing folder structure
- Storing file links in the database
"""

import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
import gspread
from gspread.exceptions import APIError

logger = logging.getLogger(__name__)


class GoogleDriveClient:
    """Client for Google Drive and Sheets operations"""
    
    def __init__(self, service_account_path: str, drive_folder_id: str):
        """
        Initialize Google Drive client.
        
        Args:
            service_account_path: Path to service account JSON key file
            drive_folder_id: Google Drive folder ID where files will be stored
        """
        self.drive_folder_id = drive_folder_id
        self.service_account_path = service_account_path
        
        # Load credentials
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(f"Service account file not found: {service_account_path}")
        
        # Load service account email for error messages
        with open(service_account_path, 'r') as f:
            service_account_data = json.load(f)
            self.service_account_email = service_account_data.get('client_email', 'unknown')
        
        # Initialize Drive API
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=[
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/spreadsheets'
            ]
        )
        
        self.drive_service = build('drive', 'v3', credentials=credentials)
        self.gc = gspread.authorize(credentials)
        
        # Verify folder access (read permissions)
        self._verify_folder_access()
        
        # Skip write permission test - it's working and creates unnecessary test folders
        # self._verify_write_permissions()
        
        logger.info(f"Google Drive client initialized (folder ID: {drive_folder_id})")
    
    def _verify_folder_access(self, max_retries=3, retry_delay=2):
        """Verify that the service account can access the target folder"""
        # Retry logic to account for permission propagation delays
        for attempt in range(max_retries):
            # Try multiple methods to verify access
            access_verified = False
            last_error = None
            
            # Method 1: Try to get folder metadata
            try:
                folder = self.drive_service.files().get(
                    fileId=self.drive_folder_id,
                    fields='id, name, permissions',
                    supportsAllDrives=True
                ).execute()
                logger.debug(f"Verified access to folder '{folder.get('name', 'unknown')}' (ID: {self.drive_folder_id})")
                access_verified = True
            except HttpError as e:
                last_error = e
                logger.debug(f"Method 1 (get metadata) failed: {e.resp.status} - {e}")
            
            # Method 2: Try to list files in the folder (more robust test)
            if not access_verified:
                try:
                    query = f"'{self.drive_folder_id}' in parents and trashed=false"
                    results = self.drive_service.files().list(
                        q=query,
                        pageSize=1,
                        fields='files(id)',
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True
                    ).execute()
                    logger.debug(f"Verified access by listing files in folder (ID: {self.drive_folder_id})")
                    access_verified = True
                except HttpError as e:
                    last_error = e
                    logger.debug(f"Method 2 (list files) failed: {e.resp.status} - {e}")
            
            # If access verified, we're done
            if access_verified:
                return
            
            # If this wasn't the last attempt, wait and retry
            if attempt < max_retries - 1:
                logger.debug(f"Access verification failed (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        # If all retries failed, raise an error with helpful message
        if not access_verified and last_error:
            status_code = last_error.resp.status if hasattr(last_error, 'resp') else 'unknown'
            if status_code == 404:
                error_msg = (
                    f"❌ Cannot access Google Drive folder (ID: {self.drive_folder_id}).\n"
                    f"   The service account '{self.service_account_email}' does not have access to this folder.\n\n"
                    f"   Troubleshooting steps:\n"
                    f"   1. Open the folder in Google Drive:\n"
                    f"      https://drive.google.com/drive/folders/{self.drive_folder_id}\n"
                    f"   2. Click 'Share' button\n"
                    f"   3. Add this email address: {self.service_account_email}\n"
                    f"   4. Grant 'Editor' permissions (not just 'Viewer')\n"
                    f"   5. Make sure 'Notify people' is unchecked (not needed for service accounts)\n"
                    f"   6. Wait 1-2 minutes for permissions to propagate\n"
                    f"   7. Try running the script again\n\n"
                    f"   Note: If you just shared the folder, Google may take a few minutes to propagate permissions."
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from last_error
            elif status_code == 403:
                error_msg = (
                    f"❌ Permission denied accessing Google Drive folder (ID: {self.drive_folder_id}).\n"
                    f"   The service account '{self.service_account_email}' may not have sufficient permissions.\n\n"
                    f"   Please ensure:\n"
                    f"   1. The folder is shared with '{self.service_account_email}' with 'Editor' role\n"
                    f"   2. The Google Drive API is enabled in your Google Cloud project:\n"
                    f"      https://console.cloud.google.com/apis/library/drive.googleapis.com\n"
                    f"   3. The service account has the necessary IAM roles in Google Cloud Console"
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from last_error
            else:
                error_msg = (
                    f"❌ Unexpected error accessing Google Drive folder (ID: {self.drive_folder_id}):\n"
                    f"   Status: {status_code}\n"
                    f"   Error: {last_error}\n"
                    f"   Service account: {self.service_account_email}"
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from last_error
    
    def _verify_write_permissions(self):
        """Verify that the service account can create folders (write permissions)"""
        test_folder_name = f".test_write_permissions_{int(time.time())}"
        test_folder_id = None
        try:
            # Try to create a test folder
            test_folder_id = self.create_folder(test_folder_name, self.drive_folder_id)
            logger.debug(f"Created test folder '{test_folder_name}' (ID: {test_folder_id})")
            
            # Wait a moment for the folder to be fully created (propagation delay)
            time.sleep(0.5)
            
            # Verify the folder exists before trying to delete
            try:
                self.drive_service.files().get(
                    fileId=test_folder_id,
                    fields='id',
                    supportsAllDrives=True
                ).execute()
            except HttpError as get_error:
                logger.warning(f"Test folder created but cannot verify it exists: {get_error}")
                # Continue with deletion attempt anyway
            
            # Try to delete it
            deleted = False
            try:
                self.drive_service.files().delete(
                    fileId=test_folder_id,
                    supportsAllDrives=True
                ).execute()
                deleted = True
                logger.debug("Verified write permissions - successfully created and deleted test folder")
            except HttpError as delete_error:
                if delete_error.resp.status == 404:
                    # Folder might have been auto-deleted or doesn't exist
                    logger.debug(f"Test folder not found for deletion (may have been auto-cleaned)")
                    deleted = True  # Consider it deleted if it doesn't exist
                elif delete_error.resp.status == 403:
                    # No delete permission, try moving to trash instead
                    logger.debug("No delete permission, trying to move test folder to trash")
                    try:
                        self.drive_service.files().update(
                            fileId=test_folder_id,
                            body={'trashed': True},
                            supportsAllDrives=True
                        ).execute()
                        deleted = True
                        logger.debug("Successfully moved test folder to trash")
                    except HttpError as trash_error:
                        # No delete permission, but creation worked - that's acceptable
                        logger.warning(
                            f"Test folder created successfully but cannot delete/trash it (no delete permission). "
                            f"This is acceptable - creation permissions verified. Folder ID: {test_folder_id}"
                        )
                else:
                    logger.warning(f"Created test folder but could not delete it: {delete_error}")
                    # Store folder ID for potential manual cleanup
                    logger.info(f"Test folder '{test_folder_name}' (ID: {test_folder_id}) may need manual cleanup")
            
            if not deleted:
                logger.info(f"Note: Test folder '{test_folder_name}' (ID: {test_folder_id}) was created but not deleted. You may want to clean it up manually.")
        except PermissionError:
            # Re-raise permission errors with improved message
            raise
        except HttpError as e:
            if e.resp.status == 404:
                error_msg = (
                    f"❌ Cannot create folders in Google Drive folder (ID: {self.drive_folder_id}).\n"
                    f"   The service account '{self.service_account_email}' needs 'Editor' permissions.\n\n"
                    f"   Current issue: The service account can READ the folder but cannot WRITE/CREATE.\n"
                    f"   This usually means it has 'Viewer' permissions instead of 'Editor' permissions.\n\n"
                    f"   To fix:\n"
                    f"   1. Open the folder in Google Drive:\n"
                    f"      https://drive.google.com/drive/folders/{self.drive_folder_id}\n"
                    f"   2. Click 'Share' button\n"
                    f"   3. Find '{self.service_account_email}' in the sharing list\n"
                    f"   4. Change permissions from 'Viewer' to 'Editor'\n"
                    f"   5. Wait 1-2 minutes for permissions to propagate\n"
                    f"   6. Try running the script again"
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from e
            elif e.resp.status == 403:
                error_msg = (
                    f"❌ Permission denied - cannot create folders.\n"
                    f"   The service account '{self.service_account_email}' needs 'Editor' permissions.\n"
                    f"   Please ensure the folder (ID: {self.drive_folder_id}) is shared with 'Editor' role."
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from e
            else:
                # For other errors, log but don't fail initialization
                logger.warning(f"Could not verify write permissions: {e}")
    
    def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> str:
        """
        Create a folder in Google Drive.
        
        Args:
            folder_name: Name of the folder to create
            parent_folder_id: Parent folder ID (defaults to self.drive_folder_id)
            
        Returns:
            Folder ID of the created folder
        """
        parent_id = parent_folder_id or self.drive_folder_id
        
        # First verify we can access the parent folder
        try:
            parent_info = self.drive_service.files().get(
                fileId=parent_id,
                fields='id, name, capabilities',
                supportsAllDrives=True
            ).execute()
            logger.debug(f"Parent folder accessible: '{parent_info.get('name', 'unknown')}' (ID: {parent_id})")
        except HttpError as e:
            error_msg = (
                f"❌ Cannot access parent folder (ID: {parent_id}) before creating subfolder '{folder_name}'.\n"
                f"   Error: {e}\n"
                f"   Please verify the folder exists and is shared with '{self.service_account_email}'."
            )
            logger.error(error_msg)
            raise PermissionError(error_msg) from e
        
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        
        try:
            # Try with supportsAllDrives=True in case this is a shared drive
            folder = self.drive_service.files().create(
                body=file_metadata,
                fields='id',
                supportsAllDrives=True
            ).execute()
            
            folder_id = folder.get('id')
            logger.debug(f"Created folder '{folder_name}' (ID: {folder_id})")
            return folder_id
        except HttpError as e:
            # Log full error details for debugging
            error_details = str(e)
            if hasattr(e, 'content'):
                try:
                    import json
                    error_content = json.loads(e.content.decode('utf-8'))
                    error_details = json.dumps(error_content, indent=2)
                except:
                    error_details = str(e.content)
            
            logger.debug(f"Full error details for folder creation: {error_details}")
            
            if e.resp.status == 404:
                error_msg = (
                    f"❌ Cannot create folder '{folder_name}' in parent folder (ID: {parent_id}).\n"
                    f"   The service account '{self.service_account_email}' needs write permissions.\n\n"
                    f"   Error details: {error_details}\n\n"
                    f"   Troubleshooting:\n"
                    f"   1. Verify the service account has 'Editor' or 'Contributor' permissions:\n"
                    f"      https://drive.google.com/drive/folders/{parent_id}\n"
                    f"   2. Check that permissions have propagated (wait 2-3 minutes after sharing)\n"
                    f"   3. Ensure the Google Drive API is enabled in your Google Cloud project\n"
                    f"   4. Try removing and re-adding the service account with 'Editor' permissions\n"
                    f"   5. Check if the folder has any organizational policies restricting folder creation"
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from e
            elif e.resp.status == 403:
                error_msg = (
                    f"❌ Permission denied creating folder '{folder_name}'.\n"
                    f"   Error details: {error_details}\n"
                    f"   The service account '{self.service_account_email}' needs 'Editor' permissions.\n"
                    f"   Please ensure the parent folder (ID: {parent_id}) is shared with 'Editor' role."
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from e
            else:
                error_msg = (
                    f"❌ Unexpected error creating folder '{folder_name}': HTTP {e.resp.status}\n"
                    f"   Error details: {error_details}"
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from e
    
    def find_files_by_name(self, file_name: str, folder_id: str) -> List[Dict[str, str]]:
        """
        Find all existing files by name in a Google Drive folder.
        
        Args:
            file_name: Name of the file to find
            folder_id: Google Drive folder ID to search in
            
        Returns:
            List of dicts with 'id', 'name', and 'modifiedTime' for each matching file
        """
        try:
            # Escape backslashes and single quotes in file name for Google Drive API query
            # First escape backslashes, then escape single quotes
            escaped_name = file_name.replace("\\", "\\\\").replace("'", "\\'")
            
            # Search for files with the exact name in the specified folder
            query = f"name='{escaped_name}' and '{folder_id}' in parents and trashed=false"
            results = self.drive_service.files().list(
                q=query,
                fields='files(id, name, modifiedTime)',
                orderBy='modifiedTime desc',  # Most recent first
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            files = results.get('files', [])
            if files:
                logger.debug(f"Found {len(files)} existing file(s) with name '{file_name}'")
            return files
        except HttpError as e:
            logger.warning(f"Error searching for existing file '{file_name}': {e}")
            return []
    
    def trash_file(self, file_id: str) -> bool:
        """
        Move a file to trash in Google Drive.
        
        Args:
            file_id: Google Drive file ID to trash
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.drive_service.files().update(
                fileId=file_id,
                body={'trashed': True},
                supportsAllDrives=True
            ).execute()
            logger.debug(f"Trashed file (ID: {file_id})")
            return True
        except HttpError as e:
            logger.warning(f"Failed to trash file (ID: {file_id}): {e}")
            return False
    
    def upload_file(self, file_path: Path, folder_id: str, file_name: Optional[str] = None) -> Dict[str, str]:
        """
        Upload a file to Google Drive. If file(s) with the same name already exist,
        the most recent one will be overwritten and the others will be moved to Trash.
        
        Args:
            file_path: Local path to the file
            folder_id: Google Drive folder ID where to upload
            file_name: Optional custom file name (defaults to file_path.name)
            
        Returns:
            Dict with 'file_id' and 'web_view_link'
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_name = file_name or file_path.name
        
        # Check if files with the same name already exist
        existing_files = self.find_files_by_name(file_name, folder_id)
        
        # Determine MIME type
        mime_type = self._get_mime_type(file_path)
        
        try:
            media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)
            
            if existing_files:
                # Files are already sorted by modifiedTime desc (most recent first)
                most_recent_file = existing_files[0]
                existing_file_id = most_recent_file.get('id')
                
                # Update the most recent file
                logger.debug(f"Updating most recent existing file '{file_name}' (ID: {existing_file_id})")
                file = self.drive_service.files().update(
                    fileId=existing_file_id,
                    media_body=media,
                    fields='id, webViewLink',
                    supportsAllDrives=True
                ).execute()
                
                # Trash all other duplicate files (if any)
                if len(existing_files) > 1:
                    duplicate_count = len(existing_files) - 1
                    logger.info(f"Found {len(existing_files)} files with name '{file_name}'. Keeping most recent (ID: {existing_file_id}), trashing {duplicate_count} duplicate(s)")
                    for duplicate_file in existing_files[1:]:  # Skip the first (most recent) one
                        duplicate_id = duplicate_file.get('id')
                        if self.trash_file(duplicate_id):
                            logger.debug(f"Trashed duplicate file '{file_name}' (ID: {duplicate_id})")
                        else:
                            logger.warning(f"Failed to trash duplicate file '{file_name}' (ID: {duplicate_id})")
                else:
                    logger.debug(f"Found 1 existing file '{file_name}', updating it")
            else:
                # Create new file
                file_metadata = {
                    'name': file_name,
                    'parents': [folder_id]
                }
                file = self.drive_service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id, webViewLink',
                    supportsAllDrives=True
                ).execute()
            
            file_id = file.get('id')
            web_view_link = file.get('webViewLink')
            
            action = "Updated" if existing_files else "Uploaded"
            logger.debug(f"{action} file '{file_name}' (ID: {file_id})")
            return {
                'file_id': file_id,
                'web_view_link': web_view_link
            }
        except HttpError as e:
            logger.error(f"Failed to upload file '{file_name}': {e}")
            raise
    
    def upload_multiple_files(self, file_paths: List[Tuple[Path, Optional[str]]], folder_id: str) -> List[Dict[str, str]]:
        """
        Upload multiple files to Google Drive.
        
        Args:
            file_paths: List of tuples (file_path, optional_file_name)
            folder_id: Google Drive folder ID where to upload
            
        Returns:
            List of dicts with 'file_id' and 'web_view_link'
        """
        results = []
        for file_path, file_name in file_paths:
            try:
                result = self.upload_file(file_path, folder_id, file_name)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to upload {file_path}: {e}")
                results.append({'file_id': None, 'web_view_link': None, 'error': str(e)})
        return results
    
    def create_spreadsheet(self, title: str, folder_id: Optional[str] = None) -> Dict[str, str]:
        """
        Create a Google Sheet.
        
        Args:
            title: Title of the spreadsheet
            folder_id: Optional folder ID (defaults to self.drive_folder_id)
            
        Returns:
            Dict with 'spreadsheet_id' and 'url'
        """
        parent_id = folder_id or self.drive_folder_id
        
        try:
            # Create spreadsheet using gspread
            spreadsheet = self.gc.create(title, folder_id=parent_id)
            
            logger.debug(f"Created spreadsheet '{title}' (ID: {spreadsheet.id})")
            return {
                'spreadsheet_id': spreadsheet.id,
                'url': spreadsheet.url
            }
        except APIError as e:
            logger.error(f"Failed to create spreadsheet '{title}': {e}")
            raise
    
    def get_or_create_folder_structure(self, path_parts: List[str]) -> str:
        """
        Get or create a folder structure in Google Drive.
        
        Args:
            path_parts: List of folder names (e.g., ['applications', 'phd', 'John_Doe'])
            
        Returns:
            Final folder ID
        """
        current_folder_id = self.drive_folder_id
        
        for folder_name in path_parts:
            # Check if folder exists
            existing_folder_id = self._find_folder_by_name(folder_name, current_folder_id)
            if existing_folder_id:
                current_folder_id = existing_folder_id
            else:
                current_folder_id = self.create_folder(folder_name, current_folder_id)
        
        return current_folder_id
    
    def _find_folder_by_name(self, folder_name: str, parent_folder_id: str) -> Optional[str]:
        """Find a folder by name in a parent folder"""
        try:
            query = f"name='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.drive_service.files().list(
                q=query,
                fields='files(id, name)',
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            folders = results.get('files', [])
            if folders:
                return folders[0]['id']
            return None
        except HttpError as e:
            if e.resp.status == 404:
                logger.debug(f"Parent folder {parent_folder_id} not accessible when searching for '{folder_name}'")
            else:
                logger.warning(f"Error searching for folder '{folder_name}': {e}")
            return None
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for a file"""
        suffix = file_path.suffix.lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
        }
        return mime_types.get(suffix, 'application/octet-stream')

