"""
Document Retrieval Service

Retrieves document content from various origins:
- Local filesystem
- Network mounts (NFS, SMB)
- SSH/SCP remote access
- IMAP email attachments

The service attempts to retrieve from the primary origin first,
falling back to secondary origins if needed.
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from uuid import UUID

from backend.core.documents.config import get_host_config, HostConfig
from backend.core.documents.models import Document, DocumentOrigin
from backend.core.documents.repository import DocumentRepository

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Error retrieving document content."""
    pass


class OriginNotAccessibleError(RetrievalError):
    """Origin is not accessible."""
    pass


class DocumentRetrievalService:
    """
    Service for retrieving document content from origins.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        temp_dir: Optional[Path] = None,
    ):
        """
        Initialize retrieval service.

        Args:
            repository: Document repository
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.repository = repository
        self.temp_dir = temp_dir or Path(tempfile.gettempdir())

    async def get_content(
        self,
        document: Document,
        origin_index: int = 0,
        fallback: bool = True,
    ) -> bytes:
        """
        Retrieve document binary content.

        Tries the specified origin first, falling back to others if needed.

        Args:
            document: Document to retrieve
            origin_index: Index of origin to try first (0 = primary)
            fallback: Whether to try other origins if first fails

        Returns:
            Document binary content

        Raises:
            RetrievalError: If content cannot be retrieved from any origin
        """
        origins = await self.repository.get_origins(document.id, include_deleted=False)

        if not origins:
            raise RetrievalError(f"No origins available for document {document.id}")

        # Sort by primary first, then by index
        origins = sorted(origins, key=lambda o: (not o.is_primary, origins.index(o)))

        # Try specified origin first if valid
        if 0 <= origin_index < len(origins):
            try_order = [origins[origin_index]] + [o for i, o in enumerate(origins) if i != origin_index]
        else:
            try_order = origins

        errors = []
        for origin in try_order:
            try:
                content = await self._retrieve_from_origin(origin)
                return content
            except Exception as e:
                errors.append(f"{origin.origin_type}:{origin.origin_host}: {e}")
                if not fallback:
                    break
                logger.warning(f"Failed to retrieve from {origin.origin_host}: {e}")
                continue

        raise RetrievalError(
            f"Could not retrieve document {document.id} from any origin. Errors: {errors}"
        )

    async def _retrieve_from_origin(self, origin: DocumentOrigin) -> bytes:
        """
        Retrieve content from a specific origin.

        Args:
            origin: Origin to retrieve from

        Returns:
            Document binary content

        Raises:
            RetrievalError: If retrieval fails
        """
        if origin.origin_type == "folder":
            return await self._retrieve_from_filesystem(origin)
        elif origin.origin_type == "email_attachment":
            return await self._retrieve_from_imap(origin)
        elif origin.origin_type == "google_drive":
            raise RetrievalError("Google Drive retrieval not yet implemented")
        else:
            raise RetrievalError(f"Unknown origin type: {origin.origin_type}")

    async def _retrieve_from_filesystem(self, origin: DocumentOrigin) -> bytes:
        """
        Retrieve from local filesystem or network mount.

        For SSH hosts, uses SCP to fetch the file.

        Args:
            origin: Filesystem origin

        Returns:
            File content as bytes
        """
        host_config = get_host_config(origin.origin_host)
        if not host_config:
            # Assume localhost if no config
            host_config = HostConfig(
                name=origin.origin_host,
                type="local",
                mount_point="/"
            )

        # Build full path
        if origin.origin_path:
            full_path = Path(origin.origin_path) / origin.origin_filename
        else:
            full_path = Path(origin.origin_filename)

        # For SSH hosts, use SCP
        if host_config.type == "ssh":
            return await self._retrieve_via_ssh(host_config, full_path)

        # For local/network mount, read directly
        # Prepend mount point if specified
        if host_config.mount_point and not str(full_path).startswith(host_config.mount_point):
            full_path = Path(host_config.mount_point) / full_path

        return await self._read_local_file(full_path)

    async def _read_local_file(self, path: Path) -> bytes:
        """
        Read a local file.

        Args:
            path: Path to file

        Returns:
            File content as bytes

        Raises:
            OriginNotAccessibleError: If file cannot be read
        """
        try:
            # Use asyncio to avoid blocking
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, path.read_bytes)
            return content
        except FileNotFoundError:
            raise OriginNotAccessibleError(f"File not found: {path}")
        except PermissionError:
            raise OriginNotAccessibleError(f"Permission denied: {path}")
        except OSError as e:
            raise OriginNotAccessibleError(f"Cannot read file {path}: {e}")

    async def _retrieve_via_ssh(self, host_config: HostConfig, remote_path: Path) -> bytes:
        """
        Retrieve file from remote host via SCP.

        Args:
            host_config: SSH host configuration
            remote_path: Path on remote host

        Returns:
            File content as bytes

        Raises:
            OriginNotAccessibleError: If SCP fails
        """
        # Create temp file for receiving
        temp_file = self.temp_dir / f"scp_{os.getpid()}_{hash(str(remote_path))}"

        try:
            # Build SCP command
            scp_cmd = ["scp", "-q"]

            # Add port if non-standard
            if host_config.ssh_port != 22:
                scp_cmd.extend(["-P", str(host_config.ssh_port)])

            # Add key file if specified
            if host_config.ssh_key_path:
                scp_cmd.extend(["-i", host_config.ssh_key_path])

            # Add source and destination
            remote_spec = f"{host_config.ssh_user}@{host_config.ssh_host}:{remote_path}"
            scp_cmd.extend([remote_spec, str(temp_file)])

            # Execute SCP
            loop = asyncio.get_event_loop()
            process = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    scp_cmd,
                    capture_output=True,
                    timeout=60,  # 60 second timeout
                )
            )

            if process.returncode != 0:
                error_msg = process.stderr.decode() if process.stderr else "Unknown error"
                raise OriginNotAccessibleError(f"SCP failed: {error_msg}")

            # Read content from temp file
            content = temp_file.read_bytes()
            return content

        except subprocess.TimeoutExpired:
            raise OriginNotAccessibleError(f"SCP timed out for {remote_path}")
        except Exception as e:
            raise OriginNotAccessibleError(f"SCP error: {e}")
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()

    async def _retrieve_from_imap(self, origin: DocumentOrigin) -> bytes:
        """
        Retrieve email attachment from IMAP.

        Uses the existing attachment extraction infrastructure.

        Args:
            origin: Email attachment origin

        Returns:
            Attachment content as bytes

        Raises:
            OriginNotAccessibleError: If attachment cannot be retrieved
        """
        if not origin.email_id:
            raise OriginNotAccessibleError("Email ID not set for email_attachment origin")

        if origin.attachment_index is None:
            raise OriginNotAccessibleError("Attachment index not set for email_attachment origin")

        try:
            # Import here to avoid circular imports
            from backend.core.email.attachment_extractor import AttachmentExtractor

            extractor = AttachmentExtractor()
            content = await extractor.get_attachment_content(
                email_id=origin.email_id,
                attachment_index=origin.attachment_index,
            )

            if content is None:
                raise OriginNotAccessibleError("Attachment not found")

            return content

        except ImportError:
            raise OriginNotAccessibleError("Attachment extractor not available")
        except Exception as e:
            raise OriginNotAccessibleError(f"Failed to retrieve attachment: {e}")

    async def check_origin_accessible(self, origin: DocumentOrigin) -> bool:
        """
        Check if an origin is accessible.

        For filesystem origins, checks if file exists and is readable.
        For email origins, checks if email and attachment exist.

        Args:
            origin: Origin to check

        Returns:
            True if accessible, False otherwise
        """
        try:
            if origin.origin_type == "folder":
                return await self._check_filesystem_accessible(origin)
            elif origin.origin_type == "email_attachment":
                return await self._check_imap_accessible(origin)
            elif origin.origin_type == "google_drive":
                return False  # Not implemented
            else:
                return False
        except Exception:
            return False

    async def _check_filesystem_accessible(self, origin: DocumentOrigin) -> bool:
        """Check if filesystem origin is accessible."""
        host_config = get_host_config(origin.origin_host)

        if not host_config:
            # Assume localhost
            host_config = HostConfig(
                name=origin.origin_host,
                type="local",
                mount_point="/"
            )

        # Build full path
        if origin.origin_path:
            full_path = Path(origin.origin_path) / origin.origin_filename
        else:
            full_path = Path(origin.origin_filename)

        # For SSH, we'd need to test connectivity
        if host_config.type == "ssh":
            return await self._check_ssh_accessible(host_config, full_path)

        # For local/network mount
        if host_config.mount_point and not str(full_path).startswith(host_config.mount_point):
            full_path = Path(host_config.mount_point) / full_path

        return full_path.exists() and full_path.is_file() and os.access(full_path, os.R_OK)

    async def _check_ssh_accessible(self, host_config: HostConfig, remote_path: Path) -> bool:
        """Check if SSH file is accessible using test command."""
        try:
            ssh_cmd = ["ssh", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]

            if host_config.ssh_port != 22:
                ssh_cmd.extend(["-p", str(host_config.ssh_port)])

            if host_config.ssh_key_path:
                ssh_cmd.extend(["-i", host_config.ssh_key_path])

            ssh_cmd.append(f"{host_config.ssh_user}@{host_config.ssh_host}")
            ssh_cmd.append(f"test -r {remote_path}")

            loop = asyncio.get_event_loop()
            process = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    timeout=10,
                )
            )

            return process.returncode == 0

        except Exception:
            return False

    async def _check_imap_accessible(self, origin: DocumentOrigin) -> bool:
        """Check if email attachment is accessible."""
        if not origin.email_id or origin.attachment_index is None:
            return False

        try:
            # Check if email exists in database
            from backend.core.database.repository import EmailRepository
            from backend.core.database import get_db

            db = next(get_db())
            repo = EmailRepository(db)
            email = await repo.get_by_id(origin.email_id)

            if not email:
                return False

            # Check if attachment index is valid
            if not email.attachments:
                return False

            return origin.attachment_index < len(email.attachments)

        except Exception:
            return False

    async def verify_all_origins(self, document: Document) -> dict:
        """
        Verify accessibility of all origins for a document.

        Args:
            document: Document to check

        Returns:
            Dict mapping origin_id to accessibility status
        """
        origins = await self.repository.get_origins(document.id, include_deleted=True)
        results = {}

        for origin in origins:
            accessible = await self.check_origin_accessible(origin)
            results[str(origin.id)] = {
                "accessible": accessible,
                "type": origin.origin_type,
                "host": origin.origin_host,
                "is_deleted": origin.is_deleted,
            }

            # Update last_verified_at
            if accessible and not origin.is_deleted:
                await self.repository.update_origin_verified(origin.id)

        return results

    def get_content_type(self, document: Document) -> str:
        """
        Get MIME content type for document.

        Args:
            document: Document

        Returns:
            MIME type string
        """
        if document.mime_type:
            return document.mime_type

        # Fallback based on extension
        if document.original_filename:
            ext = Path(document.original_filename).suffix.lower()
            mime_map = {
                ".pdf": "application/pdf",
                ".doc": "application/msword",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".xls": "application/vnd.ms-excel",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".ppt": "application/vnd.ms-powerpoint",
                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".txt": "text/plain",
                ".csv": "text/csv",
                ".md": "text/markdown",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".tiff": "image/tiff",
                ".bmp": "image/bmp",
            }
            return mime_map.get(ext, "application/octet-stream")

        return "application/octet-stream"
