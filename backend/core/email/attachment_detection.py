"""
Centralized attachment detection logic.

This module provides consistent attachment detection across all email processing:
- reprocess_applications.py (consolidation, reference letters)
- processor.py (EmailProcessor)
- Any other attachment handling code

IMPORTANT: All attachment detection should use these functions to ensure consistency.
"""

from email.message import Message
from typing import Set, Optional
import logging

logger = logging.getLogger(__name__)

# Document content types we consider as meaningful attachments
# These are file types that typically contain text/content worth extracting
DOCUMENT_CONTENT_TYPES: Set[str] = frozenset({
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'text/plain',
    'text/csv',
    'application/rtf',
    'text/rtf',
})

# Content type prefixes that indicate document types (for partial matching)
DOCUMENT_TYPE_PREFIXES = (
    'application/pdf',
    'application/vnd.openxmlformats-officedocument',
    'application/msword',
    'application/vnd.ms-excel',
    'application/vnd.ms-powerpoint',
    'text/plain',
    'text/csv',
    'application/rtf',
)

# File extensions that indicate documents
DOCUMENT_EXTENSIONS = frozenset({
    '.pdf', '.doc', '.docx', '.txt', '.rtf', 
    '.xlsx', '.xls', '.pptx', '.ppt', '.csv'
})


def is_attachment_part(part: Message) -> bool:
    """
    Determine if an email MIME part is an attachment.
    
    This uses broader detection to handle:
    - Standard attachments (Content-Disposition: attachment)
    - Inline attachments with filenames (common in Apple Mail, forwarded emails)
    - Document content types even without explicit disposition
    
    Args:
        part: An email.message.Message part from msg.walk()
        
    Returns:
        True if this part should be treated as an attachment
    """
    # Skip multipart containers - they're just wrappers
    if part.get_content_maintype() == 'multipart':
        return False
    
    disposition = part.get_content_disposition()
    content_type = part.get_content_type()
    filename = part.get_filename()
    
    # Detection logic (order matters for clarity):
    
    # 1. Explicit attachment disposition - always an attachment
    if disposition == 'attachment':
        return True
    
    # 2. Inline with a filename - treat as attachment (common in forwarded emails)
    if disposition == 'inline' and filename:
        return True
    
    # 3. Has a filename and is not a multipart - treat as attachment
    if filename and not content_type.startswith('multipart/'):
        return True
    
    # 4. Document content type with payload - treat as attachment
    #    (handles cases where disposition is missing but content is a document)
    if any(content_type.startswith(prefix) for prefix in DOCUMENT_TYPE_PREFIXES):
        # Only if it has actual content
        try:
            payload = part.get_payload(decode=True)
            if payload:
                return True
        except:
            pass
    
    return False


def is_document_type(filename: Optional[str] = None, content_type: Optional[str] = None) -> bool:
    """
    Check if a file is a document type worth extracting text from.
    
    Args:
        filename: The attachment filename (optional)
        content_type: The MIME content type (optional)
        
    Returns:
        True if this is a document type we should process
    """
    # Check by filename extension
    if filename:
        filename_lower = filename.lower()
        for ext in DOCUMENT_EXTENSIONS:
            if filename_lower.endswith(ext):
                return True
    
    # Check by content type
    if content_type:
        content_type_lower = content_type.lower()
        
        # Exact match
        if content_type_lower in DOCUMENT_CONTENT_TYPES:
            return True
        
        # Prefix match (handles parameters like charset)
        for prefix in DOCUMENT_TYPE_PREFIXES:
            if content_type_lower.startswith(prefix):
                return True
        
        # Keyword matching for edge cases
        if any(kw in content_type_lower for kw in ['pdf', 'word', 'excel', 'powerpoint', 'spreadsheet', 'presentation', 'rtf']):
            return True
    
    return False


def get_attachment_parts(msg: Message) -> list:
    """
    Extract all attachment parts from an email message.
    
    Args:
        msg: Parsed email message (from email.message_from_bytes)
        
    Returns:
        List of (part, filename, content_type) tuples for each attachment
    """
    attachments = []
    
    for part in msg.walk():
        if is_attachment_part(part):
            filename = part.get_filename()
            content_type = part.get_content_type()
            attachments.append((part, filename, content_type))
    
    return attachments

