"""
Email models following Inbox Zero's type definitions.
Port of inbox-zero's email types to Python with Pydantic.
"""
from pydantic import BaseModel, Field, EmailStr
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict
import os
from pathlib import Path


def _load_categories_from_config() -> dict:
    """Load email categories from config file (supports CONFIG_DIR overlay)."""
    import yaml

    # Determine config directory (same logic as paths.py)
    config_dir_env = os.getenv("CONFIG_DIR", "")
    if config_dir_env and Path(config_dir_env).exists():
        config_dir = Path(config_dir_env)
    else:
        config_dir = Path(__file__).parent.parent.parent.parent / "config"

    # Try to load categories.yaml, fall back to categories.example.yaml
    for filename in ["categories.yaml", "categories.example.yaml"]:
        config_path = config_dir / filename
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f)
                    if data and "categories" in data:
                        return data["categories"]
            except Exception:
                pass

    # Fallback: minimal built-in categories
    return {
        "work-urgent": "work-urgent",
        "work-other": "work-other",
        "application-phd": "application-phd",
        "application-other": "application-other",
        "newsletter": "newsletter",
        "notification": "notification",
        "receipt": "receipt",
        "personal": "personal",
        "spam": "spam",
        "marketing": "marketing",
    }


def _create_email_category_enum():
    """Dynamically create EmailCategory enum from config."""
    categories = _load_categories_from_config()

    # Build enum members: convert "work-urgent" to WORK_URGENT
    members = {}
    for cat_value in categories.values():
        member_name = cat_value.upper().replace("-", "_")
        members[member_name] = cat_value

    # Create and return the enum
    return Enum("EmailCategory", members, type=str)


# Create EmailCategory enum dynamically from config
EmailCategory = _create_email_category_enum()


class SenderType(str, Enum):
    """From inbox-zero's sender categorization"""
    KNOWN = "known"
    NEW = "new"
    COLD_EMAIL = "cold_email"
    AUTOMATED = "automated"
    NEWSLETTER = "newsletter"


class ReplyStatus(str, Enum):
    """Reply tracking statuses"""
    NEEDS_REPLY = "needs_reply"
    AWAITING_REPLY = "awaiting_reply"
    NO_ACTION = "no_action"
    COMPLETED = "completed"


class AppleMailColor(int, Enum):
    """Apple Mail color labels (IMAP X-Color extension)"""
    NONE = 0
    RED = 1
    ORANGE = 2
    YELLOW = 3
    GREEN = 4
    BLUE = 5
    PURPLE = 6
    GRAY = 7


# Port inbox-zero's email model
class Email(BaseModel):
    """Based on inbox-zero's Email type"""
    uid: str = Field(..., description="IMAP UID")
    message_id: Optional[str] = Field(None, description="RFC822 Message-ID")
    thread_id: Optional[str] = Field(None, description="Thread/conversation ID for reply tracking")
    subject: str
    from_address: str
    to_addresses: List[str] = Field(default_factory=list)
    cc_addresses: List[str] = Field(default_factory=list)
    date: datetime
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    body_markdown: Optional[str] = None
    flags: List[str] = Field(default_factory=list)
    folder: str = "INBOX"
    
    class Config:
        json_schema_extra = {
            "example": {
                "uid": "123",
                "message_id": "<abc@example.com>",
                "subject": "Test Email",
                "from_address": "sender@example.com",
                "to_addresses": ["recipient@example.com"],
                "date": "2024-01-01T12:00:00Z",
                "body_text": "Hello world",
                "folder": "INBOX"
            }
        }


class EmailAddress(BaseModel):
    """Email address with optional display name"""
    email: str = Field(..., description="Email address")
    name: Optional[str] = Field(None, description="Display name")
    
    def __str__(self) -> str:
        """String representation"""
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email


class ProcessedEmail(BaseModel):
    """Email after processing (parsing, normalization, attachment extraction)"""
    # Original email data
    uid: str
    message_id: Optional[str]
    thread_id: Optional[str] = Field(None, description="Thread ID for reply tracking")
    references: Optional[str] = Field(None, description="References header for threading chains")
    subject: str
    from_address: str  # Email address only
    from_name: Optional[str] = Field(None, description="Sender display name")
    to_addresses: List[str]  # Email addresses only
    to_names: List[str] = Field(default_factory=list, description="Recipient display names")
    cc_addresses: List[str] = Field(default_factory=list, description="CC email addresses")
    date: datetime
    
    # Processed content
    body_markdown: str = Field(..., description="Email body converted to Markdown")
    body_text: Optional[str] = Field(None, description="Plain text body (if available)")
    attachment_texts: List[str] = Field(
        default_factory=list, 
        description="Extracted text from attachments"
    )
    attachment_info: List['AttachmentInfo'] = Field(
        default_factory=list,
        description="Metadata about attachments"
    )
    
    # Metadata
    sender_domain: str = Field(..., description="Domain extracted from from_address")
    has_attachments: bool = False
    attachment_count: int = 0
    
    # Raw headers (for preprocessing)
    raw_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Raw email headers for preprocessing (X-Original-*, Resent-*, etc.)"
    )
    
    # Preprocessing flags
    was_preprocessed: bool = Field(default=False, description="Was email preprocessed/transformed")
    original_from: Optional[str] = Field(None, description="Original sender before preprocessing")
    
    # IMAP state
    folder: str = "INBOX"
    flags: List[str] = Field(default_factory=list)


class AttachmentInfo(BaseModel):
    """Information about email attachment"""
    filename: str
    content_type: str
    size: int
    extracted_text: Optional[str] = None
    extraction_error: Optional[str] = None
    # Source tracking for consolidated attachments (optional)
    source_email_date: Optional[str] = Field(None, description="Date of email containing this attachment (ISO format)")
    source_email_subject: Optional[str] = Field(None, description="Subject of email containing this attachment")
    source_email_message_id: Optional[str] = Field(None, description="Message-ID of email containing this attachment (for tracking across systems)")
    is_from_current_email: bool = Field(True, description="Whether this attachment is from the current email vs prior emails")


class EmailAction(BaseModel):
    """Action to perform on an email (from inbox-zero)"""
    type: str = Field(..., description="Action type: move, label, color, archive, forward")
    folder: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    color: Optional[AppleMailColor] = None
    forward_to: Optional[str] = None  # Email address as string
    target_account: Optional[str] = None  # For cross-account moves (e.g., 'work', 'personal')
    
    @property
    def is_valid(self) -> bool:
        """Check if action has required fields"""
        if self.type == "move" and not self.folder:
            return False
        if self.type == "forward" and not self.forward_to:
            return False
        return True
    
    def model_post_init(self, __context) -> None:
        """Pydantic v2 post-initialization validation"""
        if self.type == "move" and not self.folder:
            raise ValueError("folder is required for move action")
        if self.type == "forward" and not self.forward_to:
            raise ValueError("forward_to is required for forward action")


class ActionResult(BaseModel):
    """Result of executing an action"""
    success: bool
    dry_run: bool = False
    description: Optional[str] = None
    error: Optional[str] = None
    reversible: bool = False
    undo_data: Optional[dict] = None


class ClassificationResult(BaseModel):
    """Result of email classification (inbox-zero pattern)"""
    category: EmailCategory
    sender_type: SenderType
    reply_status: ReplyStatus
    is_urgent: bool = False
    confidence: float = Field(..., ge=0, le=1)
    project: Optional[str] = None
    reasoning: str = Field(..., description="Explanation of classification")
    action: EmailAction
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "work",
                "sender_type": "known",
                "reply_status": "needs_reply",
                "is_urgent": True,
                "confidence": 0.95,
                "reasoning": "Email from colleague about urgent project deadline",
                "action": {
                    "type": "color",
                    "color": 1
                }
            }
        }

