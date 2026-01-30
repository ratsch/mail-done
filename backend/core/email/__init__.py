"""Email processing module"""
from .models import (
    Email,
    ProcessedEmail,
    EmailCategory,
    SenderType,
    ReplyStatus,
    AppleMailColor,
    EmailAction,
    ActionResult,
    ClassificationResult,
    AttachmentInfo,
)
from .processor import EmailProcessor
from .imap_monitor import IMAPMonitor, IMAPConfig
from .smtp_sender import SMTPSender
from .email_handler import EmailHandler
from .annotations import AnnotationManager, EmailMetadata, EmailNote, IMAPKeywords
from .vip_manager import VIPManager, VIPInfo

__all__ = [
    "Email",
    "ProcessedEmail",
    "EmailCategory",
    "SenderType",
    "ReplyStatus",
    "AppleMailColor",
    "EmailAction",
    "ActionResult",
    "ClassificationResult",
    "AttachmentInfo",
    "EmailProcessor",
    "IMAPMonitor",
    "IMAPConfig",
    "SMTPSender",
    "EmailHandler",
    "AnnotationManager",
    "EmailMetadata",
    "EmailNote",
    "IMAPKeywords",
    "VIPManager",
    "VIPInfo",
]

