"""
Reply generation module for Phase 3

Generates draft responses for common email categories.
"""
from backend.core.replies.templates import ReplyTemplates
from backend.core.replies.ai_generator import AIReplyGenerator
from backend.core.replies.draft_manager import DraftManager

__all__ = ['ReplyTemplates', 'AIReplyGenerator', 'DraftManager']

