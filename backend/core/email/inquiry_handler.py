"""
Inquiry Handler for #application #info emails.

This module handles inquiry emails separately from applications:
1. Detects #info + #application tags in subject (Stage 0)
2. Validates with lightweight AI that it's a genuine inquiry
3. Classifies inquiry type (phd, postdoc, internship, etc.)
4. Extracts sender name for greeting
5. Generates draft response from templates
6. Creates draft in IMAP Drafts folder
7. Moves original email to MD/Applications/Inquiries

See docs/INQUIRY_HANDLER_PLAN.md for full design.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from backend.core.prompt_loader import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class InquiryClassificationResult:
    """Result from AI inquiry classification."""
    is_valid_inquiry: bool
    full_name: Optional[str]
    inquiry_types: List[str]
    confidence: float
    reasoning: str


def is_inquiry_email(subject: str) -> bool:
    """
    Stage 0: Check if email should be handled as an inquiry.
    
    Detects #info combined with one of:
    - #application
    - #phd
    - #postdoc
    
    Args:
        subject: Email subject line
        
    Returns:
        True if #info is present along with a qualifying tag
    """
    subject_lower = subject.lower()
    has_info = "#info" in subject_lower
    
    if not has_info:
        return False
    
    # Qualifying tags that indicate an inquiry when combined with #info
    qualifying_tags = ["#application", "#phd", "#postdoc"]
    
    for tag in qualifying_tags:
        if tag in subject_lower:
            logger.info(f"Inquiry detected: subject contains #info and {tag}")
            return True
    
    return False


def extract_inquiry_type_from_tags(subject: str) -> List[str]:
    """
    Extract inquiry type from subject line tags.
    
    Tags like #phd, #postdoc, #intern are used as hints.
    
    Args:
        subject: Email subject line
        
    Returns:
        List of inquiry types extracted from tags
    """
    TAG_MAPPING = {
        "#phd": "phd",
        "#postdoc": "postdoc",
        "#intern": "internship",
        "#internship": "internship",
        "#thesis": "thesis",
        "#visit": "visit",
        "#visiting": "visit",
        "#direct-doctorate": "direct_doctorate",
    }
    
    subject_lower = subject.lower()
    inquiry_types = []
    
    for tag, inquiry_type in TAG_MAPPING.items():
        if tag in subject_lower and inquiry_type not in inquiry_types:
            inquiry_types.append(inquiry_type)
    
    return inquiry_types


def build_inquiry_classification_prompt(subject: str, body: str, from_display_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Build the AI prompt for inquiry classification.

    Args:
        subject: Email subject line
        body: Email body text
        from_display_name: Display name from email header (optional)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Default fallback prompts
    default_system = """You are classifying an email that has #info and #application tags in the subject line.
Your job is to:
1. Verify this is a valid inquiry about academic positions from an applicant themselves
2. Determine what type(s) of position they are asking about
3. Extract the sender's full name

Respond ONLY with valid JSON."""

    # Try to load from config, fall back to default
    system_prompt = get_prompt("inquiry.classification.system_prompt", default=default_system)

    # User prompt with email-specific details
    user_prompt = get_prompt(
        "inquiry.classification.user_template",
        default="",
        subject=subject,
        from_display_name=from_display_name or 'Not provided',
        body=body[:3000]  # Limit body length for cost control
    )

    # If user_template not in config, use default inline template
    if not user_prompt:
        user_prompt = f"""Analyze this email and determine if it is a VALID INQUIRY.

A VALID INQUIRY must meet ALL of these criteria:
- The sender is inquiring about opportunities FOR THEMSELVES (not on behalf of someone else)
- The sender is asking about one of the position types listed below
- The email is asking for information, not submitting a formal application

Set is_valid_inquiry=FALSE if ANY of these apply:
- A colleague, professor, or third party is asking about opportunities for someone else
- The email is spam or irrelevant
- The email is a complaint or follow-up about a previous application
- The email is a COMPLETE formal application with CV, cover letter, AND transcripts
- The sender is asking about something unrelated (courses, collaborations, data requests)
- The inquiry type does not match any of the position types below

Position types (use ONLY these exact values):
- internship: Summer/short-term research internship (2-3 months)
- thesis: Bachelor's or Master's thesis project
- direct_doctorate: Direct doctorate program
- phd: Doctoral/PhD position (typically requires M.Sc.)
- postdoc: Postdoctoral research position (requires PhD)
- visit: Visiting researcher or visiting PhD student

Disambiguation:
- "research project" for credit → thesis
- "research project" for work experience → internship
- PhD student from elsewhere visiting → visit (not phd)

Extract:
- The sender's full name (first and last name)
- Which position type(s) they are asking about

Email subject: {subject}
From header display name: {from_display_name or 'Not provided'}
Email body:
{body[:3000]}"""

    return system_prompt, user_prompt


def parse_inquiry_classification_response(response_text: str) -> InquiryClassificationResult:
    """
    Parse the AI response into a structured result.
    
    Args:
        response_text: Raw JSON response from AI
        
    Returns:
        InquiryClassificationResult
    """
    import json
    
    try:
        data = json.loads(response_text)
        
        # Validate and extract fields (handle multiple field name formats)
        is_valid = data.get("is_valid_inquiry", data.get("is_valid", False))
        
        # Handle various name field formats
        full_name = (
            data.get("full_name") or 
            data.get("sender_full_name") or 
            data.get("name") or 
            data.get("sender_name")
        )
        
        # Handle various type field formats
        inquiry_types = (
            data.get("inquiry_types") or 
            data.get("position_types") or 
            data.get("types") or 
            []
        )
        
        confidence = data.get("confidence", 0.85 if is_valid else 0.0)
        reasoning = data.get("reasoning", data.get("reason", ""))
        
        # Validate inquiry_types
        valid_types = {"internship", "thesis", "direct_doctorate", "phd", "postdoc", "visit"}
        inquiry_types = [t for t in inquiry_types if t in valid_types]
        
        # Limit to 2 types
        if len(inquiry_types) > 2:
            inquiry_types = inquiry_types[:2]
        
        return InquiryClassificationResult(
            is_valid_inquiry=is_valid,
            full_name=full_name,
            inquiry_types=inquiry_types,
            confidence=confidence,
            reasoning=reasoning
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse inquiry classification response: {e}")
        return InquiryClassificationResult(
            is_valid_inquiry=False,
            full_name=None,
            inquiry_types=[],
            confidence=0.0,
            reasoning=f"Failed to parse AI response: {e}"
        )


def validate_extracted_name(name: Optional[str]) -> Optional[str]:
    """
    Validate an extracted name.
    
    Rejects generic phrases, device signatures, and invalid names.
    
    Args:
        name: Extracted name to validate
        
    Returns:
        Validated name or None if invalid
    """
    if not name:
        return None
    
    name = name.strip()
    
    # Reject empty or too short
    if len(name) < 3:
        return None
    
    # Reject generic phrases
    invalid_patterns = [
        "sent from", "iphone", "android", "outlook", "gmail", "mail",
        "best regards", "sincerely", "cheers", "thanks", "thank you",
        "kind regards", "regards", "with regards", "dear", "professor",
        "prospective", "applicant", "student", "researcher"
    ]
    name_lower = name.lower()
    if any(p in name_lower for p in invalid_patterns):
        return None
    
    # Require at least 2 words (first + last name)
    if len(name.split()) < 2:
        return None
    
    # Reject if contains numbers or special characters (except hyphens, apostrophes, dots)
    if re.search(r'[0-9@#$%^&*()+=\[\]{}|\\/<>]', name):
        return None
    
    return name


def get_inquiry_category(inquiry_types: List[str]) -> str:
    """
    Get the category string for an inquiry.

    Args:
        inquiry_types: List of inquiry types

    Returns:
        Category string like "inquiry-phd" or "inquiry-multiple"
    """
    if not inquiry_types:
        return "inquiry-unknown"
    if len(inquiry_types) == 1:
        return f"inquiry-{inquiry_types[0]}"
    return "inquiry-multiple"


def format_inquiry_summary(inquiry_types: List[str]) -> str:
    """
    Format inquiry types for the greeting.
    
    Args:
        inquiry_types: List of inquiry types
        
    Returns:
        Human-readable summary like "PhD positions and visiting opportunities"
    """
    type_names = {
        "internship": "internship opportunities",
        "thesis": "thesis projects",
        "direct_doctorate": "the direct doctorate program",
        "phd": "PhD positions",
        "postdoc": "postdoctoral positions",
        "visit": "visiting researcher opportunities",
    }
    
    if not inquiry_types:
        return "opportunities in our research group"
    
    if len(inquiry_types) == 1:
        return type_names.get(inquiry_types[0], "opportunities")
    
    # Multiple types
    named = [type_names.get(t, t) for t in inquiry_types]
    return " and ".join(named)


# =============================================================================
# Template Loader and Draft Generator
# =============================================================================

class InquiryTemplateLoader:
    """Load and manage inquiry response templates from YAML."""
    
    def __init__(self, templates_path: str = None):
        """
        Initialize template loader.
        
        Args:
            templates_path: Path to inquiry_templates.yaml
        """
        import os
        import yaml
        from pathlib import Path

        if templates_path is None:
            # Check CONFIG_DIR overlay first, then default location
            config_dir = os.environ.get('CONFIG_DIR')
            if config_dir:
                templates_path = Path(config_dir) / "inquiry_templates.yaml"
            else:
                templates_path = Path(__file__).parent.parent.parent.parent / "config" / "inquiry_templates.yaml"
        
        self.templates_path = Path(templates_path)
        self._templates = None
    
    @property
    def templates(self) -> Dict[str, Any]:
        """Lazy load templates."""
        if self._templates is None:
            self._load_templates()
        return self._templates
    
    def _load_templates(self):
        """Load templates from YAML file."""
        import yaml
        
        if not self.templates_path.exists():
            raise FileNotFoundError(f"Templates file not found: {self.templates_path}")
        
        with open(self.templates_path, 'r', encoding='utf-8') as f:
            self._templates = yaml.safe_load(f)
        
        logger.info(f"Loaded inquiry templates from {self.templates_path}")
    
    def get_greeting(self) -> str:
        """Get the greeting template."""
        return self.templates.get("greeting", "Dear {full_name},\n\nThank you for your inquiry.")
    
    def get_block(self, inquiry_type: str) -> Optional[str]:
        """Get a specific inquiry type block."""
        blocks = self.templates.get("blocks", {})
        return blocks.get(inquiry_type)
    
    def get_disclaimer(self) -> str:
        """Get the disclaimer template."""
        return self.templates.get("disclaimer", "")
    
    def get_signature(self) -> str:
        """Get the signature template."""
        return self.templates.get("signature", "\nBest regards")
    
    def get_cc_recipients(self) -> List[str]:
        """Get CC recipient list."""
        return self.templates.get("cc_recipients", [])


@dataclass
class InquiryDraft:
    """A generated draft response for an inquiry email."""
    to_address: str
    cc_addresses: List[str]
    subject: str
    body: str
    in_reply_to: str
    references: str


class InquiryDraftGenerator:
    """Generate draft responses for inquiry emails."""
    
    def __init__(self, template_loader: InquiryTemplateLoader = None):
        """
        Initialize draft generator.
        
        Args:
            template_loader: Template loader instance
        """
        self.template_loader = template_loader or InquiryTemplateLoader()
    
    def generate_draft(
        self,
        to_address: str,
        original_subject: str,
        original_message_id: str,
        full_name: str,
        inquiry_types: List[str]
    ) -> InquiryDraft:
        """
        Generate a draft response for an inquiry.
        
        Args:
            to_address: Recipient email address
            original_subject: Original email subject
            original_message_id: Original email's Message-ID for threading
            full_name: Extracted sender name (or "prospective applicant")
            inquiry_types: List of inquiry types to include
            
        Returns:
            InquiryDraft ready for IMAP creation
        """
        # Build body
        body_parts = []
        
        # Greeting
        inquiry_summary = format_inquiry_summary(inquiry_types)
        greeting = self.template_loader.get_greeting()
        greeting = greeting.format(full_name=full_name, inquiry_summary=inquiry_summary)
        body_parts.append(greeting)
        
        # Type-specific blocks (max 2)
        for inquiry_type in inquiry_types[:2]:
            block = self.template_loader.get_block(inquiry_type)
            if block:
                body_parts.append(block)
        
        # Disclaimer
        disclaimer = self.template_loader.get_disclaimer()
        if disclaimer:
            body_parts.append(disclaimer)
        
        # Signature
        signature = self.template_loader.get_signature()
        if signature:
            body_parts.append(signature)
        
        # Combine
        body = "\n".join(body_parts)
        
        # Build subject (add Re: if not already present)
        reply_subject = original_subject
        if not reply_subject.lower().startswith("re:"):
            reply_subject = f"Re: {reply_subject}"
        
        # Get CC recipients
        cc_addresses = self.template_loader.get_cc_recipients()
        
        return InquiryDraft(
            to_address=to_address,
            cc_addresses=cc_addresses,
            subject=reply_subject,
            body=body,
            in_reply_to=original_message_id,
            references=original_message_id
        )
