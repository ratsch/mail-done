"""
Email Classification Prompts - Versioned

Each version is tracked with:
- Date created
- Changes from previous version
- Token count estimate
- Performance notes

NOTE: This is the GENERIC version for public release with basic categories.
For full academic categories (applications, invitations, reviews, publications, grants),
use PROMPTS_DIR overlay to provide customized prompts.

Configuration loaded from config/prompts.yaml (via get_config_path)
"""
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import yaml

from backend.core.paths import get_config_path

# Load configuration using centralized path resolution
_config_path = get_config_path("prompts.yaml")
try:
    if _config_path and _config_path.exists():
        with open(_config_path, 'r') as f:
            _config = yaml.safe_load(f)
        CURRENT_VERSION = _config.get('current_version', 'v1.0')
        AVAILABLE_VERSIONS = _config.get('versions', {})
    else:
        raise FileNotFoundError("prompts.yaml not found")
except Exception as e:
    # Fallback to defaults if prompts.yaml not found
    CURRENT_VERSION = "v1.0"
    AVAILABLE_VERSIONS = {
        "v1.0": {
            "created": "2025-11-08",
            "description": "Default prompt version",
            "estimated_tokens": 1200,
            "status": "production",
            "notes": "Generic configuration - customize via PROMPTS_DIR overlay"
        }
    }


def get_classifier_prompt_v1(email_details: Dict) -> str:
    """
    Version 1.0 - Generic prompt for basic email classification

    NOTE: For academic-specific categories (applications, invitations, reviews,
    publications, grants), use PROMPTS_DIR overlay with customized prompts.

    Token breakdown:
    - System context: 100 tokens
    - Category definitions: 600 tokens
    - Instructions: 300 tokens
    - Scoring guides: 200 tokens

    Total: ~1200 tokens + email content
    """
    prompt = f"""You are an email classification assistant.

EMAIL DETAILS:
{email_details['header']}

EMAIL CONTENT:
{email_details['content']}

---

TASK: Classify this email into ONE specific category.

WORK CATEGORIES (human-written work emails):
- work-urgent: Urgent work matter needing immediate attention (explicit deadline/emergency)
- work-colleague: Emails from colleagues needing reply
- work-student: Email from students (questions, submissions, progress updates)
- work-admin: Administrative (HR, IT, department admin, forms)
- work-scheduling: Meeting scheduling requests, calendar invites
- work-other: Other work-related that doesn't fit above

TRAVEL CATEGORIES:
- travel-booking-confirm: Flight/hotel/train booking confirmations
- travel-receipt: Travel expense receipts
- travel-itinerary: Trip itineraries, schedules
- travel-reminder: Travel reminders (check-in, departure)
- travel-transport: Ground transport (taxi, car rental, parking)
- travel-other: Other travel-related

INFORMATION CATEGORIES:
- newsletter-scientific: Scientific newsletters, research updates, journal TOCs
- newsletter-general: General newsletters, announcements
- notification-system: System notifications
- notification-technical: System alerts, error notifications
- notification-calendar: Calendar invites, meeting notifications
- notification-social: LinkedIn, Twitter, social media notifications
- notification-other: Automated system notifications (bots, GitHub, Jira)
  IMPORTANT: Bot/automated messages = notification-*, not work-*

TRANSACTION CATEGORIES:
- receipt-online: Online purchases
- receipt-travel: Travel bookings, flight/hotel confirmations
- receipt-subscription: Recurring subscriptions (software, services)

PERSONAL CATEGORIES (non-work life):
- personal-family: Family correspondence
- personal-friends: Friends correspondence
- personal-other: Other personal matters

LOW PRIORITY:
- marketing: Promotional emails, advertisements
- spam: Spam, phishing, unwanted solicitations
- social-media: Social media platform notifications

CRITICAL DISTINCTIONS:
- work-colleague vs work-student: Are they your peer or your student?
- work-urgent vs work-colleague: Is there an explicit urgent deadline?
- notification-* vs work-*: notifications are AUTOMATED, work is HUMAN-WRITTEN

ANALYSIS REQUIREMENTS:
1. Choose the MOST SPECIFIC category that fits
2. Rate urgency (1-10), provide reason
3. Extract dates (YYYY-MM-DD) if mentioned
4. For receipts: Extract vendor, amount, currency
5. Needs reply? Suggest response if yes
6. Is this cold email or followup?

Be concise but actionable.
"""
    return prompt


def get_classifier_prompt_v2(email_details: Dict) -> str:
    """
    Version 2.0 - Optimized for cost (minimal token usage)

    Token breakdown:
    - System context: 50 tokens
    - Category list: 300 tokens (condensed)
    - Instructions: 150 tokens (streamlined)

    Total: ~500 tokens + email content
    """
    prompt = f"""Email classifier.

TASK: Classify into ONE category:

WORK: work-urgent, work-colleague, work-student, work-admin, work-scheduling, work-other
TRAVEL: travel-booking-confirm, travel-receipt, travel-itinerary, travel-reminder, travel-transport, travel-other
INFO: newsletter-scientific, newsletter-general, notification-system, notification-technical, notification-calendar, notification-social, notification-other
RECEIPTS: receipt-online, receipt-travel, receipt-subscription
PERSONAL: personal-family, personal-friends, personal-other
OTHER: marketing, spam, social-media

EMAIL:
{email_details['header']}

CONTENT:
{email_details['content']}

---

ANALYSIS:
1. Pick MOST SPECIFIC category
2. Rate urgency (1-10), provide reason
3. Extract dates (YYYY-MM-DD)
4. For receipts: vendor, amount, currency
5. Needs reply? Suggest response if yes
6. Is this cold email or followup?

Be concise.
"""
    return prompt


def get_classifier_prompt_v1_1(email_details: Dict) -> str:
    """
    Version 1.1 - Optimized while keeping all functionality
    Same as v1.0 but with streamlined formatting.
    """
    return get_classifier_prompt_v1(email_details)


def get_classifier_prompt_v1_2(email_details: Dict) -> str:
    """
    Version 1.2 - Multi-category support (same categories as v1.0)
    """
    return get_classifier_prompt_v1(email_details)


def get_classifier_prompt(
    email_details: Dict,
    version: str = CURRENT_VERSION,
    sender_history: Optional[Dict] = None
) -> str:
    """
    Get classification prompt for specified version.

    Args:
        email_details: Dict with 'header' and 'content' keys
        version: Prompt version to use (default: CURRENT_VERSION)
        sender_history: Optional sender history context

    Returns:
        Formatted prompt string
    """
    # Add sender history if available
    if sender_history:
        history_text = f"""
SENDER HISTORY:
- Previous emails: {sender_history.get('email_count', 0)}
- Typical category: {sender_history.get('typical_category', 'unknown')}
- Is frequent sender: {sender_history.get('is_frequent', False)}
"""
        email_details['header'] += history_text

    # Select prompt version
    if version == "v1.0":
        return get_classifier_prompt_v1(email_details)
    elif version == "v1.1":
        return get_classifier_prompt_v1_1(email_details)
    elif version == "v1.2":
        return get_classifier_prompt_v1_2(email_details)
    elif version == "v2.0":
        return get_classifier_prompt_v2(email_details)
    else:
        raise ValueError(f"Unknown prompt version: {version}. Available: {list(AVAILABLE_VERSIONS.keys())}")


def get_version_info(version: str = CURRENT_VERSION) -> Dict:
    """Get metadata about a prompt version."""
    if version not in AVAILABLE_VERSIONS:
        raise ValueError(f"Unknown version: {version}")
    return AVAILABLE_VERSIONS[version]


def compare_versions() -> str:
    """Generate comparison table of all versions."""
    comparison = """
# Classifier Prompt Versions

| Version | Tokens | Cost/1K (GPT-4o) | Cost/1K (mini) | Status | Notes |
|---------|--------|------------------|----------------|--------|-------|
"""
    for ver, info in sorted(AVAILABLE_VERSIONS.items()):
        # Estimate costs
        total_tokens_per_email = info['estimated_tokens'] + 2000  # Add ~2K for email content
        cost_4o = (total_tokens_per_email * 5 / 1_000_000) * 1000  # Input tokens only (rough)
        cost_mini = (total_tokens_per_email * 0.15 / 1_000_000) * 1000

        comparison += f"| {ver} | {info['estimated_tokens']} | ${cost_4o:.2f} | ${cost_mini:.2f} | {info['status']} | {info['notes']} |\n"

    comparison += f"\n**Current Default:** {CURRENT_VERSION}\n"
    return comparison
