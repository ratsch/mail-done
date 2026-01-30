"""
Production Email Classification Prompt - SINGLE SOURCE OF TRUTH

This module contains the EXACT prompt used by production classifier.py.
Benchmark and fine-tuning scripts MUST use this to ensure consistency.

NOTE: This is the GENERIC version for public release. For customized prompts
with specific projects, academic categories (applications, invitations, reviews,
publications, grants), use PROMPTS_DIR overlay.
"""

# Production system prompt
PRODUCTION_SYSTEM_PROMPT = "You are an expert email classifier."

# Production prompt template - Generic version for public release
# Variables: {from_display}, {to_display}, {subject}, {date_str}, {attachment_info},
#            {sender_context}, {sender_email_context}, {email_content}
PRODUCTION_PROMPT_TEMPLATE = """You are an email classification assistant.

CONTEXT:
We receive many emails daily. Most need to be triaged quickly. Some require immediate attention (urgent deadlines), others can wait (newsletters, notifications).

TASK: Classify this email into ONE specific category.

WORK CATEGORIES (human-written work emails):
- work-urgent: Urgent work matter needing immediate attention (explicit deadline/emergency)
- work-colleague: Emails from colleagues needing reply (even brief "congrats" or "thanks")
- work-student: Email from students (questions, submissions, progress updates)
- work-admin: Administrative (HR, IT, department admin, forms)
- work-scheduling: Meeting scheduling requests, Doodle polls, calendar invites (.ics attachments)
- work-other: Other work-related that doesn't fit above

TRAVEL CATEGORIES:
- travel-booking-confirm: Flight/hotel/train booking confirmations
- travel-receipt: Travel expense receipts
- travel-itinerary: Trip itineraries, schedules
- travel-reminder: Travel reminders (check-in, departure)
- travel-transport: Ground transport (taxi, car rental, parking)
- travel-other: Other travel-related

INFORMATION CATEGORIES:
- newsletter-scientific: Scientific newsletters, research updates, journal TOCs, conference announcements
- newsletter-general: General newsletters, announcements, institutional updates
- notification-system: System notifications, automated messages
- notification-technical: System alerts, error notifications, monitoring
- notification-calendar: Calendar invites, meeting notifications
- notification-social: LinkedIn, Twitter, social media notifications
- notification-other: Automated system notifications (Slack bots, GitHub, Jira)
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
- work-scheduling vs work-colleague: Scheduling = ONLY meeting logistics, work-colleague = substantive discussion
- notification-* vs work-*: notifications are AUTOMATED/SYSTEM-GENERATED, work is HUMAN-WRITTEN
- newsletter-* = BROADCAST to many recipients

ANALYSIS REQUIREMENTS:
1. Choose the MOST SPECIFIC category that fits
2. Extract "curated_sender_name": The extracted real-world name of the sender
3. For all dates: Use YYYY-MM-DD format
4. Identify if this is a cold email or followup

SENTIMENT ANALYSIS:
- positive: Praise, congratulations, enthusiasm, good news
- negative: Complaints, criticism, bad news, problems
- neutral: Factual, informational

LANGUAGE & ERROR HANDLING:
- If email body is empty/corrupted: classify as "work-other" with confidence=0.1
- If cannot determine category: use most likely category with low confidence (< 0.5)

---

NOW CLASSIFY THIS EMAIL:

===============================================================================
CURRENT EMAIL TO CLASSIFY:
===============================================================================

Received Date: {date_str}
From: {from_display}
To: {to_display}
Subject: {subject}
Attachments: {attachment_info}
{sender_context}
{sender_email_context}
-------------------------------------------------------------------------------
CURRENT EMAIL BODY:
-------------------------------------------------------------------------------

{email_content}

===============================================================================
"""


def build_production_prompt(
    from_display: str,
    to_display: str,
    subject: str,
    date_str: str,
    email_content: str,
    attachment_info: str = "No attachments",
    sender_context: str = "",
    sender_email_context: str = ""
) -> str:
    """
    Build the production prompt with email details.

    This generates the EXACT same prompt as classifier.py._build_prompt()

    Args:
        from_display: Formatted sender (e.g., "John Smith <john@example.com>")
        to_display: Formatted recipients
        subject: Email subject
        date_str: Formatted date string (YYYY-MM-DD HH:MM:SS)
        email_content: Email body text (truncated if needed)
        attachment_info: Attachment summary string
        sender_context: Optional sender statistics block
        sender_email_context: Optional sender email history block

    Returns:
        Complete prompt string ready for LLM
    """
    return PRODUCTION_PROMPT_TEMPLATE.format(
        from_display=from_display,
        to_display=to_display,
        subject=subject,
        date_str=date_str,
        email_content=email_content,
        attachment_info=attachment_info,
        sender_context=sender_context,
        sender_email_context=sender_email_context
    )
