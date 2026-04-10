"""
Portal-specific contact/info request methods.

Each portal has different ways to request more information:
- Direct email to agent
- Web form submission (Playwright)
- Login required + contact form
- Portal messaging system

This module centralizes the contact methods and German email templates.
"""

import logging
import os
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContactMethod:
    """How to contact an agent about a listing."""
    method: str            # "email", "web_form", "portal_message", "manual"
    agent_email: Optional[str] = None
    agent_phone: Optional[str] = None
    portal_url: Optional[str] = None   # URL of contact form
    needs_login: bool = False
    template_key: str = "general"      # Which email template to use


# =============================================================================
# German email/message templates
# =============================================================================

TEMPLATES = {
    "general": {
        "subject": "Anfrage zu Ihrem Objekt{ref}",
        "body": """Guten Tag

Ich interessiere mich für Ihr Objekt{title_line}.

Könnten Sie mir bitte die Verkaufsunterlagen zukommen lassen?

Insbesondere wäre ich an folgenden Informationen interessiert:
{missing_info}

Freundliche Grüsse
Prof. Dr. Gunnar Rätsch
ETH Zürich""",
    },

    "request_price": {
        "subject": "Preisanfrage{ref}",
        "body": """Guten Tag

Ich interessiere mich für Ihr Objekt{title_line}.

Könnten Sie mir bitte den Verkaufspreis mitteilen?

Freundliche Grüsse
Prof. Dr. Gunnar Rätsch
ETH Zürich""",
    },

    "request_documentation": {
        "subject": "Anfrage Verkaufsunterlagen{ref}",
        "body": """Guten Tag

Ich interessiere mich für Ihr Objekt{title_line} und würde gerne die vollständigen Verkaufsunterlagen erhalten.

Insbesondere wäre ich an folgenden Dokumenten interessiert:
- Grundrissplan
- Grundbuchauszug
- Nebenkostenabrechnung der letzten 2 Jahre
- STWE-Reglement (falls Stockwerkeigentum)
- GEAK / Energieausweis

Freundliche Grüsse
Prof. Dr. Gunnar Rätsch
ETH Zürich""",
    },

    "request_viewing": {
        "subject": "Besichtigungsanfrage{ref}",
        "body": """Guten Tag

Ich interessiere mich für Ihr Objekt{title_line} und würde es gerne besichtigen.

Wann wäre ein Termin möglich?

Freundliche Grüsse
Prof. Dr. Gunnar Rätsch
ETH Zürich""",
    },

    "klosters_interest": {
        "subject": "Interesse an Ihrem Objekt in Klosters{ref}",
        "body": """Guten Tag

Ich bin auf der Suche nach einer Ferienimmobilie in Klosters und interessiere mich für Ihr Objekt{title_line}.

Könnten Sie mir bitte die Verkaufsunterlagen sowie den Verkaufspreis zukommen lassen?

Insbesondere interessiert mich:
- Ob das Objekt als Zweitwohnung nutzbar ist
- Zustand und allfälliger Renovationsbedarf
- Jährliche Nebenkosten
{missing_info}

Freundliche Grüsse
Prof. Dr. Gunnar Rätsch
ETH Zürich""",
    },
}


def format_message(
    template_key: str,
    listing_title: str = "",
    listing_ref: str = "",
    missing_fields: list = None,
) -> Dict[str, str]:
    """
    Format a contact message from template.

    Returns: {"subject": "...", "body": "..."}
    """
    template = TEMPLATES.get(template_key, TEMPLATES["general"])

    ref = f" (Ref: {listing_ref})" if listing_ref else ""
    title_line = f" «{listing_title}»" if listing_title else ""

    missing_info = ""
    if missing_fields:
        missing_info = "\n".join(f"- {f}" for f in missing_fields)
    else:
        missing_info = "- Vollständige Verkaufsdokumentation"

    subject = template["subject"].format(ref=ref)
    body = template["body"].format(
        ref=ref,
        title_line=title_line,
        missing_info=missing_info,
    )

    return {"subject": subject, "body": body}


# =============================================================================
# Portal-specific contact configuration
# =============================================================================

PORTAL_CONTACT_CONFIG = {
    "homegate": {
        "method": "web_form",
        "needs_login": False,
        "notes": "Contact form on listing page. Auto-fill with Playwright.",
    },
    "immoscout24": {
        "method": "web_form",
        "needs_login": False,
        "notes": "Contact form on listing page.",
    },
    "engelvoelkers": {
        "method": "email",
        "needs_login": False,
        "notes": "Direct email to broker (address in original email).",
    },
    "neho": {
        "method": "web_form",
        "needs_login": True,
        "notes": "Login required. Contact form after login.",
    },
    "comparis": {
        "method": "web_form",
        "needs_login": False,
        "notes": "Redirects to source portal's contact form.",
    },
    "newhome": {
        "method": "web_form",
        "needs_login": False,
        "notes": "Contact form on listing page.",
    },
    "betterhomes": {
        "method": "email",
        "needs_login": False,
        "notes": "BETTERHOMES provides agent email directly.",
    },
    "sothebys": {
        "method": "web_form",
        "needs_login": False,
        "notes": "Contact form on listing page.",
    },
    "raiffeisen": {
        "method": "web_form",
        "needs_login": False,
        "notes": "Contact form on listing page.",
    },
    "remax": {
        "method": "web_form",
        "needs_login": True,
        "notes": "Login may be required for contact details.",
    },
    # Klosters local brokers — typically email or phone
    "fross": {
        "method": "email",
        "default_email": "info@fross.ch",
        "notes": "Small local broker. Email or phone.",
    },
    "hodel": {
        "method": "email",
        "default_email": "info@hodel-immo.ch",
        "notes": "Local broker.",
    },
    "ambuehl": {
        "method": "email",
        "default_email": "info@ambuehl-immo.ch",
        "notes": "Local broker.",
    },
    "rki": {
        "method": "email",
        "default_email": "info@rki.ch",
        "notes": "Rätia Immobilien.",
    },
    "teresas": {
        "method": "email",
        "default_email": "info@teresas-homes.ch",
        "notes": "Luxury Klosters broker.",
    },
    "ginesta": {
        "method": "email",
        "default_email": "info@ginesta.ch",
        "notes": "Graubünden broker.",
    },
}


def get_contact_method(source: str, agent_email: str = None, listing_url: str = None) -> ContactMethod:
    """
    Determine how to contact the agent for a given portal.

    Args:
        source: Portal name (e.g., "homegate", "fross")
        agent_email: Agent email if known from listing/email
        listing_url: Listing page URL (for web form contact)

    Returns:
        ContactMethod with method, email/URL, and login requirements
    """
    config = PORTAL_CONTACT_CONFIG.get(source, {})
    method = config.get("method", "manual")

    # If we have agent email, always prefer direct email
    if agent_email:
        return ContactMethod(
            method="email",
            agent_email=agent_email,
            portal_url=listing_url,
        )

    # Use portal default email for local brokers
    default_email = config.get("default_email")
    if default_email:
        return ContactMethod(
            method="email",
            agent_email=default_email,
            portal_url=listing_url,
        )

    # Web form
    if method == "web_form":
        return ContactMethod(
            method="web_form",
            portal_url=listing_url,
            needs_login=config.get("needs_login", False),
        )

    # Fallback: manual
    return ContactMethod(
        method="manual",
        portal_url=listing_url,
    )
