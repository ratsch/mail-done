"""
Engel & Völkers email parser.

E&V sends rich HTML emails from individual brokers with full listing details.
No web scraping needed — all data is in the email body.
"""

import re
import logging
from typing import List

from .base import BaseScraper, ScrapedListing
from .extractors import extract_from_text, extract_agent_info

logger = logging.getLogger(__name__)


class EngelVoelkersScraper(BaseScraper):
    PORTAL_NAME = "Engel & Völkers"
    PORTAL_DOMAIN = "engelvoelkers.com"

    async def scrape(self, url: str) -> ScrapedListing:
        """E&V listings are parsed from email, not scraped from web."""
        return ScrapedListing(listing_url=url, listing_source="engelvoelkers")

    def parse_from_email(self, email_body: str, email_subject: str = "") -> List[ScrapedListing]:
        """
        Extract listing data from E&V broker email.

        These emails contain structured property data in a table format:
        | Baujahr | 2004 |
        | Wohnfläche | ca. 175 m² |
        | Kaufpreis | CHF 2'900'000.- |
        """
        # Use shared extractor for the bulk of data
        listing = extract_from_text(email_body, source="engelvoelkers")

        # E&V-specific: extract from subject
        subj_location = re.search(r"in\s+(\w+(?:\s+\w+)?)\s*:", email_subject)
        if subj_location and not listing.municipality:
            listing.municipality = subj_location.group(1)

        # Reference number — "Referenz | W-02EXU0"
        ref_match = re.search(r"(?:Referenz|Reference|Ref)[^\w]*([\w-]+)", email_body, re.IGNORECASE)
        if ref_match:
            listing.listing_ref_id = ref_match.group(1)

        # Agent info from email signature
        agent = extract_agent_info(email_body)
        # More specific: E&V email addresses
        ev_email = re.search(r"([\w.]+@engelvoelkers\.com)", email_body)
        if ev_email:
            listing.agent_email = ev_email.group(1)
            parts = ev_email.group(1).split("@")[0].split(".")
            listing.agent_name = " ".join(p.capitalize() for p in parts)
        elif agent["email"]:
            listing.agent_email = agent["email"]

        if agent["phone"]:
            listing.agent_phone = agent["phone"]
        listing.agent_company = "Engel & Völkers"

        # E&V listing URL
        ev_url = re.search(r"(https?://[^\s]*engelvoelkers\.com[^\s<\"']+)", email_body)
        if ev_url:
            listing.listing_url = ev_url.group(1)

        # Feature detection from bullet points
        features_text = " ".join(listing.features).lower()
        if "minergie" in features_text or "minergie" in email_body.lower():
            listing.features.append("minergie")
        if any(w in email_body.lower() for w in ["cheminée", "kamin", "chemin"]):
            listing.features.append("fireplace")
        if "fussbodenheizung" in email_body.lower():
            listing.features.append("floor_heating")

        logger.info(
            f"E&V email: {listing.municipality or '?'}, "
            f"CHF {listing.price_chf or '?'}, "
            f"{listing.rooms or '?'} rooms"
        )

        return [listing] if (listing.price_known or listing.rooms or listing.municipality) else []
