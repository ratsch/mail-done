"""
Homegate.ch + ImmoScout24.ch scrapers.

Homegate is a React SPA — needs Playwright for full rendering.
Notification emails are weekly digests with 5-10 listings.
"""

import re
import logging
from typing import List

from .base import BaseScraper, ScrapedListing
from .extractors import extract_from_html, extract_from_text, extract_photos_from_html, strip_html

logger = logging.getLogger(__name__)


class HomegateScraper(BaseScraper):
    PORTAL_NAME = "Homegate"
    PORTAL_DOMAIN = "homegate.ch"

    async def scrape(self, url: str) -> ScrapedListing:
        """Scrape a Homegate listing page. Requires Playwright (React SPA)."""
        if not self.browser:
            logger.warning("Homegate requires Playwright — returning empty listing")
            return ScrapedListing(listing_url=url, listing_source="homegate")

        page = await self.browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(3000)

            html = await page.content()
            listing = extract_from_html(html, url, "homegate")
            listing.photo_urls = extract_photos_from_html(html)

            return listing
        except Exception as e:
            logger.error(f"Homegate scrape failed for {url}: {e}")
            return ScrapedListing(listing_url=url, listing_source="homegate")
        finally:
            await page.close()

    # Markers that indicate the start of the email footer / signature block.
    # Anything after these is marketing/legal text, not listing content — and
    # contains the SMG HQ address ("Thurgauerstrasse 36, 8050 Zurich") which
    # the address regex would otherwise misattribute to the last listing.
    FOOTER_MARKERS = (
        "SMG Swiss Marketplace Group",
        "Verwalten Sie Ihre Suchaufträge",
        "Manage your alerts",
        "Manage your search subscriptions",
        "Vous recevez ce message",
        "Unsubscribe",
        "Abbestellen",
    )

    def parse_from_email(self, email_body: str, email_subject: str = "") -> List[ScrapedListing]:
        """
        Parse multiple listings from a Homegate digest email.

        Each listing block has: price, PLZ+municipality, rooms, area, link.
        """
        listings = []

        # Strip the email footer before block-splitting, otherwise the last
        # block extends to end-of-body and the address regex picks up the
        # SMG company address from the footer.
        footer_pos = -1
        for marker in self.FOOTER_MARKERS:
            pos = email_body.find(marker)
            if pos >= 0 and (footer_pos < 0 or pos < footer_pos):
                footer_pos = pos
        if footer_pos > 0:
            email_body = email_body[:footer_pos]

        # Split into blocks by price line
        blocks = re.split(r'(?=(?:CHF\s|Preis auf Anfrage|Price on request))', email_body)

        for block in blocks:
            if len(block) < 20:
                continue

            listing = extract_from_text(block, source="homegate")

            # Extract listing URL (homegate or SendGrid wrapped)
            url_match = re.search(r'(https?://[^\s"\'<>\[\]]+homegate\.ch[^\s"\'<>\[\]]+)', block)
            if url_match:
                listing.listing_url = url_match.group(1)
            else:
                sg_match = re.search(r'(https?://[^\s"\'<>\[\]]*sendgrid[^\s"\'<>\[\]]+)', block)
                if sg_match:
                    listing.listing_url = sg_match.group(1)

            if listing.plz or listing.price_known or listing.rooms:
                listings.append(listing)

        logger.info(f"Homegate email: extracted {len(listings)} listings")
        return listings


class ImmoScout24Scraper(BaseScraper):
    """ImmoScout24.ch — similar to Homegate, React SPA."""

    PORTAL_NAME = "ImmoScout24"
    PORTAL_DOMAIN = "immoscout24.ch"

    async def scrape(self, url: str) -> ScrapedListing:
        if not self.browser:
            return ScrapedListing(listing_url=url, listing_source="immoscout24")

        page = await self.browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(3000)

            html = await page.content()
            listing = extract_from_html(html, url, "immoscout24")
            listing.photo_urls = extract_photos_from_html(html)

            return listing
        except Exception as e:
            logger.error(f"ImmoScout24 scrape failed for {url}: {e}")
            return ScrapedListing(listing_url=url, listing_source="immoscout24")
        finally:
            await page.close()
