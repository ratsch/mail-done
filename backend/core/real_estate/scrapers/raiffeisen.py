"""
Raiffeisen Immobilien scraper. Bank-affiliated listings.
"""

import logging
from .base import BaseScraper, ScrapedListing
from .extractors import extract_from_html, extract_photos_from_html

logger = logging.getLogger(__name__)


class RaiffeisenScraper(BaseScraper):
    PORTAL_NAME = "Raiffeisen Immobilien"
    PORTAL_DOMAIN = "raiffeisen.ch"

    async def scrape(self, url: str) -> ScrapedListing:
        if not self.browser:
            return ScrapedListing(listing_url=url, listing_source="raiffeisen")

        page = await self.browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(3000)
            html = await page.content()
            listing = extract_from_html(html, url, "raiffeisen")
            listing.photo_urls = extract_photos_from_html(html)
            return listing
        except Exception as e:
            logger.error(f"Raiffeisen scrape failed: {e}")
            return ScrapedListing(listing_url=url, listing_source="raiffeisen")
        finally:
            await page.close()
