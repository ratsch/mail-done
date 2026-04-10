"""
Comparis.ch scraper. Aggregates listings from multiple sources.
"""

import logging
from .base import BaseScraper, ScrapedListing
from .extractors import extract_from_html, extract_photos_from_html

logger = logging.getLogger(__name__)


class ComparisScraper(BaseScraper):
    PORTAL_NAME = "Comparis"
    PORTAL_DOMAIN = "comparis.ch"

    async def scrape(self, url: str) -> ScrapedListing:
        if not self.browser:
            return ScrapedListing(listing_url=url, listing_source="comparis")

        page = await self.browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(3000)

            html = await page.content()
            listing = extract_from_html(html, url, "comparis")
            listing.photo_urls = extract_photos_from_html(html)
            return listing
        except Exception as e:
            logger.error(f"Comparis scrape failed: {e}")
            return ScrapedListing(listing_url=url, listing_source="comparis")
        finally:
            await page.close()
