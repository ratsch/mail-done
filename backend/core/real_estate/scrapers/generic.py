"""
Generic scraper — fallback for unknown portals.
Uses shared extractors. Can optionally use LLM for enhanced extraction.
"""

import logging
import httpx
from .base import BaseScraper, ScrapedListing
from .extractors import extract_from_html, extract_photos_from_html

logger = logging.getLogger(__name__)


class GenericScraper(BaseScraper):
    """
    Fallback scraper using shared extractors on any page.
    Tries HTTP first, Playwright if available as fallback.
    """

    PORTAL_NAME = "Generic"
    PORTAL_DOMAIN = ""

    def can_handle(self, url: str) -> bool:
        return True  # Always available as fallback

    async def scrape(self, url: str) -> ScrapedListing:
        # Try HTTP first
        try:
            async with httpx.AsyncClient(
                timeout=15,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            ) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    listing = extract_from_html(resp.text, url, "unknown")
                    listing.photo_urls = extract_photos_from_html(resp.text)
                    if listing.completeness_pct >= 30:
                        return listing
        except Exception as e:
            logger.debug(f"Generic HTTP failed: {e}")

        # Playwright fallback
        if self.browser:
            page = await self.browser.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(2000)
                html = await page.content()
                listing = extract_from_html(html, url, "unknown")
                listing.photo_urls = extract_photos_from_html(html)
                return listing
            except Exception as e:
                logger.error(f"Generic scrape failed: {e}")
            finally:
                await page.close()

        return ScrapedListing(listing_url=url, listing_source="unknown")
