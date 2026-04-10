"""
Newhome.ch scraper. Mostly server-rendered — HTTP first, Playwright fallback.
"""

import logging
import httpx
from .base import BaseScraper, ScrapedListing
from .extractors import extract_from_html, extract_photos_from_html

logger = logging.getLogger(__name__)


class NewhomeScraper(BaseScraper):
    PORTAL_NAME = "Newhome"
    PORTAL_DOMAIN = "newhome.ch"

    async def scrape(self, url: str) -> ScrapedListing:
        # Try HTTP first (server-rendered)
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    listing = extract_from_html(resp.text, url, "newhome")
                    listing.photo_urls = extract_photos_from_html(resp.text)
                    if listing.completeness_pct >= 30:
                        return listing
        except Exception as e:
            logger.debug(f"Newhome HTTP failed, trying Playwright: {e}")

        # Fallback to Playwright
        if self.browser:
            page = await self.browser.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(2000)
                html = await page.content()
                listing = extract_from_html(html, url, "newhome")
                listing.photo_urls = extract_photos_from_html(html)
                return listing
            except Exception as e:
                logger.error(f"Newhome scrape failed: {e}")
            finally:
                await page.close()

        return ScrapedListing(listing_url=url, listing_source="newhome")
