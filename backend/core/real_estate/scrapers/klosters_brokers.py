"""
Klosters-specific local broker scrapers + RE/MAX.

Small local agencies with simple server-rendered websites.
All use the shared extractors — only the fetch method differs.
"""

import logging
import httpx
from .base import BaseScraper, ScrapedListing
from .extractors import extract_from_html, extract_photos_from_html

logger = logging.getLogger(__name__)


class _HttpScraper(BaseScraper):
    """Base for simple HTTP-based scrapers (server-rendered sites)."""

    async def scrape(self, url: str) -> ScrapedListing:
        try:
            async with httpx.AsyncClient(
                timeout=15,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            ) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    listing = extract_from_html(resp.text, url, self.PORTAL_DOMAIN.split(".")[0])
                    listing.photo_urls = extract_photos_from_html(resp.text)
                    return listing
        except Exception as e:
            logger.error(f"{self.PORTAL_NAME} scrape failed: {e}")

        return ScrapedListing(listing_url=url, listing_source=self.PORTAL_DOMAIN.split(".")[0])


class FrossScraper(_HttpScraper):
    PORTAL_NAME = "Fross Immobilien"
    PORTAL_DOMAIN = "fross.ch"


class HodelImmoScraper(_HttpScraper):
    PORTAL_NAME = "Hodel Immobilien"
    PORTAL_DOMAIN = "hodel-immo.ch"


class AmbuehlImmoScraper(_HttpScraper):
    PORTAL_NAME = "Ambühl Immobilien"
    PORTAL_DOMAIN = "ambuehl-immo.ch"


class RKIScraper(_HttpScraper):
    PORTAL_NAME = "RKI Rätia"
    PORTAL_DOMAIN = "rki.ch"


class TeresasHomesScraper(_HttpScraper):
    PORTAL_NAME = "Teresa's Homes"
    PORTAL_DOMAIN = "teresas-homes.ch"


class GinestaScraper(_HttpScraper):
    PORTAL_NAME = "Ginesta"
    PORTAL_DOMAIN = "ginesta.ch"


class RemaxScraper(BaseScraper):
    """RE/MAX — needs Playwright (React-based site)."""
    PORTAL_NAME = "RE/MAX"
    PORTAL_DOMAIN = "remax.ch"

    async def scrape(self, url: str) -> ScrapedListing:
        if not self.browser:
            return ScrapedListing(listing_url=url, listing_source="remax")

        page = await self.browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(3000)
            html = await page.content()
            listing = extract_from_html(html, url, "remax")
            listing.photo_urls = extract_photos_from_html(html)
            return listing
        except Exception as e:
            logger.error(f"RE/MAX scrape failed: {e}")
            return ScrapedListing(listing_url=url, listing_source="remax")
        finally:
            await page.close()
