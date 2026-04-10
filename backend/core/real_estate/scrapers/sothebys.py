"""
Sotheby's International Realty scraper. Luxury segment. HTTP-based.
"""

import logging
import httpx
from .base import BaseScraper, ScrapedListing
from .extractors import extract_from_html, extract_photos_from_html

logger = logging.getLogger(__name__)


class SothebysRealtyScraper(BaseScraper):
    PORTAL_NAME = "Sotheby's International Realty"
    PORTAL_DOMAIN = "sothebysrealty"

    async def scrape(self, url: str) -> ScrapedListing:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    listing = extract_from_html(resp.text, url, "sothebys")
                    listing.photo_urls = extract_photos_from_html(resp.text)
                    return listing
        except Exception as e:
            logger.error(f"Sotheby's scrape failed: {e}")

        return ScrapedListing(listing_url=url, listing_source="sothebys")
