"""
Listing Scraper — orchestrates per-portal scrapers.

Auto-detects the portal from the URL and dispatches to the right scraper.
Falls back to generic LLM-assisted scraper for unknown portals.
"""

import logging
from typing import List, Optional

from .scrapers.base import BaseScraper, ScrapedListing
from .scrapers.url_resolver import resolve_tracking_url, extract_listing_urls_from_email
from .scrapers.homegate import HomegateScraper, ImmoScout24Scraper
from .scrapers.engelvoelkers import EngelVoelkersScraper
from .scrapers.neho import NehoScraper
from .scrapers.comparis import ComparisScraper
from .scrapers.newhome import NewhomeScraper
from .scrapers.betterhomes import BetterhomesScraper
from .scrapers.sothebys import SothebysRealtyScraper
from .scrapers.raiffeisen import RaiffeisenScraper
from .scrapers.klosters_brokers import (
    FrossScraper, HodelImmoScraper, AmbuehlImmoScraper,
    RKIScraper, TeresasHomesScraper, RemaxScraper, GinestaScraper,
)
from .scrapers.generic import GenericScraper

logger = logging.getLogger(__name__)


class ListingScraper:
    """
    Orchestrate scraping across all portal scrapers.

    Usage:
        scraper = ListingScraper(playwright_browser)

        # Scrape a single URL
        listing = await scraper.scrape("https://www.homegate.ch/kaufen/12345")

        # Extract listings from email
        listings = await scraper.extract_from_email(email_body, email_subject, source)
    """

    def __init__(self, playwright_browser=None):
        self.browser = playwright_browser

        # Portal-specific scrapers (order doesn't matter — matched by domain)
        self.scrapers: List[BaseScraper] = [
            HomegateScraper(playwright_browser),
            ImmoScout24Scraper(playwright_browser),
            ComparisScraper(playwright_browser),
            NewhomeScraper(playwright_browser),
            NehoScraper(playwright_browser),
            BetterhomesScraper(playwright_browser),
            SothebysRealtyScraper(playwright_browser),
            RaiffeisenScraper(playwright_browser),
            RemaxScraper(playwright_browser),
            FrossScraper(playwright_browser),
            HodelImmoScraper(playwright_browser),
            AmbuehlImmoScraper(playwright_browser),
            RKIScraper(playwright_browser),
            TeresasHomesScraper(playwright_browser),
            GinestaScraper(playwright_browser),
        ]

        # Email-based parsers (extract from email body, no web scraping)
        self.email_parsers: List[BaseScraper] = [
            EngelVoelkersScraper(),
            HomegateScraper(playwright_browser),  # Also parses digest emails
        ]

        # Generic fallback
        self.generic = GenericScraper(playwright_browser)

    async def scrape(self, url: str) -> ScrapedListing:
        """
        Scrape a listing URL. Auto-detects portal, falls back to generic.

        Resolves tracking/redirect URLs first (SendGrid, Mailjet, etc.).
        """
        # Resolve tracking URLs
        resolved_url = await resolve_tracking_url(url)
        logger.info(f"Scraping: {resolved_url[:80]}...")

        # Find matching scraper
        for scraper in self.scrapers:
            if scraper.can_handle(resolved_url):
                logger.debug(f"Using {scraper.PORTAL_NAME} scraper")
                listing = await scraper.scrape(resolved_url)
                listing.listing_url = resolved_url
                return listing

        # Fallback to generic
        logger.debug("No portal-specific scraper matched, using generic")
        return await self.generic.scrape(resolved_url)

    async def extract_from_email(
        self,
        email_body: str,
        email_subject: str = "",
        source: str = "",
    ) -> List[ScrapedListing]:
        """
        Extract listing data from an email.

        For rich emails (E&V): parses data directly from email body.
        For digest emails (Homegate): extracts multiple listings with URLs.
        For other emails: extracts listing URLs for later scraping.

        Args:
            email_body: Email body text (HTML converted to markdown/text)
            email_subject: Email subject line
            source: Detected source (e.g., "homegate", "engelvoelkers")

        Returns:
            List of ScrapedListing (may be empty, one, or many)
        """
        listings = []

        # Try email-body parsers first (E&V, Homegate digest)
        for parser in self.email_parsers:
            if source and parser.PORTAL_DOMAIN and source.lower() in parser.PORTAL_DOMAIN:
                parsed = parser.parse_from_email(email_body, email_subject)
                if parsed:
                    logger.info(f"{parser.PORTAL_NAME}: parsed {len(parsed)} listings from email")
                    listings.extend(parsed)
                    return listings

        # For other sources: extract URLs from email and create stub listings.
        # Resolve tracking URLs (SendGrid, Mailjet, etc.) IMMEDIATELY while
        # they're still fresh — these tokens expire, so storing the wrapper
        # would prevent any later scrape attempt from succeeding.
        urls = extract_listing_urls_from_email(email_body)
        if urls:
            logger.info(f"Found {len(urls)} listing URLs in email (source: {source})")
            for url in urls:
                resolved_url = await resolve_tracking_url(url)
                if resolved_url != url:
                    logger.info(
                        f"  resolved tracking URL → {resolved_url[:80]}"
                    )
                listings.append(ScrapedListing(
                    listing_url=resolved_url,
                    listing_source=source or "unknown",
                ))

        return listings

    async def scrape_all_from_email(
        self,
        email_body: str,
        email_subject: str = "",
        source: str = "",
    ) -> List[ScrapedListing]:
        """
        Extract AND scrape all listings from an email.

        First extracts listings/URLs from the email, then scrapes each URL
        for full details. This is the full pipeline for one email.

        For Engel & Völkers: no scraping needed (data is in email).
        For Homegate digest: scrapes each extracted URL.
        """
        # Step 1: Extract from email
        listings = await self.extract_from_email(email_body, email_subject, source)

        # Step 2: For listings that only have URLs (no data), scrape them
        enriched = []
        for listing in listings:
            if listing.listing_url and listing.completeness_pct < 30:
                # Needs scraping
                try:
                    scraped = await self.scrape(listing.listing_url)
                    enriched.append(scraped)
                except Exception as e:
                    logger.warning(f"Failed to scrape {listing.listing_url[:60]}: {e}")
                    enriched.append(listing)  # Keep the stub
            else:
                enriched.append(listing)  # Already has data (e.g., E&V)

        return enriched

    # -------------------------------------------------------------------------
    # Proactive polling — check broker websites for new listings
    # -------------------------------------------------------------------------

    # Pages to poll periodically (daily cron) for new listings
    POLL_PAGES = [
        # Klosters local brokers (no email alerts available)
        {"url": "https://www.fross.ch/kauf/klosters/", "source": "fross", "region": "klosters"},
        {"url": "https://www.hodel-immo.ch/immobilien/?availabilities%5B%5D=active&salestypes%5B%5D=buy", "source": "hodel", "region": "klosters"},
        {"url": "https://ambuehl-immo.ch/immobilie-kaufen/", "source": "ambuehl", "region": "klosters"},
        {"url": "https://rki.ch/kaufen/", "source": "rki", "region": "klosters"},
        {"url": "https://www.teresas-homes.ch/en/sales/", "source": "teresas", "region": "klosters"},
        {"url": "https://www.teresas-homes.ch/en/klosters-exclusive-sales-properties/", "source": "teresas", "region": "klosters"},
        {"url": "https://www.ginesta.ch/objekte/immobilien-kaufen/", "source": "ginesta", "region": "klosters"},
    ]

    async def poll_for_new_listings(self) -> List[ScrapedListing]:
        """
        Check all broker listing pages for new properties.

        Run this daily via cron. Extracts individual listing URLs from
        each index page, then checks against known listings for new ones.

        Returns:
            List of ScrapedListing for newly discovered properties
        """
        all_listings = []

        for page_config in self.POLL_PAGES:
            url = page_config["url"]
            source = page_config["source"]
            logger.info(f"Polling {source}: {url}")

            try:
                listing_urls = await self._extract_listing_urls_from_page(url, source)
                logger.info(f"  Found {len(listing_urls)} listing URLs on {source}")

                for listing_url in listing_urls:
                    listing = ScrapedListing(
                        listing_url=listing_url,
                        listing_source=source,
                    )
                    all_listings.append(listing)

            except Exception as e:
                logger.warning(f"Failed to poll {source}: {e}")

        return all_listings

    async def _extract_listing_urls_from_page(self, index_url: str, source: str) -> List[str]:
        """Extract individual listing URLs from a broker's listing index page."""
        import httpx

        urls = []
        try:
            async with httpx.AsyncClient(
                timeout=15,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            ) as client:
                resp = await client.get(index_url)
                if resp.status_code != 200:
                    return urls

                html = resp.text
                domain = re.match(r'https?://([^/]+)', index_url).group(0)

                # Find links to individual listings
                # Most broker sites link to /kaufen/<id>, /immobilien/<id>, /property/<id>, etc.
                link_patterns = [
                    r'href=["\']([^"\']*(?:/kaufen/|/immobilien?/|/property/|/objekt/|/detail/|/sales/)[^"\']+)["\']',
                    r'href=["\']([^"\']*(?:/kauf/|/klosters/|/davos/)[^"\']*\d+[^"\']*)["\']',
                ]

                for pattern in link_patterns:
                    matches = re.findall(pattern, html, re.IGNORECASE)
                    for match in matches:
                        # Convert relative URLs to absolute
                        if match.startswith("/"):
                            match = f"{domain}{match}"
                        elif not match.startswith("http"):
                            match = f"{domain}/{match}"

                        # Skip index pages, pagination, filters
                        if any(skip in match.lower() for skip in ["page=", "sort=", "filter", "#", "javascript"]):
                            continue

                        urls.append(match)

        except Exception as e:
            logger.warning(f"Failed to extract URLs from {index_url}: {e}")

        return list(set(urls))  # Deduplicate
