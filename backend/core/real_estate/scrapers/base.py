"""
Base classes for portal-specific property listing scrapers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScrapedListing:
    """Structured data extracted from a listing page or email."""

    # Identity
    listing_url: str = ""
    listing_source: str = ""                 # "homegate", "immoscout", etc.
    listing_ref_id: Optional[str] = None     # Portal-specific reference

    # Property details
    address: Optional[str] = None            # Full address or just street
    street: Optional[str] = None             # Street name only
    street_nr: Optional[str] = None          # Street number
    plz: Optional[str] = None
    municipality: Optional[str] = None
    canton: Optional[str] = None
    price_chf: Optional[int] = None
    price_known: bool = False
    living_area_sqm: Optional[int] = None
    land_area_sqm: Optional[int] = None
    rooms: Optional[float] = None
    year_built: Optional[int] = None
    last_renovation: Optional[int] = None
    floor: Optional[int] = None
    property_type: Optional[str] = None      # "apartment", "single_family", "multi_family"
    description: Optional[str] = None

    # Specifics
    heating_type: Optional[str] = None       # "oil", "heat_pump", "gas", "district", "wood"
    heating_cost_yearly: Optional[int] = None
    parking_spaces: Optional[int] = None
    parking_included: Optional[bool] = None
    num_units_in_building: Optional[int] = None
    wertquote: Optional[str] = None          # STWE share "44/1000"
    erneuerungsfonds_chf: Optional[int] = None
    nebenkosten_yearly: Optional[int] = None
    zweitwohnung_allowed: Optional[bool] = None
    has_mountain_view: Optional[bool] = None
    has_lake_view: Optional[bool] = None
    has_garden_access: Optional[bool] = None
    has_terrace: Optional[bool] = None

    # Features + photos
    features: List[str] = field(default_factory=list)
    photo_urls: List[str] = field(default_factory=list)

    # Agent
    agent_name: Optional[str] = None
    agent_email: Optional[str] = None
    agent_phone: Optional[str] = None
    agent_company: Optional[str] = None

    @property
    def completeness_pct(self) -> int:
        """How complete is the listing data (0-100)."""
        checks = [
            self.price_known,
            self.address is not None or (self.street is not None and self.plz is not None),
            self.living_area_sqm is not None,
            self.rooms is not None,
            self.year_built is not None,
            len(self.photo_urls) > 0,
        ]
        return int(sum(checks) / len(checks) * 100)

    @property
    def missing_fields(self) -> List[str]:
        """What info is still needed."""
        missing = []
        if not self.price_known:
            missing.append("asking price")
        if not self.address and not (self.street and self.plz):
            missing.append("exact address")
        if not self.living_area_sqm:
            missing.append("living area")
        if not self.rooms:
            missing.append("number of rooms")
        if not self.year_built:
            missing.append("year of construction")
        return missing

    @property
    def full_address(self) -> Optional[str]:
        """Construct full address from components."""
        if self.address:
            return self.address
        parts = []
        if self.street:
            parts.append(f"{self.street} {self.street_nr or ''}".strip())
        if self.plz and self.municipality:
            parts.append(f"{self.plz} {self.municipality}")
        elif self.plz:
            parts.append(self.plz)
        return ", ".join(parts) if parts else None


class BaseScraper(ABC):
    """Base class for portal-specific scrapers."""

    PORTAL_NAME: str = ""       # Human-readable name
    PORTAL_DOMAIN: str = ""     # e.g., "homegate.ch"

    def __init__(self, playwright_browser=None):
        self.browser = playwright_browser

    def can_handle(self, url: str) -> bool:
        """Check if this scraper can handle the given URL."""
        return self.PORTAL_DOMAIN in url.lower()

    @abstractmethod
    async def scrape(self, url: str) -> ScrapedListing:
        """Scrape a single listing URL and return structured data."""
        ...

    def parse_from_email(self, email_body: str, email_subject: str = "") -> List[ScrapedListing]:
        """
        Extract listing data from an email body.

        Override for portals where the email contains rich listing data
        (e.g., Engel & Völkers). Default returns empty list.
        """
        return []

    def extract_urls_from_email(self, email_body: str) -> List[str]:
        """
        Extract listing URLs for this portal from email body.

        Override for portals with non-standard URL patterns.
        Default looks for URLs containing the portal domain.
        """
        import re
        url_pattern = r'https?://[^\s<>\[\]"\')\]]+'
        all_urls = re.findall(url_pattern, email_body)
        return [u for u in all_urls if self.PORTAL_DOMAIN in u.lower()]
