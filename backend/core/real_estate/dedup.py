"""
Property deduplication across multiple portal sources.

Same property appears on Homegate, ImmoScout, and broker website with
different photos, agents, sometimes different prices. This module
detects duplicates and links them to a single PropertyListing record.
"""

import hashlib
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_swiss_address(address: str) -> str:
    """
    Normalize Swiss address for dedup comparison.

    Handles:
    - Umlauts: ä→ae, ö→oe, ü→ue
    - Street abbreviations: Str. → strasse
    - Whitespace normalization
    - Case normalization
    """
    if not address:
        return ""
    addr = address.lower().strip()

    # Normalize umlauts
    addr = addr.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")

    # Normalize street abbreviations
    addr = re.sub(r"\bstr\.?\b", "strasse", addr)
    addr = re.sub(r"\bpl\.?\b", "platz", addr)

    # Remove common noise words
    addr = re.sub(r"\b(ca\.?|circa|ungefähr)\b", "", addr)

    # Normalize whitespace
    addr = re.sub(r"\s+", " ", addr).strip()

    return addr


def compute_dedup_hash(
    plz: str = None,
    address: str = None,
    street: str = None,
    street_nr: str = None,
    price: int = None,
    sqm: int = None,
    rooms: float = None,
) -> str:
    """
    Compute dedup hash from available property attributes.

    Strategy:
    - If address + PLZ known → strong hash (address-based)
    - If only PLZ + price + sqm + rooms → weak hash (attribute-based)
    - The hash is stored on PropertyListing.dedup_hash for fast lookups

    Returns:
        16-char hex hash string
    """
    parts = [str(plz or "")]

    # Build full address from components if not provided
    full_addr = address
    if not full_addr and street:
        full_addr = f"{street} {street_nr or ''}".strip()

    if full_addr:
        # Strong dedup: address-based
        parts.append(normalize_swiss_address(full_addr))
    else:
        # Weak dedup: attribute-based (less reliable — same specs ≠ same property)
        parts.extend([
            str(price or ""),
            str(sqm or ""),
            str(rooms or ""),
        ])

    key = "|".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def find_duplicate(session, scraped_listing) -> Optional[object]:
    """
    Check if a similar property already exists in the database.

    Args:
        session: SQLAlchemy session
        scraped_listing: ScrapedListing with property data

    Returns:
        Existing PropertyListing if found, None otherwise
    """
    from backend.core.database.models_property import PropertyListing

    dedup_hash = compute_dedup_hash(
        plz=scraped_listing.plz,
        address=scraped_listing.address,
        street=scraped_listing.street,
        street_nr=scraped_listing.street_nr,
        price=scraped_listing.price_chf,
        sqm=scraped_listing.living_area_sqm,
        rooms=scraped_listing.rooms,
    )

    existing = session.query(PropertyListing).filter(
        PropertyListing.dedup_hash == dedup_hash
    ).first()

    if existing:
        logger.info(
            f"Duplicate found: {scraped_listing.full_address or scraped_listing.plz} "
            f"→ existing listing {str(existing.id)[:8]}"
        )

    return existing


def link_source(session, listing, email_id, source: str, listing_url: str = None, price: int = None):
    """
    Link an additional email/source to an existing PropertyListing.

    Called when a duplicate is detected — the new email notification
    is linked as another source for the same property.
    """
    from backend.core.database.models_property import PropertyListingSource
    from datetime import datetime, timezone

    # Check if this source is already linked
    existing_source = session.query(PropertyListingSource).filter(
        PropertyListingSource.listing_id == listing.id,
        PropertyListingSource.source == source,
    ).first()

    if existing_source:
        logger.debug(f"Source {source} already linked to listing {str(listing.id)[:8]}")
        return

    new_source = PropertyListingSource(
        listing_id=listing.id,
        email_id=email_id,
        source=source,
        listing_url=listing_url,
        price_at_source=price,
        first_seen_at=datetime.now(timezone.utc),
    )
    session.add(new_source)
    logger.info(f"Linked source {source} to listing {str(listing.id)[:8]}")
