"""
Swiss Property Listing Extractors — shared extraction logic.

All portal-specific scrapers delegate the actual data extraction to these
functions. This avoids duplicating regex patterns across 15+ scrapers.

Extraction strategies (tried in order):
1. JSON-LD structured data (most reliable)
2. OpenGraph / meta tags
3. Structured CSS selectors (portal-specific, passed as config)
4. Regex on visible text (fallback, always works)
"""

import json
import re
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .base import ScrapedListing

logger = logging.getLogger(__name__)


# =============================================================================
# Swiss number/text patterns
# =============================================================================

# Price: "CHF 1'500'000.–" or "CHF 1,500,000" or "1500000"
# Note: Swiss formatting uses ' (U+2019 right single quotation mark) as thousand separator
PRICE_PATTERNS = [
    r"(?:CHF|Fr\.)\s*([\d'''\u2018\u2019.,]+)(?:\.\s*[-–])?",
    r"(?:Kaufpreis|Verkaufspreis|Price|Preis)[:\s]*(?:CHF|Fr\.)?\s*([\d'''\u2018\u2019.,]+)",
]

# "Preis auf Anfrage" / "Price on request"
PRICE_ON_REQUEST = re.compile(
    r"(?:Preis auf Anfrage|Price on request|auf Anfrage|sur demande|su richiesta)",
    re.IGNORECASE
)

# Rooms: "3.5 Zimmer" or "5½-Zimmer" or "4 rooms"
ROOMS_PATTERNS = [
    r"([\d]+[.,]?[\d]*)\s*(?:½\s*)?(?:Zimmer|Zi\.|rooms|pièces|locali)",
    r"(\d+)½[\s-]*(?:Zimmer|Zi\.)",  # "3½-Zimmer" → 3.5
]

# Area: "120 m²" or "Wohnfläche: ca. 120 m²"
AREA_PATTERNS = [
    r"(?:Wohnfläche|Nettowohnfläche|Surface habitable|Living area|Fläche)[:\s]*(?:ca\.?\s*)?(\d+)\s*m",
    r"(\d+)\s*m[²2]",
]

# Land area: "Grundstücksfläche: 500 m²"
LAND_AREA_PATTERNS = [
    r"(?:Grundstück|Grundfläche|Land area|Terrain|Parzelle)[:\s]*(?:ca\.?\s*)?(\d+)\s*m",
]

# PLZ + municipality: "8032 Zürich" or "7250 Klosters" or "7247 Saas im Prättigau"
PLZ_PATTERNS = [
    # Known multi-word municipalities
    r"\b(\d{4})\s+(Saas im Prättigau|Klosters Dorf|Klosters-Serneus|Davos Wolfgang|Davos Platz|Davos Dorf|Flims Dorf)\b",
    # Common Swiss cities
    r"\b(\d{4})\s+(Zürich|Zurich|Klosters|Davos|Zollikon|Küsnacht|Zumikon|Winterthur|Zug|Luzern|Basel|Bern|Genf|Lausanne|Schluein|Flims)\b",
    # Generic: PLZ + one or two words
    r"\b(\d{4})\s+([A-ZÀ-Ü][a-zà-ü]+(?:[\s-][A-ZÀ-Ü]?[a-zà-ü]+){0,2})\b",
]

# Address: "Bildweg 12" or "Pestalozzistrasse 33"
ADDRESS_PATTERNS = [
    r"([A-ZÀ-Ü][a-zà-ü]+(?:strasse|weg|gasse|platz|rain|acker|hof|path|allee)\s+\d+[a-z]?)",
]

# Year built: "Baujahr: 2004" or "Built: 1986"
YEAR_PATTERNS = [
    r"(?:Baujahr|Bau|Built|Construction|Année)[:\s]*(\d{4})",
]

# Floor: "1. Obergeschoss" or "2. OG" or "Erdgeschoss"
FLOOR_PATTERNS = [
    r"(\d+)\.\s*(?:Obergeschoss|OG|Stock|Etage|Floor)",
    r"(?:Erdgeschoss|Parterre|Ground\s*floor)",  # → floor 0
]

# Heating: various types
HEATING_PATTERNS = {
    "oil": r"(?:Öl|Heizöl|Oil|Mazout)",
    "heat_pump": r"(?:Wärmepumpe|Heat pump|Pompe à chaleur|Erdsonde)",
    "gas": r"(?:Gas|Erdgas)",
    "district": r"(?:Fernwärme|District|Chauffage à distance)",
    "wood": r"(?:Holz|Pellet|Wood|Bois)",
}

# Parking
PARKING_PATTERNS = [
    r"(\d+)\s*(?:x\s*)?(?:Tiefgarage|Garage|Parkplatz|Stellplatz|Parking|Einstellhall)",
    r"(?:Tiefgarage|Garage|Parkplatz|Parking)\s*(?:vorhanden|inkl|included)",
]


# =============================================================================
# Main extraction functions
# =============================================================================

def extract_from_text(text: str, url: str = "", source: str = "") -> ScrapedListing:
    """
    Extract property listing data from plain text using Swiss patterns.

    This is the universal fallback — works on any text containing
    Swiss property listing information.
    """
    listing = ScrapedListing(listing_url=url, listing_source=source)

    # Price
    listing.price_chf, listing.price_known = _extract_price(text)

    # Rooms
    listing.rooms = _extract_rooms(text)

    # Area
    listing.living_area_sqm = _extract_area(text, AREA_PATTERNS)
    listing.land_area_sqm = _extract_area(text, LAND_AREA_PATTERNS)

    # Location
    listing.plz, listing.municipality = _extract_plz(text)
    addr = _extract_address(text)
    if addr:
        listing.address = addr

    # Year built
    listing.year_built = _extract_year(text)

    # Floor
    listing.floor = _extract_floor(text)

    # Heating
    listing.heating_type = _extract_heating(text)

    # Parking
    listing.parking_spaces = _extract_parking(text)

    # Property type
    listing.property_type = _detect_property_type(text)

    # Boolean features
    text_lower = text.lower()
    listing.has_mountain_view = _detect_feature(text_lower, ["bergblick", "bergsicht", "mountain view", "alpenpanorama"])
    listing.has_lake_view = _detect_feature(text_lower, ["seesicht", "seeblick", "lake view", "vue sur le lac"])
    listing.has_garden_access = _detect_feature(text_lower, ["garten", "garden", "jardin", "gartenanteil"])
    listing.has_terrace = _detect_feature(text_lower, ["balkon", "terrasse", "balcony", "terrace", "sitzplatz", "loggia"])

    # Zweitwohnung / Erstwohnung (critical for Klosters)
    if any(kw in text_lower for kw in [
        "erstwohnung", "erstwohnungspflicht", "erstwohnungsanteil",
        "primary residence only", "hauptwohnsitz",
    ]):
        listing.zweitwohnung_allowed = False  # Erstwohnungspflicht = deal-breaker
    elif any(kw in text_lower for kw in [
        "zweitwohnung", "ferienwohnung", "feriendomizil", "vacation",
        "second home", "ohne nutzungsbeschränkung", "als zweitwohnung",
        "secondaire", "résidence secondaire",
    ]):
        listing.zweitwohnung_allowed = True

    # Features list
    listing.features = _extract_features(text)

    return listing


def extract_from_html(html: str, url: str = "", source: str = "") -> ScrapedListing:
    """
    Extract from HTML — tries JSON-LD first, then meta tags, then text.
    """
    # Try JSON-LD
    listing = _extract_json_ld(html, url, source)
    if listing and listing.completeness_pct >= 50:
        return listing

    # Try OpenGraph meta tags
    og_listing = _extract_opengraph(html, url, source)

    # Fall back to text extraction
    text = strip_html(html)
    text_listing = extract_from_text(text, url, source)

    # Merge: JSON-LD > OG > text (prefer most structured source)
    result = text_listing
    if og_listing:
        result = _merge(result, og_listing)
    if listing:
        result = _merge(result, listing)

    return result


def extract_photos_from_html(html: str) -> List[str]:
    """Extract property photo URLs from HTML."""
    photos = []

    # img tags with listing-related sources
    img_matches = re.findall(
        r'(?:src|data-src|data-lazy-src)=["\']([^"\']+\.(?:jpg|jpeg|png|webp))["\']',
        html, re.IGNORECASE
    )
    for url in img_matches:
        # Skip logos, icons, avatars, tracking pixels
        if any(skip in url.lower() for skip in [
            "logo", "icon", "avatar", "pixel", "tracking", "1x1",
            "placeholder", "loading", "spinner", "banner",
        ]):
            continue
        photos.append(url)

    # JSON-LD images
    for match in re.findall(r'"image"\s*:\s*"([^"]+)"', html):
        if match.startswith("http"):
            photos.append(match)

    # OpenGraph image
    og_image = re.search(r'<meta\s+property="og:image"\s+content="([^"]+)"', html)
    if og_image:
        photos.append(og_image.group(1))

    return list(dict.fromkeys(photos))[:20]  # Dedupe, limit to 20


def strip_html(html: str) -> str:
    """Strip HTML tags, scripts, styles to get plain text."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =============================================================================
# Agent/contact extraction
# =============================================================================

def extract_agent_info(text: str) -> Dict[str, Optional[str]]:
    """Extract agent/broker contact info from text."""
    info = {"name": None, "email": None, "phone": None, "company": None}

    # Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    if email_match:
        info["email"] = email_match.group(0)

    # Swiss phone: +41 XX XXX XX XX or 0XX XXX XX XX
    phone_match = re.search(r"(\+41[\s\d]{10,15}|0\d{2}\s?\d{3}\s?\d{2}\s?\d{2})", text)
    if phone_match:
        info["phone"] = phone_match.group(1).strip()

    return info


# =============================================================================
# Internal helpers
# =============================================================================

def _extract_price(text: str) -> tuple[Optional[int], bool]:
    """Extract price and whether it's known."""
    if PRICE_ON_REQUEST.search(text):
        return None, False

    for pattern in PRICE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            price_str = match.group(1).replace("'", "").replace("'", "").replace("\u2018", "").replace("\u2019", "").replace(",", "").replace(".", "").replace("-", "").replace("–", "")
            try:
                price = int(price_str)
                if price > 10000:  # Sanity check — not a room count
                    return price, True
            except ValueError:
                continue

    return None, False


def _extract_rooms(text: str) -> Optional[float]:
    for pattern in ROOMS_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).replace(",", ".")
            try:
                rooms = float(val)
                if "½" in match.group(0):
                    rooms += 0.5
                if 1 <= rooms <= 20:  # Sanity check
                    return rooms
            except ValueError:
                continue
    return None


def _extract_area(text: str, patterns: list) -> Optional[int]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                area = int(match.group(1))
                if 10 <= area <= 5000:  # Sanity check
                    return area
            except ValueError:
                continue
    return None


def _extract_plz(text: str) -> tuple[Optional[str], Optional[str]]:
    for pattern in PLZ_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1), match.group(2).strip()
    return None, None


def _extract_address(text: str) -> Optional[str]:
    for pattern in ADDRESS_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_year(text: str) -> Optional[int]:
    for pattern in YEAR_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if 1800 <= year <= 2030:
                return year
    return None


def _extract_floor(text: str) -> Optional[int]:
    for pattern in FLOOR_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if match.lastindex and match.lastindex >= 1:
                return int(match.group(1))
            return 0  # Erdgeschoss
    return None


def _extract_heating(text: str) -> Optional[str]:
    text_lower = text.lower()
    for heating_type, pattern in HEATING_PATTERNS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            return heating_type
    return None


def _extract_parking(text: str) -> Optional[int]:
    for pattern in PARKING_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if match.lastindex and match.lastindex >= 1:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
            return 1  # "Garage vorhanden" → at least 1
    return None


def _detect_property_type(text: str) -> Optional[str]:
    text_lower = text.lower()
    if any(w in text_lower for w in ["einfamilienhaus", "chalet", "villa", "single family", "detached"]):
        return "single_family"
    elif any(w in text_lower for w in ["wohnung", "apartment", "eigentumswohnung", "appartement"]):
        return "apartment"
    elif any(w in text_lower for w in ["mehrfamilienhaus", "multi", "renditeobjekt", "anlageobj"]):
        return "multi_family"
    return None


def _detect_feature(text_lower: str, keywords: list) -> Optional[bool]:
    return True if any(k in text_lower for k in keywords) else None


def _extract_features(text: str) -> List[str]:
    """Extract feature bullet points."""
    features = []
    for match in re.findall(r"[•\-✓]\s*(.+?)(?:\n|$)", text):
        feat = match.strip()
        if 3 < len(feat) < 100:
            features.append(feat)
    return features[:20]


def _extract_json_ld(html: str, url: str, source: str) -> Optional[ScrapedListing]:
    """Extract from JSON-LD structured data."""
    matches = re.findall(
        r'<script\s+type="application/ld\+json"[^>]*>(.*?)</script>',
        html, re.DOTALL
    )
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, list):
                data = data[0]
            if not isinstance(data, dict):
                continue
            schema_type = data.get("@type", "")
            if schema_type in ("Apartment", "House", "RealEstateListing", "Product",
                               "SingleFamilyResidence", "Residence", "Offer"):
                listing = ScrapedListing(listing_url=url, listing_source=source)

                addr = data.get("address", {})
                if isinstance(addr, dict):
                    listing.street = addr.get("streetAddress", "")
                    listing.plz = addr.get("postalCode", "")
                    listing.municipality = addr.get("addressLocality", "")
                    listing.canton = addr.get("addressRegion", "")

                offers = data.get("offers", {})
                if isinstance(offers, dict):
                    price = offers.get("price")
                    if price:
                        try:
                            listing.price_chf = int(float(str(price).replace(",", "")))
                            listing.price_known = True
                        except (ValueError, TypeError):
                            pass

                listing.rooms = _try_float(data.get("numberOfRooms"))
                floor_size = data.get("floorSize", {})
                if isinstance(floor_size, dict):
                    listing.living_area_sqm = _try_int(floor_size.get("value"))
                elif floor_size:
                    listing.living_area_sqm = _try_int(floor_size)

                listing.description = data.get("description", "")
                listing.property_type = _map_schema_type(schema_type)

                images = data.get("image", [])
                if isinstance(images, list):
                    listing.photo_urls = [
                        (img if isinstance(img, str) else img.get("url", ""))
                        for img in images
                    ]
                elif isinstance(images, str):
                    listing.photo_urls = [images]

                return listing
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def _extract_opengraph(html: str, url: str, source: str) -> Optional[ScrapedListing]:
    """Extract from OpenGraph meta tags."""
    listing = ScrapedListing(listing_url=url, listing_source=source)
    found = False

    og_title = re.search(r'<meta\s+property="og:title"\s+content="([^"]+)"', html)
    if og_title:
        listing.description = og_title.group(1)
        found = True

    og_desc = re.search(r'<meta\s+property="og:description"\s+content="([^"]+)"', html)
    if og_desc:
        # Try to extract data from OG description
        desc = og_desc.group(1)
        text_listing = extract_from_text(desc, url, source)
        if text_listing.rooms or text_listing.price_known:
            return text_listing

    return listing if found else None


def _merge(base: ScrapedListing, overlay: ScrapedListing) -> ScrapedListing:
    """Merge two listings, preferring non-None values from overlay."""
    for field_name in ScrapedListing.__dataclass_fields__:
        if field_name in ("listing_url", "listing_source"):
            continue
        overlay_val = getattr(overlay, field_name, None)
        if overlay_val is not None and overlay_val != "" and overlay_val != [] and overlay_val is not False:
            setattr(base, field_name, overlay_val)
    return base


def _try_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(str(val).replace(",", "."))
    except (ValueError, TypeError):
        return None


def _try_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(float(str(val).replace(",", "").replace("'", "")))
    except (ValueError, TypeError):
        return None


def _map_schema_type(schema_type: str) -> Optional[str]:
    return {
        "Apartment": "apartment",
        "House": "single_family",
        "SingleFamilyResidence": "single_family",
        "Residence": "apartment",
    }.get(schema_type)
