"""
Houzy REST API Client — Swiss property valuation via FPRE.

Standalone module. No dependencies on other real_estate modules.
Requires: httpx

Usage:
    from backend.core.real_estate.houzy_client import HouzyClient

    client = HouzyClient()
    await client.login()
    properties = await client.list_properties()
    val = await client.get_valuation("e4c978ab-...")

Standalone test:
    python -m backend.core.real_estate.houzy_client
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HouzyLocationScores:
    """Houzy/FPRE location sub-scores (each 0-5)."""
    gesamt: float = 0.0
    besonnung: float = 0.0
    sicht: float = 0.0
    image: float = 0.0
    laerm: float = 0.0
    oev: float = 0.0
    freizeit: float = 0.0
    strassenanbindung: float = 0.0
    dienstleistungen: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "HouzyLocationScores":
        if not data:
            return cls()
        return cls(
            gesamt=data.get("gesamt", data.get("overall", 0.0)),
            besonnung=data.get("besonnung", data.get("sun", 0.0)),
            sicht=data.get("sicht", data.get("view", 0.0)),
            image=data.get("image", data.get("imageQuartier", 0.0)),
            laerm=data.get("laerm", data.get("noise", 0.0)),
            oev=data.get("oev", data.get("publicTransport", 0.0)),
            freizeit=data.get("freizeit", data.get("leisure", 0.0)),
            strassenanbindung=data.get("strassenanbindung", data.get("roadConnection", 0.0)),
            dienstleistungen=data.get("dienstleistungen", data.get("services", 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "gesamt": self.gesamt,
            "besonnung": self.besonnung,
            "sicht": self.sicht,
            "image": self.image,
            "laerm": self.laerm,
            "oev": self.oev,
            "freizeit": self.freizeit,
            "strassenanbindung": self.strassenanbindung,
            "dienstleistungen": self.dienstleistungen,
        }


@dataclass
class HouzyValuation:
    """Complete Houzy/FPRE Marktwertschätzung result."""
    property_id: str
    address: str
    plz: str

    houzy_min: int
    houzy_mid: int
    houzy_max: int
    quality_pct: int

    zustand: float
    ausbaustandard: float
    living_area_sqm: int
    rooms: float
    property_type: str

    location_scores: HouzyLocationScores = field(default_factory=HouzyLocationScores)
    fetched_at: str = ""

    def price_vs_houzy_pct(self, asking_price: int) -> float:
        """Asking price as percentage of Houzy midpoint."""
        if self.houzy_mid <= 0:
            return 0.0
        return round((asking_price / self.houzy_mid) * 100, 1)

    def assessment(self, asking_price: int) -> str:
        """Assess asking price vs Houzy range."""
        if asking_price < self.houzy_min:
            return "below_minimum"
        elif asking_price <= self.houzy_mid:
            return "undervalued"
        elif asking_price <= self.houzy_max:
            return "fair"
        else:
            return "overpriced"


@dataclass
class HouzyProperty:
    """Summary of a property stored in Houzy account."""
    property_id: str
    address: str
    plz: str
    municipality: str
    property_type: str
    raw_data: dict = field(default_factory=dict, repr=False)


# =============================================================================
# Client
# =============================================================================

class HouzyClient:
    """
    Houzy REST API client.

    Rate-limited: minimum 3 seconds between requests.
    Caches session and valuation results.
    """

    API_BASE = "https://app.houzy.ch/api-v1"
    MIN_REQUEST_INTERVAL = 3.0
    SESSION_CACHE_DIR = Path.home() / ".cache" / "houzy-client"

    # realEstateType UUID that Houzy actually uses for computing
    # valuations. Empirically verified: EVERY successful market-value
    # response in the cache has realEstateType set to this UUID, and
    # every failing one has realEstateType=null. Applies to both
    # apartments and detached houses — confirmed by manually
    # recreating a Klosters house in the Houzy UI, which resulted in
    # buildingType=1 + typeUuid=<this UUID> and a working valuation.
    #
    # The name "APARTMENT" is a misnomer but kept for backwards-
    # compatibility with earlier commits. In reality this is Houzy's
    # default "individually-owned residential property" type (STWE
    # umbrella — covers condos AND single-family homes).
    APARTMENT_REAL_ESTATE_TYPE_UUID = "a2af1c4a-9739-4807-a048-92920f59e10b"

    def __init__(self, email: str = None, password: str = None):
        self.email = email or os.getenv("PORTAL_HOUZY_EMAIL")
        self.password = password or os.getenv("PORTAL_HOUZY_PASSWORD")

        if not self.email or not self.password:
            raise ValueError(
                "Houzy credentials required. Set PORTAL_HOUZY_EMAIL and "
                "PORTAL_HOUZY_PASSWORD in .env or pass to constructor."
            )

        self._cookies: Optional[httpx.Cookies] = None
        self._headers: Dict[str, str] = {}
        self._last_request_time: float = 0
        self._valuation_cache: Dict[str, HouzyValuation] = {}
        self._authenticated = False

        self.SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Rate limiting
    # -------------------------------------------------------------------------

    async def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            wait = self.MIN_REQUEST_INTERVAL - elapsed
            logger.debug(f"Rate limiting: waiting {wait:.1f}s")
            await asyncio.sleep(wait)
        self._last_request_time = time.time()

    # -------------------------------------------------------------------------
    # HTTP helpers
    # -------------------------------------------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        """Build httpx client with current auth state."""
        return httpx.AsyncClient(
            cookies=self._cookies,
            headers=self._headers,
            timeout=30.0,
            follow_redirects=True,
        )

    async def _request(self, method: str, path: str,
                       json_data: dict = None, params: dict = None,
                       _retry: bool = True) -> dict:
        """Make authenticated API request with rate limiting."""
        if not self._authenticated and "/auth/" not in path:
            await self.login()

        await self._rate_limit()

        url = f"{self.API_BASE}{path}"
        logger.debug(f"Houzy API: {method} {path}")

        async with self._build_client() as client:
            resp = await client.request(method, url, json=json_data, params=params)

            if resp.status_code == 401 and _retry:
                logger.info("Houzy session expired, re-authenticating...")
                self._authenticated = False
                await self.login()
                return await self._request(method, path, json_data, params, _retry=False)

            if resp.status_code == 429:
                logger.warning("Houzy rate limit hit (429). Waiting 60s...")
                await asyncio.sleep(60)
                raise RuntimeError("Houzy rate limit exceeded. Try again later.")

            resp.raise_for_status()

            # Update cookies from response (session maintenance)
            if resp.cookies:
                if self._cookies is None:
                    self._cookies = resp.cookies
                else:
                    self._cookies.update(resp.cookies)

            ct = resp.headers.get("content-type", "")
            if "application/json" in ct and resp.content:
                return resp.json()
            return {}

    async def _get(self, path: str, **kwargs) -> dict:
        return await self._request("GET", path, **kwargs)

    async def _post(self, path: str, json_data: dict = None, **kwargs) -> dict:
        return await self._request("POST", path, json_data=json_data, **kwargs)

    async def _put(self, path: str, json_data: dict = None, **kwargs) -> dict:
        return await self._request("PUT", path, json_data=json_data, **kwargs)

    async def _patch(self, path: str, json_data: dict = None, **kwargs) -> dict:
        return await self._request("PATCH", path, json_data=json_data, **kwargs)

    async def _delete(self, path: str, **kwargs) -> dict:
        return await self._request("DELETE", path, **kwargs)

    # -------------------------------------------------------------------------
    # Debug: save raw API responses
    # -------------------------------------------------------------------------

    def _save_debug(self, name: str, data) -> Path:
        """Save raw API response for field mapping discovery."""
        debug_file = self.SESSION_CACHE_DIR / f"{name}.json"
        with open(debug_file, "w") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        logger.debug(f"Saved debug response: {debug_file}")
        return debug_file

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    async def login(self) -> dict:
        """Authenticate and store session cookies."""
        logger.info(f"Logging into Houzy as {self.email}...")

        await self._rate_limit()

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Step 1: check-user (returns loginType needed for step 2)
            await self._rate_limit()
            check_resp = None
            try:
                check_resp = await client.post(
                    f"{self.API_BASE}/auth/check-user",
                    json={"email": self.email}
                )
            except Exception:
                pass  # Not critical — will use default loginType

            # Step 2: login (loginType from check-user response, typically "Default")
            login_type = "Default"
            if r1_data := (check_resp.json() if check_resp.status_code == 201 else {}):
                login_type = r1_data.get("loginType", "Default")

            await self._rate_limit()
            resp = await client.post(
                f"{self.API_BASE}/auth/login",
                json={
                    "email": self.email,
                    "password": self.password,
                    "loginType": login_type,
                }
            )
            resp.raise_for_status()

            self._cookies = resp.cookies
            self._authenticated = True

            # Extract auth token from response body
            result = {}
            if "application/json" in resp.headers.get("content-type", ""):
                result = resp.json()
                token = (
                    result.get("access_token") or
                    result.get("accessToken") or
                    result.get("token")
                )
                if token:
                    self._headers["Authorization"] = f"Bearer {token}"
                    logger.debug("Bearer token extracted from login response")

            self._save_debug("login_response", result)
            logger.info("Houzy login successful")
            return result

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    async def list_properties(self) -> List[HouzyProperty]:
        """List all properties saved in the Houzy account."""
        data = await self._get("/real-estate")
        self._save_debug("real_estate_list", data)

        properties = []
        items = data if isinstance(data, list) else data.get("items", data.get("realEstates", []))

        for p in items if isinstance(items, list) else []:
            # Address can be nested dict or flat fields
            addr = p.get("address", {})
            if isinstance(addr, dict):
                street = addr.get("street", "")
                nr = addr.get("nr", "")
                plz = str(addr.get("zip", ""))
                city = addr.get("city", "")
            else:
                street = p.get("street", "")
                nr = p.get("nr", "")
                plz = str(p.get("zip", p.get("plz", "")))
                city = p.get("city", p.get("municipality", ""))

            address = f"{street} {nr}".strip() if street else ""

            properties.append(HouzyProperty(
                property_id=str(p.get("uuid", p.get("id", p.get("_id", "")))),
                address=address,
                plz=plz,
                municipality=city,
                property_type=p.get("type", p.get("propertyType", "")),
                raw_data=p,
            ))

        logger.info(f"Houzy: {len(properties)} properties in account")
        return properties

    async def create_property(self,
                               street: str,
                               nr: str,
                               plz: str,
                               municipality: str = "",
                               property_type: str = "apartment",
                               living_area_sqm: int = 100,
                               rooms: float = 4.0,
                               year_built: int = 2000,
                               zustand: float = 3.0,
                               ausbaustandard: float = 3.0,
                               floor: int = 1,
                               ) -> str:
        """
        Create a new property in Houzy and return its property_id (UUID).

        Two-step process:
        1. POST /real-estate/open-data-based-on-the-address — pre-fill from Swiss cadastral
        2. POST /real-estate/create — create with full parameters

        Args:
            street: Street name (e.g., "Pestalozzistrasse")
            nr: Street number (e.g., "33")
            plz: Postal code (e.g., "8032")
            municipality: City/village (e.g., "Zürich")
            property_type: "apartment" or "house"
            living_area_sqm: Living area in m²
            rooms: Number of rooms (e.g., 3.5)
            year_built: Construction year
            zustand: Condition 1-5
            ausbaustandard: Finish quality 1-5
            floor: Floor number (for apartments)

        Returns:
            Houzy property UUID string
        """
        address_payload = {
            "street": street,
            "city": municipality,
            "nr": nr,
            "zip": plz,
            "postingUuid": None,
        }

        # Step 1: Get open data for the address (construction year, area, etc.)
        open_data = {}
        try:
            open_data = await self._post(
                "/real-estate/open-data-based-on-the-address",
                json_data=address_payload,
            )
            self._save_debug("open_data_response", open_data)
            logger.info(f"Houzy open data for {street} {nr}, {plz}: {list(open_data.keys()) if isinstance(open_data, dict) else 'ok'}")
        except Exception as e:
            logger.warning(f"Houzy open data lookup failed (non-critical): {e}")

        # Use open data to fill defaults, but our parameters take priority
        area_property = open_data.get("areaProperty", 0) if isinstance(open_data, dict) else 0
        area_base = open_data.get("areaBase", 0) if isinstance(open_data, dict) else 0
        num_floors = open_data.get("numberOfFloors", 4) if isinstance(open_data, dict) else 4
        roof_type = open_data.get("roofType", "flatroof") if isinstance(open_data, dict) else "flatroof"
        construction_year = year_built or (open_data.get("constructionYear", 2000) if isinstance(open_data, dict) else 2000)

        # CRITICAL: Houzy's valuation endpoint only returns a prediction
        # when the property has buildingType=1 AND realEstateType set to
        # the STWE/EFH UUID — this is true for BOTH apartments and single-
        # family houses. Empirically verified by comparing every successful
        # vs failing market-value response in the cache, and by manually
        # recreating a Klosters house in the Houzy UI (which results in
        # exactly this combination).
        #
        # The previous code used buildingType=2 for houses, which produced
        # calculationState="None" every time for rural addresses.
        building_type = 1
        real_estate_type = self.APARTMENT_REAL_ESTATE_TYPE_UUID
        # condominiumPositionTypeId: floor position (1=EG, 2=OG, 3=DG, etc.)
        # For houses the floor is usually 0/EG; apartments can be higher.
        condo_position = max(1, floor + 1) if floor is not None else 2  # 2 = OG
        condo_types = []

        # Step 2: Create the property
        create_payload = {
            "userType": 3,  # 3 = buyer/interested
            "valuationIntention": None,
            "uuid": "new",
            "numberOfRooms": rooms,
            "isSubmitted": False,
            "address": {
                "street": street,
                "city": municipality,
                "nr": nr,
                "zip": plz,
            },
            "areaProperty": area_property,
            "buildingType": building_type,
            "condition": int(zustand),
            "condominiumPositionTypeId": condo_position,
            "condominiumTypeIds": condo_types,
            "constructionYear": construction_year,
            "livingSpace": living_area_sqm,
            "numberOfFloors": num_floors,
            "realEstateType": real_estate_type,
            "roofType": roof_type,
            "standard": int(ausbaustandard),
            "addOrBrowseProperties": "AddObject",
            "areaBase": area_base,
        }

        result = await self._post("/real-estate/create", json_data=create_payload)
        self._save_debug("create_response", result)

        # Extract UUID from response
        new_uuid = None
        if isinstance(result, dict):
            new_uuid = result.get("uuid", result.get("id", result.get("_id")))
        elif isinstance(result, str):
            new_uuid = result

        if not new_uuid:
            # The create endpoint may not return the UUID directly.
            # Fall back to listing properties and finding the new one.
            logger.info("Create response didn't include UUID, searching property list...")
            await asyncio.sleep(2)
            props = await self.list_properties()
            for prop in props:
                if prop.plz == plz and street.lower() in prop.address.lower():
                    new_uuid = prop.property_id
                    break

        if not new_uuid:
            raise RuntimeError(f"Failed to get UUID for newly created property {street} {nr}, {plz}")

        logger.info(f"Houzy property created: {new_uuid} — {street} {nr}, {plz} {municipality}")
        return new_uuid

    async def delete_property(self, property_id: str) -> dict:
        """Delete a property from the Houzy account."""
        result = await self._delete(f"/real-estate/{property_id}")
        logger.info(f"Houzy property deleted: {property_id[:8]}")
        return result

    async def update_property_params(self,
                                      property_id: str,
                                      zustand: float = None,
                                      ausbaustandard: float = None,
                                      living_area_sqm: int = None,
                                      rooms: float = None,
                                      floor: int = None,
                                      ) -> dict:
        """
        Update property parameters that affect the valuation.

        Changing Zustand from 2/5 to 4/5 can shift the valuation by 30-40%.
        Uses PUT /real-estate/{uuid} with the full property payload.
        """
        # First, get current property data
        props = await self.list_properties()
        current = None
        for p in props:
            if p.property_id == property_id:
                current = p.raw_data
                break

        if not current:
            raise ValueError(f"Property {property_id} not found in Houzy account")

        # Update only the specified fields
        if zustand is not None:
            current["condition"] = int(zustand)
        if ausbaustandard is not None:
            current["standard"] = int(ausbaustandard)
        if living_area_sqm is not None:
            current["areaLiving"] = living_area_sqm
        if rooms is not None:
            current["rooms"] = str(rooms)
        if floor is not None and current.get("condominiumPositionTypeId") is not None:
            current["condominiumPositionTypeId"] = floor

        # Try PUT first, then PATCH
        try:
            result = await self._put(f"/real-estate/{property_id}", json_data=current)
        except Exception:
            result = await self._patch(f"/real-estate/{property_id}", json_data=current)

        self._save_debug(f"update_{property_id[:8]}", result)
        logger.info(f"Houzy property updated: {property_id[:8]} — condition={current.get('condition')}, standard={current.get('standard')}")
        return result

    # -------------------------------------------------------------------------
    # Valuation
    # -------------------------------------------------------------------------

    async def get_valuation(self, property_id: str,
                            use_cache: bool = True) -> HouzyValuation:
        """
        Get Marktwertschätzung for a property.

        Uses the finance/current-market-value endpoint.
        The property_id here is the "buy real estate" UUID — the ID used
        in the buy context (from app-session-data.lastVisitedBuyRealEstateUuid
        or from the real-estate list).

        Args:
            property_id: Houzy buy-context UUID
            use_cache: Use cached result if available

        Returns:
            HouzyValuation with min/mid/max values, location scores, etc.
        """
        if use_cache and property_id in self._valuation_cache:
            logger.debug(f"Houzy valuation cache hit: {property_id[:8]}")
            return self._valuation_cache[property_id]

        data = await self._get(
            f"/finance/real-estate/{property_id}/current-market-value",
            params={"requestSource": "ToolPage"}
        )
        self._save_debug(f"market_value_{property_id[:8]}", data)

        valuation = self._parse_valuation(property_id, data)
        self._valuation_cache[property_id] = valuation

        logger.info(
            f"Houzy valuation: {valuation.address}, {valuation.plz} — "
            f"CHF {valuation.houzy_min:,} – {valuation.houzy_mid:,} – {valuation.houzy_max:,} "
            f"(quality: {valuation.quality_pct}%)"
        )

        return valuation

    def _parse_valuation(self, property_id: str, data: dict) -> HouzyValuation:
        """
        Parse HouzyValuation from /finance/real-estate/{id}/current-market-value response.

        Known response structure (verified 2026-03-22):
        {
            "prediction": 1590000,
            "predictionMin": 1479000,
            "predictionMax": 1701000,
            "accuracy": 3,                    # Maps to ~80% Schätzqualität
            "realEstateCondition": 4,          # Zustand 1-5
            "realEstateStandard": 3,           # Ausbaustandard 1-5
            "realEstateLivingSpace": 91,       # sqm
            "realEstateNumberOfRooms": "3.5",
            "realEstateAreaProperty": 3002,    # Land area sqm
            "buildingTypeId": 2,
            "condominiumPosition": 2,          # Floor/position for apartments
            "fpreRating": {
                "overallRating": 3.3,
                "tanning": 2.4,               # Besonnung
                "view": 3.5,                   # Sicht
                "neighborhoodImage": 3.7,      # Image
                "noisePollution": 4.8,         # Lärmbelastung
                "publicTransport": 2.6,        # ÖV
                "recreation": 2.7,             # Freizeit
                "roadConnection": 2.5,
                "services": 1.5
            }
        }
        """
        if not isinstance(data, dict):
            logger.warning(f"Unexpected market-value response type: {type(data)}")
            return self._empty_valuation(property_id)

        # Valuation range
        houzy_mid = data.get("prediction", 0)
        houzy_min = data.get("predictionMin", 0)
        houzy_max = data.get("predictionMax", 0)

        # Accuracy: Houzy uses 1-5 scale, we convert to percentage
        accuracy_raw = data.get("accuracy", 0)
        accuracy_map = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}
        quality_pct = accuracy_map.get(accuracy_raw, accuracy_raw * 20 if accuracy_raw else 0)

        # Property parameters
        zustand = float(data.get("realEstateCondition", 3))
        ausbau = float(data.get("realEstateStandard", 3))
        sqm = int(data.get("realEstateLivingSpace", 0))
        rooms_raw = data.get("realEstateNumberOfRooms", "0")
        rooms = float(rooms_raw) if rooms_raw else 0.0

        # FPRE location scores
        fpre = data.get("fpreRating", {}) or {}
        location_scores = HouzyLocationScores(
            gesamt=fpre.get("overallRating", 0.0),
            besonnung=fpre.get("tanning", 0.0),
            sicht=fpre.get("view", 0.0),
            image=fpre.get("neighborhoodImage", 0.0),
            laerm=fpre.get("noisePollution", 0.0),
            oev=fpre.get("publicTransport", 0.0),
            freizeit=fpre.get("recreation", 0.0),
            strassenanbindung=fpre.get("roadConnection", 0.0),
            dienstleistungen=fpre.get("services", 0.0),
        )

        return HouzyValuation(
            property_id=property_id,
            address="",  # Not in market-value response — fill from property list
            plz="",
            houzy_min=houzy_min,
            houzy_mid=houzy_mid,
            houzy_max=houzy_max,
            quality_pct=quality_pct,
            zustand=zustand,
            ausbaustandard=ausbau,
            living_area_sqm=sqm,
            rooms=rooms,
            property_type="",
            location_scores=location_scores,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    def _empty_valuation(self, property_id: str) -> HouzyValuation:
        return HouzyValuation(
            property_id=property_id, address="", plz="",
            houzy_min=0, houzy_mid=0, houzy_max=0, quality_pct=0,
            zustand=0, ausbaustandard=0, living_area_sqm=0, rooms=0,
            property_type="",
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    # -------------------------------------------------------------------------
    # Field extraction helpers (handle unknown API response structure)
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_int(data: dict, keys: list, default: int = 0) -> int:
        for key in keys:
            val = data.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    continue
        return default

    @staticmethod
    def _extract_float(data: dict, keys: list, default: float = 0.0) -> float:
        for key in keys:
            val = data.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return default

    @staticmethod
    def _extract_str(data: dict, keys: list) -> Optional[str]:
        for key in keys:
            val = data.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
        return None

    # -------------------------------------------------------------------------
    # Convenience: end-to-end valuation for a new address
    # -------------------------------------------------------------------------

    async def valuate_address(self,
                               street: str,
                               nr: str,
                               plz: str,
                               municipality: str = "",
                               property_type: str = "apartment",
                               living_area_sqm: int = 100,
                               rooms: float = 4.0,
                               year_built: int = 2000,
                               zustand: float = 3.0,
                               ausbaustandard: float = 3.0,
                               ) -> HouzyValuation:
        """
        End-to-end: find or create property and return valuation.

        This is the main entry point used by reprocess_listings.py (Plan D).

        Args:
            street: Street name (e.g., "Bildweg")
            nr: Street number (e.g., "12")
            plz: Postal code
            municipality: City/village
            ... (other params for property creation)

        Returns:
            HouzyValuation with min/mid/max, location scores, etc.
        """
        full_address = f"{street} {nr}".strip()

        # Check if already in account
        existing = await self._find_property_by_address(full_address, plz)
        if existing:
            # The existing Houzy property must have the STWE/EFH pattern
            # (buildingType=1 + typeUuid=<apartment UUID>) to produce a
            # valid valuation — this applies to BOTH apartments and houses.
            # Delete + recreate on any mismatch.
            raw = existing.raw_data or {}
            existing_building_type = raw.get("buildingType", {})
            if isinstance(existing_building_type, dict):
                existing_building_type_id = existing_building_type.get("id")
            else:
                existing_building_type_id = existing_building_type
            existing_type_uuid = raw.get("typeUuid")

            expected_bt = 1
            expected_uuid = self.APARTMENT_REAL_ESTATE_TYPE_UUID

            mismatch = (
                existing_building_type_id != expected_bt
                or existing_type_uuid != expected_uuid
            )

            if mismatch:
                logger.info(
                    f"Existing Houzy property {existing.property_id[:8]} "
                    f"mismatch: buildingType={existing_building_type_id} "
                    f"typeUuid={existing_type_uuid} — recreating as "
                    f"{property_type} (buildingType={expected_bt})"
                )
                try:
                    await self.delete_property(existing.property_id)
                except Exception as e:
                    logger.warning(f"Failed to delete old Houzy property: {e}")
                existing = None
            else:
                logger.info(
                    f"Found existing Houzy property: "
                    f"{existing.property_id[:8]} — {full_address}"
                )
                return await self.get_valuation(existing.property_id)

        # Create new
        property_id = await self.create_property(
            street=street, nr=nr, plz=plz, municipality=municipality,
            property_type=property_type, living_area_sqm=living_area_sqm,
            rooms=rooms, year_built=year_built,
            zustand=zustand, ausbaustandard=ausbaustandard,
        )

        # Wait for Houzy to compute the valuation (async on their side)
        # Retry up to 3 times with increasing delay
        valuation = None
        for attempt in range(3):
            delay = 5 + attempt * 5  # 5s, 10s, 15s
            logger.info(f"Waiting {delay}s for Houzy valuation (attempt {attempt + 1}/3)...")
            await asyncio.sleep(delay)

            valuation = await self.get_valuation(property_id, use_cache=False)
            if valuation.houzy_mid > 0:
                break
            logger.info(f"Valuation not ready yet (calculationState may be 'None')")

        if valuation is None:
            valuation = self._empty_valuation(property_id)

        # Fill address from our input since the valuation endpoint doesn't return it
        valuation.address = full_address
        valuation.plz = plz
        return valuation

    async def _find_property_by_address(self, address: str, plz: str) -> Optional[HouzyProperty]:
        """Check if a property with this address exists in the account."""
        properties = await self.list_properties()
        addr_lower = address.lower().strip()
        for prop in properties:
            if prop.plz == plz and addr_lower in prop.address.lower():
                return prop
        return None

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def clear_cache(self):
        """Clear the valuation cache."""
        self._valuation_cache.clear()
        logger.debug("Houzy valuation cache cleared")


# =============================================================================
# Standalone test
# =============================================================================

async def _standalone_test():
    """
    Test Houzy API and discover response structures.

    Run: python -m backend.core.real_estate.houzy_client
    Or:  cd ~/git/services/mail-done && poetry run python -m backend.core.real_estate.houzy_client
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load .env from mail-done-config if available
    try:
        from dotenv import load_dotenv
        for env_path in [
            Path(__file__).parent.parent.parent.parent.parent / "mail-done-config" / ".env",
            Path.home() / "git" / "services" / "mail-done-config" / ".env",
            Path.cwd() / ".env",
        ]:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded .env from: {env_path}")
                break
    except ImportError:
        pass

    client = HouzyClient()

    # Step 1: Login
    print("\n" + "=" * 60)
    print("STEP 1: Login")
    print("=" * 60)
    await client.login()
    print("OK")

    # Step 2: List properties
    print("\n" + "=" * 60)
    print("STEP 2: List Properties")
    print("=" * 60)
    properties = await client.list_properties()
    for p in properties:
        print(f"  [{p.property_id[:8]}]  {p.address}, {p.plz} {p.municipality}  ({p.property_type})")

    if not properties:
        print("  No properties found in account.")
        print("\n  Raw response saved to ~/.cache/houzy-client/real_estate_list.json")
        print("  Check the file to understand the response structure.")

    # Step 3: Get valuation for known property
    print("\n" + "=" * 60)
    print("STEP 3: Get Valuation")
    print("=" * 60)

    known_id = "e4c978ab-1c6d-4868-9d59-5d3d9e79a25a"  # Bildweg 12

    if properties:
        # Use first property if known_id not in list
        target_id = known_id
        if not any(p.property_id == known_id for p in properties):
            target_id = properties[0].property_id
            print(f"  Known ID not found, using first property: {target_id[:8]}")
    else:
        target_id = known_id

    try:
        val = await client.get_valuation(target_id, use_cache=False)
        print(f"  Address:    {val.address}, {val.plz}")
        print(f"  Valuation:  CHF {val.houzy_min:,} – {val.houzy_mid:,} – {val.houzy_max:,}")
        print(f"  Quality:    {val.quality_pct}%")
        print(f"  Zustand:    {val.zustand}/5")
        print(f"  Ausbau:     {val.ausbaustandard}/5")
        print(f"  Location:   {val.location_scores.to_dict()}")

        if val.houzy_mid == 0:
            print("\n  WARNING: Valuation returned 0 — field mapping needs adjustment.")
            print(f"  Check raw response at: ~/.cache/houzy-client/dashboard_{target_id[:8]}.json")
    except Exception as e:
        print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("DEBUG FILES")
    print("=" * 60)
    cache_dir = client.SESSION_CACHE_DIR
    for f in sorted(cache_dir.glob("*.json")):
        size = f.stat().st_size
        print(f"  {f.name}  ({size:,} bytes)")

    print(f"\nInspect these files to verify/fix the field mapping in _parse_valuation().")
    print("Expected: dashboard_*.json should contain the valuation data.")


if __name__ == "__main__":
    asyncio.run(_standalone_test())
