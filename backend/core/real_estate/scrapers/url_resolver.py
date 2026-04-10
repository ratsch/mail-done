"""
URL resolver — follow tracking/redirect URLs to get the actual listing URL.

Handles SendGrid, Mailjet, and other email tracking services that wrap
listing URLs in redirect chains.
"""

import logging
from typing import Optional
from urllib.parse import urlparse, parse_qs, unquote

import httpx

logger = logging.getLogger(__name__)

# Known tracking domains that wrap real URLs
TRACKING_DOMAINS = [
    "sendgrid.net",
    "ct.sendgrid.net",
    "u.sendgrid.net",
    "mailjet.com",
    "email.homegate.ch",
    "notifications.homegate.ch",
    "click.mailerlite.com",
    "links.mail.comparis.ch",
]

# Portal domains we're interested in (final destination)
PORTAL_DOMAINS = [
    "homegate.ch",
    "immoscout24.ch",
    "comparis.ch",
    "newhome.ch",
    "engelvoelkers.com",
    "neho.ch",
    "betterhomes.ch",
    "sothebysrealty.com",
    "raiffeisen.ch",
]


async def resolve_tracking_url(url: str, max_redirects: int = 5) -> str:
    """
    Follow redirect chain to resolve the final URL.

    SendGrid URLs (e.g., ct.sendgrid.net/ls/click?upn=...) redirect
    to the actual listing URL. Follow the chain to get the real URL.

    Args:
        url: Possibly-wrapped tracking URL
        max_redirects: Maximum number of redirects to follow

    Returns:
        The final resolved URL, or the original if no redirects needed
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # If already a portal URL, return as-is
    if any(portal in domain for portal in PORTAL_DOMAINS):
        return url

    # If not a known tracking domain, return as-is
    if not any(tracker in domain for tracker in TRACKING_DOMAINS):
        return url

    logger.debug(f"Resolving tracking URL: {url[:80]}...")

    try:
        async with httpx.AsyncClient(
            follow_redirects=False,  # We follow manually to track the chain
            timeout=15.0,
        ) as client:
            current_url = url
            for _ in range(max_redirects):
                resp = await client.get(current_url)

                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location", "")
                    if location:
                        # Handle relative redirects
                        if location.startswith("/"):
                            parsed_current = urlparse(current_url)
                            location = f"{parsed_current.scheme}://{parsed_current.netloc}{location}"
                        current_url = location
                        logger.debug(f"  → {current_url[:80]}")

                        # Check if we've reached a portal URL
                        if any(portal in urlparse(current_url).netloc.lower() for portal in PORTAL_DOMAINS):
                            logger.debug(f"Resolved to: {current_url[:80]}")
                            return current_url
                    else:
                        break
                else:
                    # No more redirects
                    break

            logger.debug(f"Resolved to: {current_url[:80]}")
            return current_url

    except Exception as e:
        logger.warning(f"Failed to resolve URL {url[:60]}: {e}")
        return url


def extract_listing_urls_from_email(email_body: str) -> list[str]:
    """
    Extract all listing URLs from an email body.

    Finds URLs from known portals and tracking services.
    Returns deduplicated list of URLs (tracking URLs will be
    resolved later during scraping).
    """
    import re

    # Match URLs — be generous with the pattern
    url_pattern = r'https?://[^\s<>\[\]"\')\]}>]+'
    raw_urls = re.findall(url_pattern, email_body)

    listing_urls = []
    seen = set()

    for url in raw_urls:
        # Clean up common trailing characters
        url = url.rstrip(".,;:")

        # Skip if already seen
        if url in seen:
            continue
        seen.add(url)

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Direct portal URLs
        if any(portal in domain for portal in PORTAL_DOMAINS):
            listing_urls.append(url)
        # Tracking URLs (will be resolved later)
        elif any(tracker in domain for tracker in TRACKING_DOMAINS):
            listing_urls.append(url)

    return listing_urls
