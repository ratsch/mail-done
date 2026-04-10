"""
Slack alerts for high-scoring property listings.

Uses email-to-Slack channel integration:
- Klosters (Scenario A) → #klosters channel
- Zurich (Scenarios B/C/D) → #immobilie channel

Sends from ETH work email (raetsch@ethz.ch) via SMTP.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Slack channel email addresses (email-to-channel integration)
SLACK_CHANNELS = {
    "A": "klosters-aaaan5odmmnbpqnttrmnactunu@no-gu.slack.com",
    "B": "immobilie-aaaaa6ckq4widejesj3uzx2cxm@no-gu.slack.com",
    "C": "immobilie-aaaaa6ckq4widejesj3uzx2cxm@no-gu.slack.com",
    "D": "immobilie-aaaaa6ckq4widejesj3uzx2cxm@no-gu.slack.com",
}

# Alert thresholds
ALERT_SCORE_THRESHOLD = 6  # overall_recommendation >= 6
ALERT_ON_UNDERVALUED = True  # Always alert if houzy_assessment == "undervalued"


def should_alert(listing) -> bool:
    """Check if listing warrants a Slack alert."""
    if listing.overall_recommendation and listing.overall_recommendation >= ALERT_SCORE_THRESHOLD:
        return True
    if ALERT_ON_UNDERVALUED and listing.houzy_assessment == "undervalued":
        return True
    return False


def format_alert(listing, scenario: str) -> Dict[str, str]:
    """Format a Slack alert message for a listing."""
    # Stars based on score
    score = listing.overall_recommendation or 0
    stars = "⭐" * min(score, 5) if score >= 6 else ""

    # Subject line
    address = listing.address or listing.municipality or "New listing"
    subject = f"{stars} {address}"
    if listing.price_known and listing.price_chf:
        subject += f" — CHF {listing.price_chf:,.0f}"
    if listing.houzy_assessment == "undervalued":
        subject += " 🔥 UNDERVALUED"

    # Body
    lines = [
        f"📍 {address}",
        f"   PLZ: {listing.plz or '?'} {listing.municipality or ''}",
        "",
    ]

    # Price + Houzy comparison
    if listing.price_known and listing.price_chf:
        lines.append(f"💰 CHF {listing.price_chf:,.0f}")
        if listing.houzy_mid:
            delta = listing.price_vs_houzy_pct
            if delta:
                indicator = "✅" if delta <= 100 else "⚠️"
                lines.append(f"   Houzy: CHF {listing.houzy_mid:,.0f} ({indicator} {delta:.0f}%)")
    else:
        lines.append("💰 Preis auf Anfrage")

    lines.append("")

    # Key details
    details = []
    if listing.rooms:
        details.append(f"{listing.rooms} Zi.")
    if listing.living_area_sqm:
        details.append(f"{listing.living_area_sqm} m²")
    if listing.property_type:
        type_map = {"apartment": "Wohnung", "single_family": "EFH", "multi_family": "MFH"}
        details.append(type_map.get(listing.property_type, listing.property_type))
    if listing.year_built:
        details.append(f"Bj. {listing.year_built}")
    if details:
        lines.append(f"🏠 {' | '.join(details)}")

    lines.append("")

    # Scores
    if listing.overall_recommendation:
        lines.append(f"📊 Score: {listing.overall_recommendation}/10 (Scenario {scenario})")

    # Red flags
    if listing.red_flags:
        flags = listing.red_flags[:3] if isinstance(listing.red_flags, list) else []
        if flags:
            lines.append(f"⚠️ {', '.join(flags)}")

    # Highlights
    if listing.highlights:
        highlights = listing.highlights[:3] if isinstance(listing.highlights, list) else []
        if highlights:
            lines.append(f"✨ {', '.join(highlights)}")

    lines.append("")

    # Link
    if listing.listing_url:
        lines.append(f"🔗 {listing.listing_url}")

    lines.append(f"📱 Source: {listing.listing_source or 'unknown'}")

    body = "\n".join(lines)
    return {"subject": subject, "body": body}


def send_alert(listing, scenario: str, smtp_config: Dict) -> bool:
    """
    Send listing alert to Slack channel via email.

    Args:
        listing: PropertyListing instance
        scenario: "A", "B", "C", or "D"
        smtp_config: {host, port, username, password, from_address}

    Returns:
        True if sent successfully
    """
    channel_email = SLACK_CHANNELS.get(scenario)
    if not channel_email:
        logger.warning(f"No Slack channel for scenario {scenario}")
        return False

    alert = format_alert(listing, scenario)

    try:
        msg = MIMEText(alert["body"], "plain", "utf-8")
        msg["Subject"] = alert["subject"]
        msg["From"] = smtp_config["from_address"]
        msg["To"] = channel_email

        with smtplib.SMTP(smtp_config["host"], smtp_config["port"]) as server:
            server.starttls()
            server.login(smtp_config["username"], smtp_config["password"])
            server.send_message(msg)

        logger.info(f"Slack alert sent to #{scenario}: {alert['subject'][:60]}")
        return True

    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")
        return False


def send_alerts_for_listing(listing, smtp_config: Dict) -> int:
    """
    Send alerts for all applicable scenarios of a listing.

    Returns number of alerts sent.
    """
    if not should_alert(listing):
        return 0

    scenarios = listing.applicable_scenarios or []
    if not scenarios:
        # Infer scenario from PLZ
        plz = listing.plz or ""
        if plz.startswith("72"):  # Klosters/Davos region
            scenarios = ["A"]
        elif plz.startswith("80"):  # Zurich
            scenarios = ["D"]  # Default to apartment
        else:
            scenarios = ["D"]

    sent = 0
    for scenario in scenarios:
        if send_alert(listing, scenario, smtp_config):
            sent += 1

    return sent
