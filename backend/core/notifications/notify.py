"""Notify-worthy email emitter.

When the classifier flags an email as ``notify_worthy`` (see
``NOTIFY_WORTHY DETECTION`` in the classifier prompt) AND the flag
survives Stage-2 verification, this module emits a short email to the
configured Slack channel address so the user gets a real-time push.

The trigger is not limited to "urgent" emails — it includes major
positive outcomes (grants awarded, top-venue accepts, prizes),
major negative outcomes (proposals rejected, paper rejections,
compliance flags) and high-stakes decisions. See the classifier
prompt for the exact four patterns.

Design choices:
- Email channel, not HTTP: Slack's per-channel email address gives us a
  reliable push path without adding an OAuth/webhook integration.
- Plain text, not HTML: Slack renders the channel-email plaintext body
  cleanly; HTML adds noise with no benefit.
- One notification per email, ever. A ``notify_sent_at`` timestamp is
  stored in ``category_metadata`` so re-runs (``--reprocess``) do not
  re-notify.
- No rate limiting during initial calibration — we want full visibility
  into what the classifier flags so the prompt can be tuned.
- Kill switch: ``MAIL_DONE_NOTIFY_ENABLED=false`` disables emission.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import String, cast, func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class NotifyConfig:
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    from_name: str
    from_email: str
    to_address: str
    use_tls: bool = True
    rate_limit_per_hour: int = 3
    enabled: bool = True


def load_notify_config() -> Optional[NotifyConfig]:
    """Load config from environment; return None if notifications disabled or misconfigured."""
    if os.getenv("MAIL_DONE_NOTIFY_ENABLED", "true").lower() in ("false", "0", "no"):
        logger.debug("Notifier disabled via MAIL_DONE_NOTIFY_ENABLED")
        return None

    to_address = os.getenv(
        "MAIL_DONE_NOTIFY_TO",
        "urgent_notificaton-aaaat5jiytwwot2ous45vf4ery@ratschlab.slack.com",
    )
    smtp_host = os.getenv("MAIL_DONE_NOTIFY_SMTP_HOST", "bmail7.sui-inter.net")
    smtp_port = int(os.getenv("MAIL_DONE_NOTIFY_SMTP_PORT", "587"))
    smtp_username = os.getenv("MAIL_DONE_NOTIFY_SMTP_USERNAME") or os.getenv("SMTP_USERNAME_WORK")
    smtp_password = os.getenv("MAIL_DONE_NOTIFY_SMTP_PASSWORD") or os.getenv("SMTP_PASSWORD_WORK")
    from_email = os.getenv("MAIL_DONE_NOTIFY_FROM", smtp_username or "")
    from_name = os.getenv("MAIL_DONE_NOTIFY_FROM_NAME", "mail-done notify")

    if not (smtp_host and smtp_username and smtp_password and to_address):
        logger.warning(
            "Notifier misconfigured "
            "(need SMTP_USERNAME_WORK / SMTP_PASSWORD_WORK / MAIL_DONE_NOTIFY_TO). "
            "Notifications disabled."
        )
        return None

    return NotifyConfig(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        from_name=from_name,
        from_email=from_email,
        to_address=to_address,
        rate_limit_per_hour=int(os.getenv("MAIL_DONE_NOTIFY_RATE_PER_HOUR", "3")),
    )


def _recent_notification_count(db: Session, account_id: str, hours: int) -> int:
    """Count notify_sent_at timestamps in the last ``hours`` for this account.

    ``category_metadata`` is a PG ``json`` column (not ``jsonb``), so we can't
    use the ORM's ``.astext`` operator directly. Cheapest reliable path: cast
    to text and LIKE-filter on the serialized key before narrowing down in
    Python. Rate-limit checks run once per notify-worthy email (rare), so the
    overhead is immaterial.
    """
    from backend.core.database.models import Email, EmailMetadata  # lazy import

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(EmailMetadata.category_metadata)
        .join(Email, Email.id == EmailMetadata.email_id)
        .filter(Email.account_id == account_id)
        .filter(EmailMetadata.category_metadata.isnot(None))
        .filter(cast(EmailMetadata.category_metadata, String).like("%notify_sent_at%"))
        .all()
    )
    count = 0
    for (meta,) in rows:
        ts = (meta or {}).get("notify_sent_at")
        if ts and ts >= cutoff.isoformat():
            count += 1
    return count


def _already_notified(metadata) -> bool:
    """True if this email already had a notification sent (dup-guard for --reprocess)."""
    if not metadata or not metadata.category_metadata:
        return False
    return bool(metadata.category_metadata.get("notify_sent_at"))


def _format_body(
    *,
    notify_reason: str,
    category: str,
    urgency_score: Optional[int],
    deadline: Optional[str],
    deadline_consequence: Optional[str],
    from_display: str,
    date_str: str,
    subject: str,
    summary: str,
    stage_2_triggered: bool,
) -> str:
    verified = "Stage 2 verified" if stage_2_triggered else "Stage 1 only"
    lines = [f"Why: {notify_reason}", f"Category: {category}"]
    if urgency_score is not None:
        lines.append(f"Urgency: {urgency_score}/10")
    if deadline:
        consequence = f" — {deadline_consequence}" if deadline_consequence else ""
        lines.append(f"Deadline: {deadline}{consequence}")
    lines.extend([
        "",
        f"From:      {from_display}",
        f"Received:  {date_str}",
        f"Subject:   {subject}",
        "",
        "Summary:",
        summary or "(no summary)",
        "",
        f"— mail-done classifier ({verified})",
    ])
    return "\n".join(lines)


def send_notification(
    *,
    db: Session,
    account_id: str,
    email_metadata,
    email_db_row,
    ai_classification,
    stage_2_triggered: bool,
    dry_run: bool = False,
) -> Optional[str]:
    """Emit a notification email if this classification warrants one.

    Returns the Message-ID of the sent notification, ``"dry-run"`` when
    suppressed by dry-run, or ``None`` when no notification was emitted
    (not notify-worthy, already notified, rate-limited, or failed).
    """
    if not getattr(ai_classification, "notify_worthy", False):
        return None

    cfg = load_notify_config()
    if not cfg:
        return None

    if _already_notified(email_metadata):
        logger.debug(
            f"Skipping notification for {email_db_row.id}: already notified"
        )
        return None

    # Rate limit intentionally not enforced yet — we want full visibility
    # into what the classifier flags while we calibrate. Per-email dedup
    # (notify_sent_at in category_metadata) still prevents duplicate
    # notifications for the same email. Re-enable by setting
    # MAIL_DONE_NOTIFY_RATE_PER_HOUR in env and re-introducing the check.

    if dry_run:
        logger.info(
            f"DRY-RUN: would notify for {email_db_row.id}: {ai_classification.notify_reason}"
        )
        return "dry-run"

    # Build the email.
    notify_reason = (ai_classification.notify_reason or "(no reason given)")[:200]
    subject = f"[mail-done] {notify_reason}"

    from_display = (
        f"{email_db_row.from_name} <{email_db_row.from_address}>"
        if email_db_row.from_name
        else (email_db_row.from_address or "(unknown sender)")
    )
    date_str = email_db_row.date.isoformat(timespec="seconds") if email_db_row.date else "(unknown)"

    body = _format_body(
        notify_reason=notify_reason,
        category=ai_classification.category,
        urgency_score=getattr(ai_classification, "urgency_score", None),
        deadline=getattr(ai_classification, "deadline", None),
        deadline_consequence=getattr(ai_classification, "deadline_consequence", None),
        from_display=from_display,
        date_str=date_str,
        subject=email_db_row.subject or "(no subject)",
        summary=ai_classification.summary or "",
        stage_2_triggered=stage_2_triggered,
    )

    # Send via existing SMTPSender. Using the Work account by default —
    # the channel address is off ratschlab.org so sending from the work
    # SMTP keeps the SPF/DKIM alignment clean.
    from backend.core.email.smtp_sender import SMTPSender

    sender = SMTPSender(
        smtp_host=cfg.smtp_host,
        smtp_port=cfg.smtp_port,
        smtp_username=cfg.smtp_username,
        smtp_password=cfg.smtp_password,
        from_name=cfg.from_name,
        from_email=cfg.from_email,
        use_tls=cfg.use_tls,
    )
    message_id = sender.send_email(
        to_address=cfg.to_address,
        subject=subject,
        body=body,
    )

    if not message_id:
        logger.error(
            f"Notify SMTP failure for email {email_db_row.id} "
            f"({notify_reason}); will retry on next reprocess"
        )
        return None

    # Persist dedup marker.
    from sqlalchemy.orm.attributes import flag_modified

    existing = dict(email_metadata.category_metadata or {})
    existing["notify_sent_at"] = datetime.utcnow().isoformat(timespec="seconds")
    existing["notify_message_id"] = message_id
    email_metadata.category_metadata = existing
    flag_modified(email_metadata, "category_metadata")

    logger.info(
        f"✉️  Notification sent for email {email_db_row.id}: "
        f"{notify_reason} (message_id={message_id})"
    )
    return message_id
