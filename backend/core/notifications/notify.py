"""Notify-worthy email emitter.

When the classifier flags an email as ``notify_worthy`` (see
``NOTIFY_WORTHY DETECTION`` in the classifier prompt) AND the flag
survives Stage-2 verification, this module emits a short email to the
configured Slack channel address so the user gets a real-time push.

The trigger is not limited to "urgent" emails — it includes major
positive outcomes (grants awarded, top-venue accepts, prizes),
major negative outcomes (proposals rejected, paper rejections,
compliance flags), high-stakes decisions, and personal/family
emergencies. See the classifier prompt for the exact five patterns.

Design choices:
- Email channel, not HTTP: Slack's per-channel email address gives us a
  reliable push path without adding an OAuth/webhook integration.
- Plain text, not HTML: Slack renders the channel-email plaintext body
  cleanly; HTML adds noise with no benefit.
- One notification per email, ever (dup-guard: notify_sent_at column).
- Per-sender cooldown (configurable in vip_senders.yaml → notify_settings).
  Prevents Slack flood when a single person sends several emails in a row.
- Category blocklist (also in notify_settings): never notify for automated
  mail (newsletters, receipts, system notifications) even if the LLM
  somehow sets notify_worthy=true.
- Kill switch: ``MAIL_DONE_NOTIFY_ENABLED=false`` disables emission.

Transaction ordering
--------------------
The SMTP send happens OUTSIDE the main email-processing transaction. The
caller commits the classification + metadata first, then calls
``send_notification`` which opens its own session, sends the email, and
writes the dedup marker in a separate tiny transaction. Rationale:
previously send happened before the main commit, so a commit failure
after a successful SMTP send would roll back the dedup flag → duplicate
Slack message on retry. The two-commit pattern isolates that window.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Set

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
    # Cooldown / blocklist loaded from vip_senders.yaml → notify_settings.
    # Populated by `attach_vip_settings()` after VIPManager is available.
    cooldown_minutes_per_sender: int = 0
    category_blocklist: Set[str] = field(default_factory=set)
    enabled: bool = True


def load_notify_config() -> Optional[NotifyConfig]:
    """Load SMTP config from environment; return None if disabled or misconfigured."""
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
    )


def attach_vip_settings(cfg: NotifyConfig, vip_manager) -> NotifyConfig:
    """Copy cooldown + blocklist from a loaded VIPManager into the config."""
    ns = getattr(vip_manager, "notify_settings", None) or {}
    cfg.cooldown_minutes_per_sender = int(ns.get("cooldown_minutes_per_sender", 0) or 0)
    cfg.category_blocklist = set(ns.get("category_blocklist", set()) or set())
    return cfg


# ---------------------------------------------------------------------------
# Dedup / cooldown / blocklist checks
# ---------------------------------------------------------------------------


def _is_already_notified(metadata) -> bool:
    """True if this email already had a notification sent.

    Checks both the proper column (``notify_sent_at``) and the legacy JSON
    marker in ``category_metadata`` for rows written before the column was
    added. New rows write both fields so either source answers correctly.
    """
    if metadata is None:
        return False
    col = getattr(metadata, "notify_sent_at", None)
    if col:
        return True
    cm = getattr(metadata, "category_metadata", None) or {}
    return bool(cm.get("notify_sent_at"))


def _sender_in_cooldown(
    db: Session, account_id: str, from_address: str, minutes: int
) -> Optional[datetime]:
    """Return the timestamp of the most recent notification from this sender
    within the cooldown window, or None if cooldown doesn't block the send.
    """
    if not minutes or not from_address:
        return None
    from backend.core.database.models import Email, EmailMetadata  # lazy import

    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    row = (
        db.query(EmailMetadata.notify_sent_at)
        .join(Email, Email.id == EmailMetadata.email_id)
        .filter(Email.account_id == account_id)
        .filter(Email.from_address.ilike(from_address))
        .filter(EmailMetadata.notify_sent_at.isnot(None))
        .filter(EmailMetadata.notify_sent_at >= cutoff)
        .order_by(EmailMetadata.notify_sent_at.desc())
        .first()
    )
    return row[0] if row else None


def _category_blocked(category: Optional[str], blocklist: Set[str]) -> bool:
    """True if the classified category should never trigger a notification."""
    if not category or not blocklist:
        return False
    return category.lower() in blocklist


# ---------------------------------------------------------------------------
# Body formatting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main entry point — send + persist in its own transaction
# ---------------------------------------------------------------------------


def send_notification(
    *,
    account_id: str,
    email_id,
    ai_classification,
    stage_2_triggered: bool,
    vip_manager=None,
    dry_run: bool = False,
    db_factory=None,
) -> Optional[str]:
    """Emit a notification email if this classification warrants one.

    Opens its own DB session (via ``db_factory`` or the global ``get_db``)
    so that the send + dedup write happens OUTSIDE any caller's
    long-running transaction. Returns the Message-ID of the sent email,
    ``"dry-run"`` when suppressed by dry-run, or ``None`` when no
    notification was emitted (not notify-worthy, already notified,
    sender in cooldown, category blocklisted, or SMTP failed).
    """
    if not getattr(ai_classification, "notify_worthy", False):
        return None

    cfg = load_notify_config()
    if not cfg:
        return None
    if vip_manager is not None:
        attach_vip_settings(cfg, vip_manager)

    # Category-blocklist guard runs BEFORE any DB work — cheap reject.
    if _category_blocked(ai_classification.category, cfg.category_blocklist):
        logger.info(
            f"Skipping notification for {email_id}: category "
            f"{ai_classification.category!r} is on notify blocklist"
        )
        return None

    # Open our own session — the caller has typically already committed.
    if db_factory is None:
        from backend.core.database import get_db  # lazy import
        db = next(get_db())
    else:
        db = db_factory()
    try:
        from backend.core.database.models import Email, EmailMetadata  # lazy import

        email_row = db.query(Email).filter(Email.id == email_id).first()
        if email_row is None:
            logger.warning(f"Notification skipped: email {email_id} not in DB")
            return None
        metadata = db.query(EmailMetadata).filter(
            EmailMetadata.email_id == email_id
        ).first()

        # Per-email dedup — never send twice for the same email.
        if _is_already_notified(metadata):
            logger.debug(f"Skipping notification for {email_id}: already notified")
            return None

        # Per-sender cooldown — suppress if we just notified from the same
        # sender recently. Logged at INFO so the user can see why something
        # they might have expected didn't fire.
        if cfg.cooldown_minutes_per_sender and email_row.from_address:
            last = _sender_in_cooldown(
                db, account_id, email_row.from_address, cfg.cooldown_minutes_per_sender
            )
            if last is not None:
                delta = datetime.utcnow() - last
                logger.info(
                    f"Skipping notification for {email_id}: sender "
                    f"{email_row.from_address!r} is in cooldown "
                    f"(last notified {delta.total_seconds() / 60:.0f} min ago, "
                    f"window {cfg.cooldown_minutes_per_sender} min)"
                )
                return None

        if dry_run:
            logger.info(
                f"DRY-RUN: would notify for {email_id}: "
                f"{ai_classification.notify_reason}"
            )
            return "dry-run"

        # Build the email and send.
        notify_reason = (ai_classification.notify_reason or "(no reason given)")[:200]
        subject = f"[mail-done] {notify_reason}"

        from_display = (
            f"{email_row.from_name} <{email_row.from_address}>"
            if email_row.from_name
            else (email_row.from_address or "(unknown sender)")
        )
        date_str = (
            email_row.date.isoformat(timespec="seconds")
            if email_row.date
            else "(unknown)"
        )

        body = _format_body(
            notify_reason=notify_reason,
            category=ai_classification.category,
            urgency_score=getattr(ai_classification, "urgency_score", None),
            deadline=getattr(ai_classification, "deadline", None),
            deadline_consequence=getattr(ai_classification, "deadline_consequence", None),
            from_display=from_display,
            date_str=date_str,
            subject=email_row.subject or "(no subject)",
            summary=ai_classification.summary or "",
            stage_2_triggered=stage_2_triggered,
        )

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
                f"Notify SMTP failure for email {email_id} "
                f"({notify_reason}); will retry on next reprocess"
            )
            return None

        # Persist dedup marker. Write both the proper timestamp column
        # (fast indexed queries for cooldown/analytics) AND the JSON
        # marker (backward compatibility with any consumer still reading
        # category_metadata).
        if metadata is not None:
            now = datetime.utcnow()
            metadata.notify_sent_at = now
            metadata.notify_message_id = message_id
            from sqlalchemy.orm.attributes import flag_modified
            existing = dict(metadata.category_metadata or {})
            existing["notify_sent_at"] = now.isoformat(timespec="seconds")
            existing["notify_message_id"] = message_id
            metadata.category_metadata = existing
            flag_modified(metadata, "category_metadata")
            db.commit()
        else:
            logger.warning(
                f"Notification sent for {email_id} but no EmailMetadata row "
                f"to record dedup — may retrigger on next reprocess"
            )

        logger.info(
            f"✉️  Notification sent for email {email_id}: "
            f"{notify_reason} (message_id={message_id})"
        )
        return message_id
    finally:
        try:
            db.close()
        except Exception:
            pass
