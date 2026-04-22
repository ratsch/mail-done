"""Add notify_sent_at / notify_message_id columns to email_metadata

Revision ID: 42a1b2c3d4e5
Revises: 41a1b2c3d4e5
Create Date: 2026-04-22

Previously the notification-emitter tracked which emails had already been
pushed to Slack via a ``notify_sent_at`` key inside
``email_metadata.category_metadata`` (a PostgreSQL JSON column). That
worked but made cooldown and per-sender dedup queries expensive —
filtering by timestamp required a cast-to-text LIKE followed by an
in-Python date comparison.

Promote the dedup marker to a proper ``timestamp`` column so cooldown
windows and analytics are cheap indexed lookups. The JSON marker is left
in place for historical rows; ``send_notification`` now writes to both
during a transition period.
"""
from typing import Sequence, Union

from alembic import op

revision: str = '42a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = '41a1b2c3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE email_metadata
        ADD COLUMN IF NOT EXISTS notify_sent_at timestamp
    """)
    op.execute("""
        ALTER TABLE email_metadata
        ADD COLUMN IF NOT EXISTS notify_message_id varchar(255)
    """)
    # Partial index — most rows will never be notified; only non-null rows
    # are interesting for cooldown and analytics.
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_email_metadata_notify_sent_at
        ON email_metadata (notify_sent_at)
        WHERE notify_sent_at IS NOT NULL
    """)
    # Backfill from the JSON marker so historical rows are visible in the
    # new column too. Values produced by send_notification before this
    # migration land are ISO-8601 strings with second precision.
    # category_metadata is a `json` column (not jsonb), so the ? operator
    # isn't available — use ->> which works on json and returns text/NULL.
    op.execute("""
        UPDATE email_metadata
        SET notify_sent_at = (category_metadata->>'notify_sent_at')::timestamp,
            notify_message_id = category_metadata->>'notify_message_id'
        WHERE notify_sent_at IS NULL
          AND category_metadata->>'notify_sent_at' IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_email_metadata_notify_sent_at")
    op.drop_column('email_metadata', 'notify_message_id')
    op.drop_column('email_metadata', 'notify_sent_at')
