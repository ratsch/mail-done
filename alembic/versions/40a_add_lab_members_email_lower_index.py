"""Add functional index on LOWER(lab_members.email)

Revision ID: 40a1b2c3d4e5
Revises: 39a1b2c3d4e5
Create Date: 2026-04-13

The auth code matches LabMember by case-insensitive email lookup
(func.lower(LabMember.email) == email.lower()). Without a functional
index on LOWER(email), Postgres falls back to a sequential scan even
though we have a normal B-tree on `email`. This migration adds a
matching functional index so case-insensitive lookups stay O(log n).

Companion change: a `@validates('email')` hook on the LabMember model
normalizes incoming emails to lowercase so all rows are invariant
lowercase. The functional index then collapses to the same set of keys
as the normal index, but stays correct under any historical mixed-case
data and any external SQL that bypasses the ORM.
"""
from typing import Sequence, Union
from alembic import op

revision: str = '40a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = '39a1b2c3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_lab_members_email_lower "
        "ON lab_members (LOWER(email))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_lab_members_email_lower")
