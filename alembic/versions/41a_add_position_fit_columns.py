"""Add position_fit_score and position_fit_reason to email_metadata

Revision ID: 41a1b2c3d4e5
Revises: 40a1b2c3d4e5
Create Date: 2026-04-17

Adds two columns to support per-job-posting application scoring:
- position_fit_score (int 1-10): score against the specific job posting,
  separate from research_fit_score (which scores against the broad lab portfolio)
- position_fit_reason (encrypted text): justification for the score

Populated only when the source email carries an X-Position-URL header
(e.g., applications imported from jobs.ethz.ch via the application-import pipeline).
"""
from typing import Sequence, Union

from alembic import op

revision: str = '41a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = '40a1b2c3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE email_metadata
        ADD COLUMN IF NOT EXISTS position_fit_score integer
    """)
    op.execute("""
        ALTER TABLE email_metadata
        ADD COLUMN IF NOT EXISTS position_fit_reason text
    """)


def downgrade() -> None:
    op.drop_column('email_metadata', 'position_fit_reason')
    op.drop_column('email_metadata', 'position_fit_score')
