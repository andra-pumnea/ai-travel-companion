"""drop and recreate user_facts

Revision ID: 0e533a688f39
Revises: "1a3d0600970d"
Create Date: 2025-07-09 14:30:00.173263

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0e533a688f39"
down_revision: Union[str, Sequence[str], None] = "1a3d0600970d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_table("user_facts")
    op.create_table(
        "user_facts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("fact", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("user_facts")
