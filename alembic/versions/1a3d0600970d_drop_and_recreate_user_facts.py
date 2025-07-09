"""drop and recreate user_facts

Revision ID: 1a3d0600970d
Revises: c7b1a638b67f
Create Date: 2025-07-09 14:11:56.927229

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1a3d0600970d"
down_revision: Union[str, Sequence[str], None] = "c7b1a638b67f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
