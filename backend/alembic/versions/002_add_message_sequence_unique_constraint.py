"""add message sequence unique constraint

Revision ID: 002_msg_seq_unique
Revises: 001_initial_schema
Create Date: 2026-09-09

"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '002_msg_seq_unique'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        'uq_messages_conversation_sequence',
        'messages',
        ['conversation_id', 'sequence_number']
    )


def downgrade() -> None:
    op.drop_constraint(
        'uq_messages_conversation_sequence',
        'messages',
        type_='unique'
    )
