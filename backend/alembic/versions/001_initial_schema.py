"""initial schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-09-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('active_barcode', sa.String(length=32), nullable=True),
        sa.Column('active_comparison_barcodes', sa.JSON(), nullable=True),
        sa.Column('ocr_combined_text', sa.Text(), nullable=True),
        sa.Column('ocr_nutrition_text', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_table(
        'messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sequence_number', sa.Integer(), nullable=False),
        sa.CheckConstraint("role IN ('user', 'assistant', 'system')", name='ck_messages_role'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_messages_conversation_sequence', 'messages', ['conversation_id', 'sequence_number'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_messages_conversation_sequence', table_name='messages')
    op.drop_table('messages')
    op.drop_table('conversations')
