import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Conversation, Message


async def get_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> Conversation | None:
    stmt = select(Conversation).where(Conversation.id == conversation_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def create_conversation(
    session: AsyncSession,
    conversation_id: uuid.UUID | None = None,
    active_barcode: str | None = None,
    active_comparison_barcodes: list[str] | None = None,
    ocr_combined_text: str | None = None,
    ocr_nutrition_text: str | None = None,
) -> Conversation:
    conversation = Conversation(
        id=conversation_id or uuid.uuid4(),
        active_barcode=active_barcode,
        active_comparison_barcodes=active_comparison_barcodes,
        ocr_combined_text=ocr_combined_text,
        ocr_nutrition_text=ocr_nutrition_text,
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def update_conversation_context(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    active_barcode: str | None = None,
    active_comparison_barcodes: list[str] | None = None,
    ocr_combined_text: str | None = None,
    ocr_nutrition_text: str | None = None,
) -> Conversation | None:
    conversation = await get_conversation(session, conversation_id)
    if not conversation:
        return None

    if active_barcode is not None:
        conversation.active_barcode = active_barcode
    if active_comparison_barcodes is not None:
        conversation.active_comparison_barcodes = active_comparison_barcodes
    if ocr_combined_text is not None:
        conversation.ocr_combined_text = ocr_combined_text
    if ocr_nutrition_text is not None:
        conversation.ocr_nutrition_text = ocr_nutrition_text

    conversation.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def add_message(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
) -> Message:
    # Lock the parent Conversation row using SELECT FOR UPDATE to serialize sequence generation per conversation
    lock_stmt = select(Conversation).where(Conversation.id == conversation_id).with_for_update()
    await session.execute(lock_stmt)

    # Concurrency-safe sequence calculation while parent row lock is held
    stmt = select(func.coalesce(func.max(Message.sequence_number), 0)).where(
        Message.conversation_id == conversation_id
    )
    res = await session.execute(stmt)
    current_max = res.scalar_one()
    next_sequence = current_max + 1

    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sequence_number=next_sequence,
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return msg


async def get_conversation_messages(
    session: AsyncSession,
    conversation_id: uuid.UUID,
) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.sequence_number.asc())
    )
    res = await session.execute(stmt)
    return list(res.scalars().all())
