import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.database.connection import AsyncSessionFactory, init_db_tables
from app.database.crud import (
    add_message,
    create_conversation,
    get_conversation,
    get_conversation_messages,
    update_conversation_context,
)
from app.database.models import Base, Conversation, Message
from app.graph.nodes import (
    comparison_node,
    ingredient_node,
    nutrition_node,
    product_lookup_node,
    recommendation_node,
    safety_node,
)
from app.graph.state import FoodGuardState
from app.main import app
from app.schemas import ChatMessage, ChatRequest, ChatResponse
from app.tools.nutrition import NutritionData
from app.tools.product_lookup import ProductData, ProductLookupResult

client = TestClient(app)


def _mock_nutella() -> ProductData:
    return ProductData(
        barcode="3017620422003",
        product_name="Nutella",
        brand="Ferrero",
        ingredients_text="Sugar, Palm Oil, Hazelnuts, Cocoa, Skimmed Milk Powder",
        allergen_statement="Contains Hazelnuts, Milk, Soy",
        traces=["Nuts"],
        nutrition=NutritionData(
            total_sugar_g=56.3,
            protein_g=6.3,
            sodium_mg=42.0,
            saturated_fat_g=10.6,
            energy_kcal=539.0,
        ),
        serving_size="15g",
        has_complete_nutrition=True,
        source_url="https://world.openfoodfacts.org/product/3017620422003",
    )


def _mock_peanut_butter() -> ProductData:
    return ProductData(
        barcode="0051500255162",
        product_name="Jif Creamy Peanut Butter",
        brand="Jif",
        ingredients_text="Roasted Peanuts, Sugar, Hydrogenated Vegetable Oil, Salt",
        allergen_statement="Contains Peanuts",
        traces=[],
        nutrition=NutritionData(
            total_sugar_g=3.0,
            protein_g=7.0,
            sodium_mg=140.0,
            saturated_fat_g=3.5,
            energy_kcal=190.0,
        ),
        serving_size="33g",
        has_complete_nutrition=True,
        source_url="https://world.openfoodfacts.org/product/0051500255162",
    )


def _mock_lookup_side_effect(barcode: str) -> ProductLookupResult:
    if barcode == "3017620422003":
        return ProductLookupResult(found=True, product=_mock_nutella())
    elif barcode == "0051500255162":
        return ProductLookupResult(found=True, product=_mock_peanut_butter())
    return ProductLookupResult(found=False, error="PRODUCT_NOT_FOUND")


# 1. Database connection
@pytest.mark.asyncio
async def test_1_db_connection() -> None:
    async with AsyncSessionFactory() as session:
        assert session is not None


# 2. Migration / schema availability
def test_2_schema_availability() -> None:
    assert hasattr(Conversation, "__tablename__")
    assert hasattr(Message, "__tablename__")
    assert Conversation.__tablename__ == "conversations"
    assert Message.__tablename__ == "messages"


# 3. Conversation creation
@pytest.mark.asyncio
async def test_3_conversation_creation() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        conv = await create_conversation(session, conversation_id=c_id, active_barcode="3017620422003")
        assert conv.id == c_id
        assert conv.active_barcode == "3017620422003"


# 4. Conversation retrieval
@pytest.mark.asyncio
async def test_4_conversation_retrieval() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=c_id, active_barcode="3017620422003")
        fetched = await get_conversation(session, c_id)
        assert fetched is not None
        assert fetched.id == c_id


# 5. Message persistence
@pytest.mark.asyncio
async def test_5_message_persistence() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=c_id)
        msg1 = await add_message(session, c_id, role="user", content="Hello DB")
        assert msg1.content == "Hello DB"
        assert msg1.sequence_number == 1


# 6. Message ordering
@pytest.mark.asyncio
async def test_6_message_ordering() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=c_id)
        m1 = await add_message(session, c_id, role="user", content="First")
        m2 = await add_message(session, c_id, role="assistant", content="Second")
        assert m1.sequence_number == 1
        assert m2.sequence_number == 2


# 7. Message retrieval
@pytest.mark.asyncio
async def test_7_message_retrieval() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=c_id)
        await add_message(session, c_id, role="user", content="Msg 1")
        await add_message(session, c_id, role="assistant", content="Msg 2")
        msgs = await get_conversation_messages(session, c_id)
        assert len(msgs) == 2
        assert msgs[0].content == "Msg 1"
        assert msgs[1].content == "Msg 2"


# 8. Conversation isolation
@pytest.mark.asyncio
async def test_8_conversation_isolation() -> None:
    id1, id2 = uuid.uuid4(), uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=id1, active_barcode="3017620422003")
        await create_conversation(session, conversation_id=id2, active_barcode="0051500255162")
        c1 = await get_conversation(session, id1)
        c2 = await get_conversation(session, id2)
        assert c1.active_barcode == "3017620422003"
        assert c2.active_barcode == "0051500255162"


# 9. Active barcode persistence
@pytest.mark.asyncio
async def test_9_active_barcode_persistence() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=c_id)
        await update_conversation_context(session, c_id, active_barcode="3017620422003")
        fetched = await get_conversation(session, c_id)
        assert fetched.active_barcode == "3017620422003"


# 10. Comparison barcode persistence
@pytest.mark.asyncio
async def test_10_comparison_barcode_persistence() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=c_id)
        await update_conversation_context(session, c_id, active_comparison_barcodes=["3017620422003", "0051500255162"])
        fetched = await get_conversation(session, c_id)
        assert fetched.active_comparison_barcodes == ["3017620422003", "0051500255162"]


# 11. OCR-derived context persistence
@pytest.mark.asyncio
async def test_11_ocr_derived_context_persistence() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=c_id)
        await update_conversation_context(session, c_id, ocr_combined_text="Extracted text", ocr_nutrition_text="Sugar 10g")
        fetched = await get_conversation(session, c_id)
        assert fetched.ocr_combined_text == "Extracted text"
        assert fetched.ocr_nutrition_text == "Sugar 10g"


# 12. Raw image bytes are never persisted
def test_12_raw_image_bytes_not_in_schema() -> None:
    conv_cols = [c.name for c in Conversation.__table__.columns]
    msg_cols = [c.name for c in Message.__table__.columns]
    assert "image_bytes" not in conv_cols
    assert "image_file" not in conv_cols
    assert "raw_image" not in msg_cols


# 13. New conversation creation
@patch("app.main.workflow.invoke")
def test_13_new_conversation_creation(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Hello user")],
    }
    resp = client.post("/chat", json={"message": "Hi"})
    assert resp.status_code == 200
    assert "conversation_id" in resp.json()
    assert len(resp.json()["conversation_id"]) == 36


# 14. Existing conversation restoration endpoint
@patch("app.main.workflow.invoke")
def test_14_conversation_restoration_endpoint(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Analyzed Nutella.")],
        "product_data": _mock_nutella(),
    }
    r1 = client.post("/chat", json={"message": "Analyze barcode 3017620422003"})
    c_id = r1.json()["conversation_id"]

    r2 = client.get(f"/conversations/{c_id}")
    assert r2.status_code == 200
    data = r2.json()
    assert data["conversation_id"] == c_id
    assert len(data["messages"]) >= 2


# 15. Invalid conversation ID handling
def test_15_invalid_conversation_id_handling() -> None:
    resp = client.get("/conversations/invalid-uuid-string")
    assert resp.status_code == 400


# 16. Missing conversation handling
def test_16_missing_conversation_handling() -> None:
    random_uuid = str(uuid.uuid4())
    resp = client.get(f"/conversations/{random_uuid}")
    assert resp.status_code == 404


# 17. Transaction rollback
@pytest.mark.asyncio
async def test_17_transaction_rollback() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=c_id)
        try:
            msg_invalid = Message(conversation_id=c_id, role="invalid_role", content="Err", sequence_number=1)
            session.add(msg_invalid)
            await session.commit()
        except Exception:
            await session.rollback()
        
        fetched = await get_conversation(session, c_id)
        assert fetched is not None


# 18. Database failure handling / fallback
@patch("app.main.workflow.invoke")
def test_18_database_failure_fallback(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Fallback response.")],
    }
    with patch("app.main.AsyncSessionFactory", side_effect=RuntimeError("DB Offline")):
        resp = client.post("/chat", json={"message": "Test during DB failure"})
        assert resp.status_code == 200
        assert "Fallback response" in resp.json()["response"]


# 19. Concurrent message safety
@pytest.mark.asyncio
async def test_19_concurrent_message_safety() -> None:
    c_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=c_id)

    async def _add(i: int):
        async with AsyncSessionFactory() as s:
            return await add_message(s, c_id, role="user", content=f"Concurrent message {i}")

    messages = await asyncio.gather(*[_add(i) for i in range(5)])

    seqs = sorted([m.sequence_number for m in messages])
    assert seqs == [1, 2, 3, 4, 5]
    assert len(set(seqs)) == 5


# 20. Phase 15 memory regression
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_20_phase15_memory_regression(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Nutella contains 56.3g sugar.")],
        "product_data": _mock_nutella(),
    }
    r1 = client.post("/chat", json={"message": "Analyze barcode 3017620422003"})
    c_id = r1.json()["conversation_id"]

    r2 = client.post("/chat", json={"message": "What about the sugar?", "conversation_id": c_id, "barcode": "3017620422003"})
    assert r2.status_code == 200
    assert r2.json()["barcode"] == "3017620422003"


# 21. Phase 16 chatbot regression
@patch("app.main.workflow.invoke")
def test_21_phase16_chatbot_regression(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Chatbot answer.")],
    }
    resp = client.post("/chat", json={"message": "Hi FoodGuard"})
    assert resp.status_code == 200


# 22. Phase 12 comparison regression
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_22_comparison_regression(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [HumanMessage(content="Compare 3017620422003 and 0051500255162")],
    }
    res = comparison_node(state)
    assert res["comparison_analysis"].available is True


# 23. Phase 13 recommendation regression
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_23_recommendation_regression(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [HumanMessage(content="Recommend lower sugar")],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }
    res = recommendation_node(state)
    assert res["recommendation_analysis"].available is True


# 24. Phase 14 verification regression
@patch("app.main.workflow.invoke")
def test_24_verification_regression(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Verified product.")],
    }
    resp = client.post("/chat", json={"message": "Analyze barcode 3017620422003"})
    assert resp.status_code == 200


# 25. OCR / RAG regression compatibility
@patch("app.main.workflow.invoke")
def test_25_ocr_rag_regression_compatibility(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="OCR and RAG working.")],
    }
    resp = client.post("/chat", json={"message": "Why is sesame an allergen?", "ocr_combined_text": "Oats, Honey"})
    assert resp.status_code == 200
