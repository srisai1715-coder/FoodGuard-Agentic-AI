import io
import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.database.connection import AsyncSessionFactory
from app.database.crud import add_message, create_conversation, get_conversation_messages
from app.tools.product_lookup import ProductLookupError, ProductLookupResult, lookup_product_by_barcode
from app.tools.ocr import process_images, ImageValidationError, OCRStatus
from app.rag.retriever import Retriever

client = TestClient(app)


def test_corrupted_image_binary_handled_gracefully() -> None:
    corrupted_data = b"NOT_AN_IMAGE_HEADER_12345"
    with pytest.raises(ImageValidationError) as exc_info:
        process_images([(corrupted_data, "bad.jpg", "image/jpeg")])
    assert exc_info.value.status_code == 400


def test_zero_byte_image_rejected() -> None:
    empty_data = b""
    with pytest.raises(ImageValidationError) as exc_info:
        process_images([(empty_data, "empty.jpg", "image/jpeg")])
    assert exc_info.value.status_code == 400


def test_openfoodfacts_http_500_error_handling() -> None:
    with patch("httpx.Client.get") as mock_get:
        mock_get.return_value.status_code = 500
        result = lookup_product_by_barcode("3017620422003")
        assert result.found is False
        assert result.error == ProductLookupError.UPSTREAM_UNAVAILABLE


def test_openfoodfacts_timeout_error_handling() -> None:
    import httpx
    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Connection timed out")):
        result = lookup_product_by_barcode("3017620422003")
        assert result.found is False
        assert result.error == ProductLookupError.UPSTREAM_TIMEOUT


def test_rag_empty_query_returns_empty_context() -> None:
    retriever = Retriever()
    res = retriever.retrieve("")
    assert res == []


@pytest.mark.asyncio
async def test_long_20_turn_conversation_sequence_integrity() -> None:
    conv_id = uuid.uuid4()
    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=conv_id)

        for i in range(1, 21):
            role = "user" if i % 2 != 0 else "assistant"
            msg = await add_message(session, conversation_id=conv_id, role=role, content=f"Turn message {i}")
            assert msg.sequence_number == i

        messages = await get_conversation_messages(session, conversation_id=conv_id)
        assert len(messages) == 20
        sequence_numbers = [m.sequence_number for m in messages]
        assert sequence_numbers == list(range(1, 21))
        assert len(set(sequence_numbers)) == 20
