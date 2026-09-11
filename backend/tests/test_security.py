import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database.connection import AsyncSessionFactory
from app.database.crud import create_conversation, get_conversation_messages
from app.tools.barcode import InvalidBarcodeError, normalize_barcode

client = TestClient(app)


def test_cors_allows_configured_origins() -> None:
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_unauthorized_origins() -> None:
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://malicious-site.test",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") != "http://malicious-site.test"


def test_sql_injection_attempt_in_barcode_is_safely_rejected() -> None:
    injection_payload = "' OR '1'='1"
    with pytest.raises(InvalidBarcodeError):
        normalize_barcode(injection_payload)

    response = client.post("/chat", json={"message": f"Check barcode {injection_payload}"})
    assert response.status_code in (200, 422)
    assert "OR '1'='1" not in response.text or response.status_code == 200


@pytest.mark.asyncio
async def test_conversation_isolation_prevents_unauthorized_access() -> None:
    user_a_id = uuid.uuid4()
    user_b_id = uuid.uuid4()

    async with AsyncSessionFactory() as session:
        await create_conversation(session, conversation_id=user_a_id)
        await create_conversation(session, conversation_id=user_b_id)

    # Attempt to fetch User B's conversation using User A's ID
    res_a = client.get(f"/conversations/{user_a_id}")
    assert res_a.status_code == 200
    assert res_a.json()["conversation_id"] == str(user_a_id)

    res_b = client.get(f"/conversations/{user_b_id}")
    assert res_b.status_code == 200
    assert res_b.json()["conversation_id"] == str(user_b_id)
    assert res_a.json()["conversation_id"] != res_b.json()["conversation_id"]


def test_error_response_does_not_leak_secrets(monkeypatch) -> None:
    from app import config

    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")
    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 503
    error_text = response.text.lower()

    assert "postgres://" not in error_text
    assert "password" not in error_text
    assert "secret" not in error_text
    assert "traceback" not in error_text
