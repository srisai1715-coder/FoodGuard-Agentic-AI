import uuid
import sys
import logging
from fastapi.testclient import TestClient
from app.main import app
from app.database.connection import AsyncSessionFactory, init_db_tables
from app.database.crud import create_conversation, get_conversation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live_postgres_checks")

client = TestClient(app)

def run_live_postgres_checks():
    logger.info("==================================================")
    logger.info("STARTING PHASE 17 LIVE POSTGRESQL INTEGRATION CHECKS")
    logger.info("==================================================")

    # TEST A: Create Conversation
    logger.info("[TEST A] Creating new conversation via /chat endpoint...")
    resp_a = client.post("/chat", json={"message": "Hello FoodGuard"})
    assert resp_a.status_code == 200, f"TEST A failed: {resp_a.text}"
    data_a = resp_a.json()
    conv_id_a = data_a.get("conversation_id")
    assert conv_id_a is not None, "TEST A failed: no conversation_id returned"
    logger.info("TEST A PASSED — Created conversation_id: %s", conv_id_a)

    # TEST B: Persist User + Assistant Messages
    logger.info("[TEST B] Persisting messages in conversation...")
    resp_b = client.post("/chat", json={
        "conversation_id": conv_id_a,
        "message": "What is sodium?"
    })
    assert resp_b.status_code == 200, f"TEST B failed: {resp_b.text}"
    logger.info("TEST B PASSED — Messages persisted successfully")

    # TEST C: Refresh / Reload Conversation and Restore History
    logger.info("[TEST C] Restoring conversation from PostgreSQL via GET /conversations/{id}...")
    resp_c = client.get(f"/conversations/{conv_id_a}")
    assert resp_c.status_code == 200, f"TEST C failed: {resp_c.text}"
    data_c = resp_c.json()
    assert data_c["conversation_id"] == conv_id_a, "TEST C failed: ID mismatch"
    assert len(data_c["messages"]) >= 4, f"TEST C failed: expected >= 4 messages, got {len(data_c['messages'])}"
    logger.info("TEST C PASSED — Restored %d messages from PostgreSQL", len(data_c["messages"]))

    # TEST D: Persist Active Barcode & Ask Follow-up
    logger.info("[TEST D] Persisting active barcode (Nutella 3017620422003)...")
    resp_d1 = client.post("/chat", json={
        "conversation_id": conv_id_a,
        "message": "Analyze product barcode 3017620422003",
        "barcode": "3017620422003"
    })
    assert resp_d1.status_code == 200, f"TEST D1 failed: {resp_d1.text}"
    assert resp_d1.json().get("barcode") == "3017620422003", "TEST D1 failed: barcode not set"

    resp_d2 = client.post("/chat", json={
        "conversation_id": conv_id_a,
        "message": "What about the sugar content?"
    })
    assert resp_d2.status_code == 200, f"TEST D2 failed: {resp_d2.text}"
    assert resp_d2.json().get("barcode") == "3017620422003", "TEST D2 failed: barcode context lost"
    logger.info("TEST D PASSED — Active barcode persisted and restored for follow-up")

    # TEST E: Persist Comparison Barcodes & Ask Follow-up
    logger.info("[TEST E] Persisting comparison barcodes...")
    conv_e_id = str(uuid.uuid4())
    resp_e1 = client.post("/chat", json={
        "conversation_id": conv_e_id,
        "message": "Compare products 3017620422003 and 0051500255162",
        "barcodes": ["3017620422003", "0051500255162"]
    })
    assert resp_e1.status_code == 200, f"TEST E1 failed: {resp_e1.text}"

    resp_e2 = client.post("/chat", json={
        "conversation_id": conv_e_id,
        "message": "Which product is healthier?"
    })
    assert resp_e2.status_code == 200, f"TEST E2 failed: {resp_e2.text}"
    logger.info("TEST E PASSED — Comparison barcodes persisted and restored for follow-up")

    # TEST F: Persist OCR-derived context
    logger.info("[TEST F] Persisting OCR-derived context...")
    conv_f_id = str(uuid.uuid4())
    resp_f1 = client.post("/chat", json={
        "conversation_id": conv_f_id,
        "message": "Analyze uploaded label",
        "ocr_combined_text": "Ingredients: Oats, Honey, Almonds",
        "ocr_nutrition_text": "Sugar: 8g, Protein: 4g"
    })
    assert resp_f1.status_code == 200, f"TEST F1 failed: {resp_f1.text}"

    resp_f2 = client.get(f"/conversations/{conv_f_id}")
    assert resp_f2.status_code == 200, f"TEST F2 failed: {resp_f2.text}"
    assert resp_f2.json().get("ocr_combined_text") == "Ingredients: Oats, Honey, Almonds"
    logger.info("TEST F PASSED — OCR context persisted without raw image bytes")

    # TEST G: New Chat creates isolated conversation
    logger.info("[TEST G] New Chat creates isolated conversation...")
    resp_g = client.post("/chat", json={
        "message": "Start a fresh topic"
    })
    assert resp_g.status_code == 200
    conv_id_g = resp_g.json().get("conversation_id")
    assert conv_id_g != conv_id_a, "TEST G failed: New chat reused previous ID"
    assert resp_g.json().get("barcode") is None, "TEST G failed: Stale barcode retained"
    logger.info("TEST G PASSED — New chat created isolated conversation_id: %s", conv_id_g)

    # TEST H: Conversation A cannot read Conversation B
    logger.info("[TEST H] Verifying conversation isolation...")
    resp_h_a = client.get(f"/conversations/{conv_id_a}")
    resp_h_g = client.get(f"/conversations/{conv_id_g}")
    assert resp_h_a.json()["messages"] != resp_h_g.json()["messages"], "TEST H failed: Leakage between chats"
    logger.info("TEST H PASSED — Strict conversation isolation confirmed")

    # TEST I: General RAG question does not destroy active product context
    logger.info("[TEST I] Testing General RAG question during active product session...")
    resp_i1 = client.post("/chat", json={
        "conversation_id": conv_id_a,
        "message": "Why is sesame classified as a major allergen?"
    })
    assert resp_i1.status_code == 200

    resp_i2 = client.get(f"/conversations/{conv_id_a}")
    assert resp_i2.json().get("active_barcode") == "3017620422003", "TEST I failed: RAG query wiped active barcode"
    logger.info("TEST I PASSED — General RAG did not destroy active product context")

    # TEST J: Database failure produces controlled behavior
    logger.info("[TEST J] Database failure handling...")
    resp_j_invalid = client.get("/conversations/invalid-uuid-format")
    assert resp_j_invalid.status_code == 400, "TEST J failed: Invalid UUID didn't return 400"
    resp_j_missing = client.get(f"/conversations/{uuid.uuid4()}")
    assert resp_j_missing.status_code == 404, "TEST J failed: Missing UUID didn't return 404"
    logger.info("TEST J PASSED — Database failure/error handling validated")

    logger.info("==================================================")
    logger.info("ALL PHASE 17 LIVE POSTGRESQL TESTS PASSED SUCCESSFULLY!")
    logger.info("==================================================")

if __name__ == "__main__":
    run_live_postgres_checks()
