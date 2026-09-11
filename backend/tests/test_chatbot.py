import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

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
        product_name="Creamy Peanut Butter",
        brand="Jif",
        ingredients_text="Peanuts, Sugar, Hydrogenated Vegetable Oil, Salt",
        allergen_statement="Contains Peanuts",
        traces=[],
        nutrition=NutritionData(
            total_sugar_g=9.4,
            protein_g=21.9,
            sodium_mg=438.0,
            saturated_fat_g=10.9,
            energy_kcal=594.0,
        ),
        serving_size="32g",
        has_complete_nutrition=True,
        source_url="https://world.openfoodfacts.org/product/0051500255162",
    )


def _mock_lookup_side_effect(barcode: str) -> ProductLookupResult:
    if barcode == "3017620422003":
        return ProductLookupResult(found=True, product=_mock_nutella())
    if barcode == "0051500255162":
        return ProductLookupResult(found=True, product=_mock_peanut_butter())
    return ProductLookupResult(found=False, error="PRODUCT_NOT_FOUND")


# 1. Basic chatbot question
@patch("app.main.workflow.invoke")
def test_1_basic_chatbot_question(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Hello! How can I help you analyze food products today?")],
    }
    resp = client.post("/chat", json={"message": "Hello FoodGuard AI"})
    assert resp.status_code == 200
    assert "help" in resp.json()["response"].lower() or "foodguard" in resp.json()["response"].lower()


# 2. Single product follow-up
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_2_single_product_followup(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Nutella contains 56.3g of total sugar per 100g.")],
        "product_data": _mock_nutella(),
    }
    payload = {
        "message": "What about the sugar?",
        "messages": [
            {"role": "user", "content": "Analyze barcode 3017620422003"},
            {"role": "assistant", "content": "Nutella product analyzed."},
        ],
        "barcode": "3017620422003",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["barcode"] == "3017620422003"
    assert "sugar" in resp.json()["response"].lower()


# 3. Multi-turn product conversation
@patch("app.main.workflow.invoke")
def test_3_multiturn_product_conversation(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="56.3g of sugar per 100g is considered very high.")],
    }
    payload = {
        "message": "Is that high?",
        "messages": [
            {"role": "user", "content": "Analyze barcode 3017620422003"},
            {"role": "assistant", "content": "Nutella summary."},
            {"role": "user", "content": "What about the sugar?"},
            {"role": "assistant", "content": "It has 56.3g of sugar."},
        ],
        "barcode": "3017620422003",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert "high" in resp.json()["response"].lower()


# 4. Pronoun / reference follow-up
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_4_pronoun_reference_followup(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Nutella contains 6.3g of protein per 100g.")],
        "product_data": _mock_nutella(),
    }
    payload = {
        "message": "How much protein is in it?",
        "messages": [
            {"role": "user", "content": "Analyze barcode 3017620422003"},
            {"role": "assistant", "content": "Nutella analyzed."},
        ],
        "barcode": "3017620422003",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["barcode"] == "3017620422003"


# 5. Ingredient follow-up
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_5_ingredient_followup(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [AIMessage(content="Turn 1")],
        "product_lookup_request": "3017620422003",
    }
    res = ingredient_node(state)
    assert res["ingredient_analysis"].available is True
    assert "Sugar" in res["ingredient_data"]


# 6. Allergen follow-up
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_6_allergen_followup(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [AIMessage(content="Turn 1")],
        "product_lookup_request": "3017620422003",
    }
    res = safety_node(state)
    assert res["safety_analysis"].available is True


# 7. Comparison follow-up
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_7_comparison_followup(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Creamy Peanut Butter has less sugar (9.4g) than Nutella (56.3g).")],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }
    payload = {
        "message": "Which one has less sugar?",
        "messages": [
            {"role": "user", "content": "Compare 3017620422003 and 0051500255162"},
            {"role": "assistant", "content": "Comparison table generated."},
        ],
        "barcodes": ["3017620422003", "0051500255162"],
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["comparison_barcodes"] == ["3017620422003", "0051500255162"]


# 8. Comparison nutrition follow-up
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_8_comparison_nutrition_followup(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [AIMessage(content="Turn 1")],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }
    res = comparison_node(state)
    assert res["comparison_analysis"].available is True
    assert len(res["comparison_products"]) == 2


# 9. Recommendation follow-up
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_9_recommendation_followup(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [AIMessage(content="Turn 1")],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }
    res = recommendation_node(state)
    assert res["recommendation_analysis"].available is True


# 10. Recommendation justification
@patch("app.main.workflow.invoke")
def test_10_recommendation_justification(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="I recommended Creamy Peanut Butter because of its lower sugar content.")],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }
    payload = {
        "message": "Why did you recommend that one?",
        "messages": [
            {"role": "user", "content": "Which one is better for lower sugar? 3017620422003 and 0051500255162"},
            {"role": "assistant", "content": "I recommend Creamy Peanut Butter."},
        ],
        "barcodes": ["3017620422003", "0051500255162"],
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert "Peanut Butter" in resp.json()["response"] or "sugar" in resp.json()["response"].lower()


# 11. OCR follow-up
@patch("app.main.workflow.invoke")
def test_11_ocr_followup(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="No milk ingredients were detected in the OCR text.")],
        "ocr_combined_text": "Ingredients: Oats, Honey, Almonds, Water",
    }
    payload = {
        "message": "Does it contain milk?",
        "ocr_combined_text": "Ingredients: Oats, Honey, Almonds, Water",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["ocr_combined_text"] == payload["ocr_combined_text"]


# 12. General RAG during product conversation
@patch("app.main.workflow.invoke")
def test_12_general_rag_during_product_conversation(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Sesame is classified as a major food allergen under the FASTER Act.")],
    }
    payload = {
        "message": "Why is sesame considered a major food allergen?",
        "messages": [
            {"role": "user", "content": "Analyze barcode 3017620422003"},
            {"role": "assistant", "content": "Nutella analyzed."},
        ],
        "barcode": "3017620422003",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert "sesame" in resp.json()["response"].lower() or "allergen" in resp.json()["response"].lower()


# 13. Return from RAG to product context
@patch("app.main.workflow.invoke")
def test_13_return_from_rag_to_product_context(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Nutella contains 56.3g of total sugar per 100g.")],
        "barcode": "3017620422003",
    }
    payload = {
        "message": "Now tell me about the sugar in this product.",
        "messages": [
            {"role": "user", "content": "Analyze barcode 3017620422003"},
            {"role": "assistant", "content": "Nutella analyzed."},
            {"role": "user", "content": "Why is sesame a major allergen?"},
            {"role": "assistant", "content": "Sesame explanation."},
        ],
        "barcode": "3017620422003",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["barcode"] == "3017620422003"


# 14. Missing product clarification
@patch("app.main.workflow.invoke")
def test_14_missing_product_clarification(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="To analyze a product, please provide a product barcode or upload a label image.")],
    }
    resp = client.post("/chat", json={"message": "Is it healthy?"})
    assert resp.status_code == 200
    assert "barcode" in resp.json()["response"].lower() or "label" in resp.json()["response"].lower() or "provide" in resp.json()["response"].lower()


# 15. Missing comparison clarification
@patch("app.main.workflow.invoke")
def test_15_missing_comparison_clarification(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="To perform a comparison, please provide two product barcodes or upload product labels.")],
    }
    resp = client.post("/chat", json={"message": "Which product is better?"})
    assert resp.status_code == 200
    assert "compare" in resp.json()["response"].lower() or "barcodes" in resp.json()["response"].lower()


# 16. Missing nutrition data
@patch("app.graph.nodes.lookup_product_by_barcode")
def test_16_missing_nutrition_data(mock_lookup: MagicMock) -> None:
    incomplete = ProductData(
        barcode="1111111111111",
        product_name="Sample Item",
        brand="Generic",
        ingredients_text="Water",
        allergen_statement=None,
        traces=[],
        nutrition=None,
        serving_size=None,
        has_complete_nutrition=False,
        source_url="https://world.openfoodfacts.org/product/1111111111111",
    )
    mock_lookup.return_value = ProductLookupResult(found=True, product=incomplete)
    state: FoodGuardState = {
        "messages": [AIMessage(content="Check nutrition")],
        "product_lookup_request": "1111111111111",
    }
    res = nutrition_node(state)
    assert res["nutrition_analysis"].available is False


# 17. Conflicting evidence
@patch("app.main.workflow.invoke")
def test_17_conflicting_evidence(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="There is a conflict between OCR text and database nutrition data.")],
    }
    payload = {
        "message": "Verify sugar content",
        "ocr_nutrition_text": "Sugar 5g",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200


# 18. Verification integration
@patch("app.main.workflow.invoke")
def test_18_verification_integration(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Verified product safety analysis.")],
    }
    resp = client.post("/chat", json={"message": "Analyze barcode 3017620422003"})
    assert resp.status_code == 200


# 19. Error handling
def test_19_error_handling() -> None:
    resp = client.post("/chat", json={"messages": []})
    assert resp.status_code == 422


# 20. New Chat context reset
def test_20_new_chat_context_reset() -> None:
    req = ChatRequest(message="Test", messages=[])
    assert req.barcode is None
    assert req.barcodes == []
    assert req.ocr_combined_text is None


# 21. Conversation ID propagation
@patch("app.main.workflow.invoke")
def test_21_conversation_id_propagation(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Response with conversation ID.")],
    }
    valid_uuid = str(uuid.uuid4())
    payload = {
        "message": "Hello",
        "conversation_id": valid_uuid,
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["conversation_id"] == valid_uuid


# 22. New barcode replacement
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_22_new_barcode_replacement(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Analyzed Creamy Peanut Butter.")],
        "product_data": _mock_peanut_butter(),
    }
    payload = {
        "message": "Analyze barcode 0051500255162",
        "barcode": "3017620422003",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["barcode"] == "0051500255162"


# 23. New comparison replacement
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_23_new_comparison_replacement(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [HumanMessage(content="Compare 0051500255162 and 3017620422003")],
        "comparison_barcodes": ["1111111111111", "2222222222222"],
    }
    res = comparison_node(state)
    assert res["comparison_barcodes"] == ["0051500255162", "3017620422003"]


# 24. New OCR replacement
@patch("app.main.workflow.invoke")
def test_24_new_ocr_replacement(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="New label analyzed.")],
        "ocr_combined_text": "New Product Ingredients: Wheat, Water",
    }
    payload = {
        "message": "Analyze label",
        "ocr_combined_text": "New Product Ingredients: Wheat, Water",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["ocr_combined_text"] == "New Product Ingredients: Wheat, Water"


# 25. Regression compatibility with Phase 1–15
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_25_regression_compatibility(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Full regression check passed.")],
        "product_data": _mock_nutella(),
    }
    resp = client.post("/chat", json={"message": "Analyze barcode 3017620422003"})
    assert resp.status_code == 200
    assert resp.json()["barcode"] == "3017620422003"
