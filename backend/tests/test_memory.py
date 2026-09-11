from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

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


def _mock_nutella_product() -> ProductData:
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


def _mock_peanut_butter_product() -> ProductData:
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
        return ProductLookupResult(found=True, product=_mock_nutella_product())
    if barcode == "0051500255162":
        return ProductLookupResult(found=True, product=_mock_peanut_butter_product())
    return ProductLookupResult(found=False, error="PRODUCT_NOT_FOUND")


# ------------------------------------------------------------
# TEST 1 — INITIAL SINGLE BARCODE
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_1_initial_single_barcode(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Nutella is a hazelnut cocoa spread.")],
        "product_data": _mock_nutella_product(),
    }

    response = client.post(
        "/chat",
        json={"message": "Analyze barcode 3017620422003"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["barcode"] == "3017620422003"
    assert "Nutella" in data["response"]


# ------------------------------------------------------------
# TEST 2 — SINGLE BARCODE FOLLOW-UP
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_2_single_barcode_followup(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Nutella contains 56.3g of sugar per 100g.")],
        "product_data": _mock_nutella_product(),
    }

    payload = {
        "message": "What about the sugar?",
        "messages": [
            {"role": "user", "content": "Analyze barcode 3017620422003"},
            {"role": "assistant", "content": "Nutella is a hazelnut cocoa spread."},
        ],
        "barcode": "3017620422003",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["barcode"] == "3017620422003"
    assert "sugar" in data["response"].lower()


# ------------------------------------------------------------
# TEST 3 — FOLLOW-UP PRONOUN
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_3_followup_pronoun(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Yes, 56.3g of sugar is considered high.")],
        "product_data": _mock_nutella_product(),
    }

    payload = {
        "message": "Is that high?",
        "messages": [
            {"role": "user", "content": "Analyze barcode 3017620422003"},
            {"role": "assistant", "content": "Nutella contains 56.3g of sugar."},
        ],
        "barcode": "3017620422003",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["barcode"] == "3017620422003"


# ------------------------------------------------------------
# TEST 4 — INGREDIENT FOLLOW-UP
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_4_ingredient_followup(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [
            AIMessage(content="Nutella analyzed."),
        ],
        "product_lookup_request": "3017620422003",
        "product_data": None,
        "ingredient_data": None,
    }
    result = ingredient_node(state)
    assert result["ingredient_analysis"].available is True
    assert "Sugar" in result["ingredient_data"]
    assert result["product_data"].product_name == "Nutella"


# ------------------------------------------------------------
# TEST 5 — ALLERGEN FOLLOW-UP
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_5_allergen_followup(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [
            AIMessage(content="Nutella analyzed."),
        ],
        "product_lookup_request": "3017620422003",
        "product_data": None,
        "safety_data": {},
    }
    result = safety_node(state)
    assert result["safety_analysis"].available is True
    assert result["product_data"].product_name == "Nutella"


# ------------------------------------------------------------
# TEST 6 — COMPARISON FOLLOW-UP
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_6_comparison_followup(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Creamy Peanut Butter has significantly less sugar than Nutella.")],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }

    payload = {
        "message": "Which one has less sugar?",
        "messages": [
            {"role": "user", "content": "Compare 3017620422003 and 0051500255162"},
            {"role": "assistant", "content": "Here is the comparison table for Nutella and Peanut Butter."},
        ],
        "barcodes": ["3017620422003", "0051500255162"],
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["comparison_barcodes"] == ["3017620422003", "0051500255162"]
    assert "Peanut Butter" in data["response"] or "sugar" in data["response"].lower()


# ------------------------------------------------------------
# TEST 7 — COMPARISON PROTEIN FOLLOW-UP
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_7_comparison_protein_followup(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [
            AIMessage(content="Previous comparison done."),
        ],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }
    result = comparison_node(state)
    assert result["comparison_analysis"].available is True
    assert len(result["comparison_products"]) == 2
    assert result["comparison_barcodes"] == ["3017620422003", "0051500255162"]


# ------------------------------------------------------------
# TEST 8 — RECOMMENDATION FOLLOW-UP
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_8_recommendation_followup(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [
            AIMessage(content="Recommend lower sugar product."),
        ],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }
    result = recommendation_node(state)
    assert result["recommendation_analysis"].available is True
    assert len(result["comparison_products"]) == 2


# ------------------------------------------------------------
# TEST 9 — RECOMMENDATION JUSTIFICATION
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_9_recommendation_justification(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="I recommended Creamy Peanut Butter because it contains 9.4g of sugar vs 56.3g in Nutella.")],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }

    payload = {
        "message": "Why did you recommend that one?",
        "messages": [
            {"role": "user", "content": "Which product is better if I want lower sugar? 3017620422003 and 0051500255162"},
            {"role": "assistant", "content": "I recommend Creamy Peanut Butter due to lower sugar content."},
        ],
        "barcodes": ["3017620422003", "0051500255162"],
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Creamy Peanut Butter" in data["response"] or "sugar" in data["response"].lower()


# ------------------------------------------------------------
# TEST 10 — OCR MULTI-TURN
# ------------------------------------------------------------
@patch("app.main.workflow.invoke")
def test_10_ocr_multi_turn(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="The OCR text shows 12g of sugar per serving.")],
    }

    payload = {
        "message": "What about sugar?",
        "messages": [
            {"role": "user", "content": "Analyze uploaded label"},
            {"role": "assistant", "content": "I extracted ingredients: Peanuts, Salt."},
        ],
        "ocr_combined_text": "Ingredients: Peanuts, Salt. Nutrition: Sugar 12g, Protein 8g.",
        "ocr_nutrition_text": "Sugar 12g, Protein 8g",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ocr_combined_text"] == payload["ocr_combined_text"]


# ------------------------------------------------------------
# TEST 11 — NEW PRODUCT REPLACEMENT
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_11_new_product_replacement(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Analyzed Creamy Peanut Butter.")],
        "product_data": _mock_peanut_butter_product(),
    }

    payload = {
        "message": "Analyze barcode 0051500255162",
        "messages": [
            {"role": "user", "content": "Analyze barcode 3017620422003"},
            {"role": "assistant", "content": "Nutella analyzed."},
        ],
        "barcode": "3017620422003",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["barcode"] == "0051500255162"


# ------------------------------------------------------------
# TEST 12 — NEW BARCODE REPLACEMENT
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_12_new_barcode_replacement(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [
            AIMessage(content="Previous nutella context"),
        ],
        "product_lookup_request": "0051500255162",
    }
    result = product_lookup_node(state)
    assert result["product_data"].barcode == "0051500255162"
    assert result["product_data"].product_name == "Creamy Peanut Butter"


# ------------------------------------------------------------
# TEST 13 — GENERAL RAG QUESTION
# ------------------------------------------------------------
@patch("app.main.workflow.invoke")
def test_13_general_rag_question(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Peanuts are one of the 9 major food allergens required to be labeled under FASTER Act.")],
    }

    payload = {
        "message": "Why are peanuts a major food allergen?",
        "messages": [
            {"role": "user", "content": "Analyze barcode 3017620422003"},
            {"role": "assistant", "content": "Nutella analyzed."},
        ],
        "barcode": "3017620422003",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "allergens" in data["response"].lower() or "peanuts" in data["response"].lower()


# ------------------------------------------------------------
# TEST 14 — CROSS-CONTEXT ISOLATION
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_14_cross_context_isolation(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Nutella response")],
        "product_data": _mock_nutella_product(),
    }
    req_a = client.post("/chat", json={"message": "What about sugar?", "barcode": "3017620422003"})
    assert req_a.json()["barcode"] == "3017620422003"

    mock_invoke.return_value = {
        "messages": [AIMessage(content="Peanut butter response")],
        "product_data": _mock_peanut_butter_product(),
    }
    req_b = client.post("/chat", json={"message": "What about protein?", "barcode": "0051500255162"})
    assert req_b.json()["barcode"] == "0051500255162"


# ------------------------------------------------------------
# TEST 15 — NEW CHAT RESET
# ------------------------------------------------------------
def test_15_new_chat_reset() -> None:
    req = ChatRequest(
        message="Hello",
        messages=[],
        barcode=None,
        barcodes=[],
        conversation_id=None,
        ocr_combined_text=None,
    )
    assert req.barcode is None
    assert req.barcodes == []
    assert req.conversation_id is None


# ------------------------------------------------------------
# TEST 16 — VERIFICATION REGRESSION
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
@patch("app.main.workflow.invoke")
def test_16_verification_regression(mock_invoke: MagicMock, mock_lookup: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Verified Nutella nutrition analysis.")],
        "verification_analysis": None,
    }
    response = client.post("/chat", json={"message": "Analyze barcode 3017620422003"})
    assert response.status_code == 200


# ------------------------------------------------------------
# TEST 17 — COMPARISON REGRESSION
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_17_comparison_regression(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [
            AIMessage(content="User wants comparison"),
        ],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }
    result = comparison_node(state)
    assert result["comparison_analysis"].available is True
    assert len(result["comparison_products"]) == 2


# ------------------------------------------------------------
# TEST 18 — RECOMMENDATION REGRESSION
# ------------------------------------------------------------
@patch("app.graph.nodes.lookup_product_by_barcode", side_effect=_mock_lookup_side_effect)
def test_18_recommendation_regression(mock_lookup: MagicMock) -> None:
    state: FoodGuardState = {
        "messages": [
            AIMessage(content="Recommend"),
        ],
        "comparison_barcodes": ["3017620422003", "0051500255162"],
    }
    result = recommendation_node(state)
    assert result["recommendation_analysis"].available is True


# ------------------------------------------------------------
# TEST 19 — OCR REGRESSION
# ------------------------------------------------------------
@patch("app.main.workflow.invoke")
def test_19_ocr_regression(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="OCR text analyzed.")],
    }

    payload = {
        "message": "Analyze this text",
        "ocr_combined_text": "Ingredients: Oats, Honey, Water",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert response.json()["ocr_combined_text"] == "Ingredients: Oats, Honey, Water"


# ------------------------------------------------------------
# TEST 20 — RAG REGRESSION
# ------------------------------------------------------------
@patch("app.main.workflow.invoke")
def test_20_rag_regression(mock_invoke: MagicMock) -> None:
    mock_invoke.return_value = {
        "messages": [AIMessage(content="FDA daily value for added sugars is 50g per day.")],
    }
    response = client.post("/chat", json={"message": "What is the daily value for sugar?"})
    assert response.status_code == 200
    assert "daily value" in response.json()["response"].lower() or "50g" in response.json()["response"]
