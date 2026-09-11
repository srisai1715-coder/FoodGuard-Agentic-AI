from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage

from app.agents.recommendation import RecommendationAnalysis, analyze_recommendation
from app.graph.workflow import build_workflow
from app.tools.comparison import ComparisonResult, compare_products
from app.tools.nutrition import NutritionData
from app.tools.product_lookup import ProductData, ProductLookupResult


def test_recommendation_lower_sodium():
    prod = ProductData(
        barcode="1001",
        product_name="Low Sodium Soup",
        nutrition=NutritionData(serving_size_g=100.0, sodium_mg=120.0, calories=80.0),
        source_url="http://test/1001",
    )
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"available": true, "summary": "Low Sodium Soup is a suitable choice for a low-sodium diet.", '
        '"suitability_assessment": "Contains 120mg sodium per 100g serving.", '
        '"goal_alignment": ["Matches lower sodium goal with only 120mg sodium."], '
        '"nutritional_tradeoffs": [], "safety_warnings": [], "uncertain_items": []}'
    )

    res = analyze_recommendation(
        conversation=[{"role": "user", "content": "I want a lower sodium option for hypertension guidance."}],
        product_data=prod,
        nutrition_data=prod.nutrition,
        provider=mock_provider,
    )

    assert res.available is True
    assert "low-sodium" in res.summary.lower()
    assert len(res.goal_alignment) == 1
    assert "sodium" in res.goal_alignment[0].lower()


def test_recommendation_higher_protein():
    prod = ProductData(
        barcode="1002",
        product_name="High Protein Shake",
        nutrition=NutritionData(serving_size_g=100.0, protein_g=25.0, calories=150.0),
        source_url="http://test/1002",
    )
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"available": true, "summary": "High Protein Shake aligns well with your higher protein goal.", '
        '"suitability_assessment": "Provides 25g protein per serving.", '
        '"goal_alignment": ["Delivers 25g protein per 100g."], '
        '"nutritional_tradeoffs": [], "safety_warnings": [], "uncertain_items": []}'
    )

    res = analyze_recommendation(
        conversation=[{"role": "user", "content": "Which option is best for higher protein?"}],
        product_data=prod,
        nutrition_data=prod.nutrition,
        provider=mock_provider,
    )

    assert res.available is True
    assert "protein" in res.summary.lower()
    assert len(res.goal_alignment) == 1


def test_recommendation_lower_sugar():
    prod = ProductData(
        barcode="1003",
        product_name="Zero Sugar Yogurt",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=1.0, calories=60.0),
        source_url="http://test/1003",
    )
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"available": true, "summary": "Zero Sugar Yogurt is an excellent lower-sugar choice.", '
        '"suitability_assessment": "Contains only 1g sugar per serving.", '
        '"goal_alignment": ["Only 1g sugar per 100g."], '
        '"nutritional_tradeoffs": [], "safety_warnings": [], "uncertain_items": []}'
    )

    res = analyze_recommendation(
        conversation=[{"role": "user", "content": "Which yogurt is better if I want less sugar?"}],
        product_data=prod,
        nutrition_data=prod.nutrition,
        provider=mock_provider,
    )

    assert res.available is True
    assert "lower-sugar" in res.summary.lower() or "sugar" in res.summary.lower()


def test_recommendation_ocr_flow():
    ocr_text = "Ingredients: Oats, Honey, Almonds. Nutrition Facts: Calories 150, Protein 5g, Sugar 8g."
    ocr_nutr = "Calories 150, Protein 5g, Sugar 8g."

    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"available": true, "summary": "Extracted label information shows a balanced oat bar.", '
        '"suitability_assessment": "Contains 5g protein and 8g sugar.", '
        '"goal_alignment": ["Contains 5g protein."], '
        '"nutritional_tradeoffs": ["Contains 8g added/total sugar."], '
        '"safety_warnings": ["Contains almonds (tree nuts)."], "uncertain_items": []}'
    )

    res = analyze_recommendation(
        conversation=[{"role": "user", "content": "Is this OCR label good for my high protein goal?"}],
        ocr_combined_text=ocr_text,
        ocr_nutrition_text=ocr_nutr,
        provider=mock_provider,
    )

    assert res.available is True
    assert "oat bar" in res.summary.lower()
    assert len(res.safety_warnings) == 1


def test_recommendation_allergen_preference():
    safety_info = {
        "ingredient_text": "Milk, Wheat, Sugar",
        "allergen_statement": "Contains Milk, Wheat",
        "cross_contact_statement": "May contain peanuts",
    }
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"available": true, "summary": "Product contains confirmed milk and wheat allergens with potential peanut cross-contact.", '
        '"suitability_assessment": "Unsuitable for individuals with milk, wheat, or peanut allergies.", '
        '"goal_alignment": [], '
        '"nutritional_tradeoffs": [], '
        '"safety_warnings": ["Contains Milk and Wheat.", "May contain peanuts via cross-contact."], '
        '"uncertain_items": ["Soy allergen status is unverified and cannot be assumed absent."]}'
    )

    res = analyze_recommendation(
        conversation=[{"role": "user", "content": "Is this safe for someone with peanut and milk allergies?"}],
        safety_data=safety_info,
        provider=mock_provider,
    )

    assert res.available is True
    assert len(res.safety_warnings) == 2
    assert len(res.uncertain_items) == 1
    assert "absent" in res.uncertain_items[0] or "unverified" in res.uncertain_items[0]


def test_recommendation_missing_data_handling():
    prod = ProductData(
        barcode="1004",
        product_name="Unknown Cereal",
        nutrition=None,
        source_url="http://test/1004",
    )
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"available": true, "summary": "Nutrition facts are missing for Unknown Cereal.", '
        '"suitability_assessment": "Cannot verify sodium or sugar content.", '
        '"goal_alignment": [], "nutritional_tradeoffs": [], "safety_warnings": [], '
        '"uncertain_items": ["Sodium content is missing.", "Sugar content is missing."]}'
    )

    res = analyze_recommendation(
        conversation=[{"role": "user", "content": "Is this cereal low in sodium?"}],
        product_data=prod,
        provider=mock_provider,
    )

    assert res.available is True
    assert len(res.uncertain_items) >= 2


def test_recommendation_two_product_comparison():
    prod_a = ProductData(
        barcode="3017620422003",
        product_name="Nutella",
        nutrition=NutritionData(serving_size_g=30.0, calories=160.0, total_sugar_g=17.0, protein_g=2.0),
        source_url="http://test/nutella",
    )
    prod_b = ProductData(
        barcode="0051500255162",
        product_name="Creamy Peanut Butter",
        nutrition=NutritionData(serving_size_g=33.0, calories=190.0, total_sugar_g=3.0, protein_g=7.0),
        source_url="http://test/pb",
    )
    comp_res = compare_products(prod_a, prod_b)

    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"available": true, "summary": "Creamy Peanut Butter is recommended for a lower-sugar goal.", '
        '"suitability_assessment": "Peanut Butter has 3g sugar vs Nutella 17g sugar per serving.", '
        '"goal_alignment": ["Peanut Butter has 14g less sugar per serving."], '
        '"nutritional_tradeoffs": ["Peanut butter has higher calories and total fat."], '
        '"safety_warnings": ["Peanut Butter contains peanuts."], "uncertain_items": []}'
    )

    res = analyze_recommendation(
        conversation=[{"role": "user", "content": "Which is better if I want lower sugar?"}],
        comparison_products=[prod_a, prod_b],
        provider=mock_provider,
    )

    assert res.available is True
    assert "peanut butter" in res.summary.lower()
    assert len(res.nutritional_tradeoffs) == 1


def test_recommendation_tradeoffs():
    prod_a = ProductData(
        barcode="2001",
        product_name="Option A",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=2.0, saturated_fat_g=8.0),
        source_url="http://test/a",
    )
    prod_b = ProductData(
        barcode="2002",
        product_name="Option B",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=10.0, saturated_fat_g=1.0),
        source_url="http://test/b",
    )

    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"available": true, "summary": "Option A wins on sugar, but Option B has much lower saturated fat.", '
        '"suitability_assessment": "Option A aligns with lower sugar, but presents a saturated fat trade-off.", '
        '"goal_alignment": ["Option A has 2g sugar vs Option B 10g sugar."], '
        '"nutritional_tradeoffs": ["Option A contains 8g saturated fat vs Option B 1g saturated fat."], '
        '"safety_warnings": [], "uncertain_items": []}'
    )

    res = analyze_recommendation(
        conversation=[{"role": "user", "content": "Which is better if lower sugar is my priority?"}],
        comparison_products=[prod_a, prod_b],
        provider=mock_provider,
    )

    assert res.available is True
    assert len(res.nutritional_tradeoffs) == 1
    assert "saturated fat" in res.nutritional_tradeoffs[0].lower()


@patch("app.graph.nodes.lookup_product_by_barcode")
def test_recommendation_graph_routing(mock_lookup):
    prod_a = ProductData(
        barcode="3017620422003",
        product_name="Nutella",
        nutrition=NutritionData(serving_size_g=30.0, calories=160.0, total_sugar_g=17.0),
        source_url="http://test/a",
    )
    mock_lookup.return_value = ProductLookupResult(found=True, product=prod_a)

    mock_provider = MagicMock()
    mock_provider.generate_response.side_effect = [
        '{"intent": "RECOMMENDATION", "confidence": 0.95, "reason": "User asks which product is better based on preference"}',
        '{"available": true, "summary": "Nutella is moderate in sugar and calories.", "suitability_assessment": "Contains 17g sugar.", "goal_alignment": ["Contains 160 kcal."], "nutritional_tradeoffs": ["High in sugar."], "safety_warnings": [], "uncertain_items": []}',
        "Based on your preference for a lower-sugar choice, Nutella contains 17g of sugar per serving which is relatively high.",
    ]

    app = build_workflow(provider=mock_provider)
    result = app.invoke(
        {
            "messages": [HumanMessage(content="Which product is better if I want lower sugar? Barcode 3017620422003")],
            "product_lookup_request": "3017620422003",
        }
    )

    assert "messages" in result
    last_msg = result["messages"][-1].content
    assert "Nutella" in last_msg or "sugar" in last_msg
