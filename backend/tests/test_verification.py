from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage

from app.agents.comparison import ComparisonAnalysis
from app.agents.recommendation import RecommendationAnalysis
from app.agents.verification import VerificationAnalysis, VerificationStatus, verify_analysis
from app.graph.workflow import build_workflow
from app.tools.comparison import compare_products
from app.tools.nutrition import NutritionData
from app.tools.product_lookup import ProductData, ProductLookupResult


def test_1_verified_nutrition_claim():
    prod = ProductData(
        barcode="1001",
        product_name="Product A",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=9.0),
        source_url="http://test/1001",
    )
    res = verify_analysis(
        conversation=[{"role": "user", "content": "What is the sugar content?"}],
        product_data=prod,
        nutrition_data=prod.nutrition,
    )
    assert res.verified is True
    assert res.status == VerificationStatus.VERIFIED


def test_2_conflicting_nutrition_claim():
    prod = ProductData(
        barcode="1001",
        product_name="Product A",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=9.0),
        source_url="http://test/1001",
    )
    # Mock LLM auditing a claim that product has 15g sugar
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"verified": false, "status": "CONFLICT", "summary": "Claim states 15g sugar but actual data is 9g.", '
        '"verified_claims": [], "unsupported_claims": [], "conflicting_claims": ["Claim says 15g sugar, actual is 9g"], '
        '"safety_flags": [], "uncertain_items": []}'
    )
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Does this product have 15g sugar?"}],
        product_data=prod,
        nutrition_data=prod.nutrition,
        recommendation_analysis=RecommendationAnalysis(summary="Has 15g sugar"),
        provider=mock_provider,
    )
    assert res.verified is False
    assert res.status == VerificationStatus.CONFLICT
    assert len(res.conflicting_claims) == 1


def test_3_verified_comparison():
    prod_a = ProductData(
        barcode="1001",
        product_name="High Sugar Cereal",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=25.0),
        source_url="http://test/1001",
    )
    prod_b = ProductData(
        barcode="1002",
        product_name="Low Sugar Cereal",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=5.0),
        source_url="http://test/1002",
    )
    comp_res = compare_products(prod_a, prod_b)
    comp_analysis = ComparisonAnalysis(
        summary="Low Sugar Cereal has lower sugar than High Sugar Cereal."
    )
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Compare sugar content."}],
        comparison_products=[prod_a, prod_b],
        comparison_analysis=comp_analysis,
    )
    assert res.verified is True


def test_4_wrong_comparison():
    prod_a = ProductData(
        barcode="1001",
        product_name="High Sugar Cereal",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=25.0),
        source_url="http://test/1001",
    )
    prod_b = ProductData(
        barcode="1002",
        product_name="Low Sugar Cereal",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=5.0),
        source_url="http://test/1002",
    )
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"verified": false, "status": "CONFLICT", "summary": "Claim says High Sugar Cereal has less sugar, but Low Sugar Cereal has 5g vs 25g.", '
        '"verified_claims": [], "unsupported_claims": [], "conflicting_claims": ["High Sugar Cereal has lower sugar claim is false."], '
        '"safety_flags": [], "uncertain_items": []}'
    )
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Which cereal has lower sugar?"}],
        comparison_products=[prod_a, prod_b],
        comparison_analysis=ComparisonAnalysis(summary="High Sugar Cereal has lower sugar."),
        provider=mock_provider,
    )
    assert res.verified is False
    assert res.status == VerificationStatus.CONFLICT


def test_5_verified_recommendation():
    rec_analysis = RecommendationAnalysis(
        summary="Product B is recommended for lower sugar goal.",
        goal_alignment=["Product B has 5g sugar vs Product A 25g sugar."],
    )
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Which product is better if I want lower sugar?"}],
        recommendation_analysis=rec_analysis,
    )
    assert res.verified is True
    assert res.status == VerificationStatus.VERIFIED


def test_6_contradictory_recommendation():
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"verified": false, "status": "CONFLICT", "summary": "Recommendation recommends Product A for lower sugar, but Product A contains 5 times more sugar.", '
        '"verified_claims": [], "unsupported_claims": [], "conflicting_claims": ["Product A recommended for lower sugar contradicts actual values."], '
        '"safety_flags": [], "uncertain_items": []}'
    )
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Which product is better for lower sugar?"}],
        recommendation_analysis=RecommendationAnalysis(summary="Product A recommended for lower sugar."),
        provider=mock_provider,
    )
    assert res.verified is False
    assert res.status == VerificationStatus.CONFLICT


def test_7_missing_data():
    prod = ProductData(
        barcode="1004",
        product_name="Unknown Cereal",
        nutrition=None,
        source_url="http://test/1004",
    )
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Is this lower in sodium?"}],
        product_data=prod,
    )
    assert res.verified is False
    assert res.status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert len(res.uncertain_items) >= 1


def test_8_unsupported_allergen_free_claim():
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Is this product 100% safe and completely allergen-free for peanuts?"}],
        safety_data={"ingredient_text": "Oats, Honey, Flavor"},
    )
    assert res.verified is False
    assert len(res.safety_flags) >= 1


def test_9_unsupported_medical_claim():
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Is this product medically safe for hypertension and will it cure high blood pressure?"}],
        safety_data={"ingredient_text": "Salt, Spices"},
    )
    assert res.verified is False
    assert len(res.safety_flags) >= 1


def test_10_ocr_vs_structured_conflict():
    prod = ProductData(
        barcode="1005",
        product_name="Juice Box",
        nutrition=NutritionData(serving_size_g=100.0, total_sugar_g=20.0),
        source_url="http://test/1005",
    )
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"verified": false, "status": "CONFLICT", "summary": "OCR reads 18g sugar while database reads 20g sugar.", '
        '"verified_claims": [], "unsupported_claims": [], "conflicting_claims": ["OCR sugar 18g vs database sugar 20g."], '
        '"safety_flags": [], "uncertain_items": ["Discrepancy between OCR and database sugar values."]}'
    )
    res = verify_analysis(
        conversation=[{"role": "user", "content": "What is the sugar content?"}],
        product_data=prod,
        ocr_nutrition_text="Sugar 18g",
        recommendation_analysis=RecommendationAnalysis(summary="Sugar is 18g according to OCR."),
        provider=mock_provider,
    )
    assert res.verified is False
    assert res.status == VerificationStatus.CONFLICT


def test_11_rag_supported_claim():
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Why is sesame a major food allergen?"}],
        rag_context="Sesame is listed as the 9th major food allergen under the FASTER Act.",
    )
    assert res.verified is True
    assert res.status == VerificationStatus.VERIFIED


def test_12_rag_unsupported_claim():
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"verified": false, "status": "UNSUPPORTED", "summary": "Claim that apples are a major FDA allergen is unsupported.", '
        '"verified_claims": [], "unsupported_claims": ["Apples are not an FDA Big 9 major food allergen."], '
        '"conflicting_claims": [], "safety_flags": [], "uncertain_items": []}'
    )
    res = verify_analysis(
        conversation=[{"role": "user", "content": "Are apples one of the Big 9 major food allergens?"}],
        rag_context="The Big 9 major food allergens are milk, eggs, fish, shellfish, tree nuts, peanuts, wheat, soybeans, and sesame.",
        recommendation_analysis=RecommendationAnalysis(summary="Apples are an FDA major allergen."),
        provider=mock_provider,
    )
    assert res.verified is False
    assert res.status == VerificationStatus.UNSUPPORTED


@patch("app.graph.nodes.lookup_product_by_barcode")
def test_15_general_question_and_graph_routing(mock_lookup):
    prod_a = ProductData(
        barcode="3017620422003",
        product_name="Nutella",
        nutrition=NutritionData(serving_size_g=30.0, calories=160.0, total_sugar_g=17.0),
        source_url="http://test/a",
    )
    mock_lookup.return_value = ProductLookupResult(found=True, product=prod_a)

    mock_provider = MagicMock()
    mock_provider.generate_response.side_effect = [
        '{"intent": "NUTRITION_ANALYSIS", "confidence": 0.95, "reason": "User asks about sugar content"}',
        '{"available": true, "summary": "Nutella has 17g sugar per serving.", "key_nutrients": ["Sugar: 17g"], "dietary_flags": [], "uncertain_items": []}',
        "Nutella contains 17g of sugar per 30g serving, which has been verified against product database entries.",
    ]

    app = build_workflow(provider=mock_provider)
    result = app.invoke(
        {
            "messages": [HumanMessage(content="What is the sugar content of Nutella? Barcode 3017620422003")],
            "product_lookup_request": "3017620422003",
        }
    )

    assert "messages" in result
    last_msg = result["messages"][-1].content
    assert "Nutella" in last_msg or "sugar" in last_msg
