from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from app.agents.comparison import analyze_comparison
from app.graph.workflow import build_workflow
from app.tools.comparison import compare_products
from app.tools.nutrition import NutritionData
from app.tools.product_lookup import ProductData, ProductLookupResult


def test_deterministic_product_comparison_same_serving():
    prod_a = ProductData(
        barcode="1111111111111",
        product_name="Product A (High Protein)",
        ingredients_text="Water, Soy Protein, Salt",
        allergens=["soy"],
        nutrition=NutritionData(
            serving_size_g=100.0,
            calories=150.0,
            protein_g=15.0,
            total_sugar_g=2.0,
            sodium_mg=300.0,
        ),
        source_url="http://test/a",
    )
    prod_b = ProductData(
        barcode="2222222222222",
        product_name="Product B (High Sugar)",
        ingredients_text="Water, Sugar, Salt",
        allergens=[],
        nutrition=NutritionData(
            serving_size_g=100.0,
            calories=200.0,
            protein_g=2.0,
            total_sugar_g=25.0,
            sodium_mg=500.0,
        ),
        source_url="http://test/b",
    )

    res = compare_products(prod_a, prod_b)

    assert res.product_a_name == "Product A (High Protein)"
    assert res.product_b_name == "Product B (High Sugar)"
    assert res.serving_size_match is True
    assert "soy" in res.allergens_a
    assert "soy" not in res.allergens_b
    assert "water" in res.common_ingredients
    assert "soy protein" in res.unique_ingredients_a
    assert "sugar" in res.unique_ingredients_b

    # Compare sugar: B is higher
    sugar = res.nutrients["total_sugar_g"]
    assert sugar.value_a_serving == 2.0
    assert sugar.value_b_serving == 25.0
    assert sugar.higher_product == "B"
    assert sugar.lower_product == "A"

    # Compare protein: A is higher
    protein = res.nutrients["protein_g"]
    assert protein.higher_product == "A"
    assert protein.lower_product == "B"


def test_deterministic_product_comparison_different_servings():
    prod_a = ProductData(
        barcode="111",
        product_name="Snack A",
        nutrition=NutritionData(
            serving_size_g=50.0,  # 50g serving
            calories=100.0,        # 200 kcal / 100g
            sodium_mg=200.0,       # 400 mg / 100g
        ),
        source_url="http://test/a",
    )
    prod_b = ProductData(
        barcode="222",
        product_name="Snack B",
        nutrition=NutritionData(
            serving_size_g=100.0, # 100g serving
            calories=180.0,       # 180 kcal / 100g
            sodium_mg=500.0,      # 500 mg / 100g
        ),
        source_url="http://test/b",
    )

    res = compare_products(prod_a, prod_b)
    assert res.serving_size_match is False
    assert len(res.data_quality_notes) >= 1
    assert "Serving sizes differ" in res.data_quality_notes[0]

    # Per 100g calories: A = 200, B = 180 -> B is lower per 100g
    cals = res.nutrients["calories"]
    assert cals.value_a_100g == 200.0
    assert cals.value_b_100g == 180.0
    assert cals.lower_product == "B"


def test_comparison_missing_fields_handling():
    prod_a = ProductData(
        barcode="111",
        product_name="Product A",
        nutrition=NutritionData(
            serving_size_g=100.0,
            calories=100.0,
            sodium_mg=200.0,
        ),
        source_url="http://test/a",
    )
    prod_b = ProductData(
        barcode="222",
        product_name="Product B",
        nutrition=NutritionData(
            serving_size_g=100.0,
            calories=150.0,
            # sodium_mg missing
        ),
        source_url="http://test/b",
    )

    res = compare_products(prod_a, prod_b)
    sod = res.nutrients["sodium_mg"]
    assert sod.status == "PARTIAL"
    assert sod.delta_100g is None
    assert sod.lower_product is None


def test_comparison_agent_analysis():
    prod_a = ProductData(
        barcode="111",
        product_name="Low Sodium Cereal",
        nutrition=NutritionData(serving_size_g=100.0, sodium_mg=100.0),
        source_url="http://test/a",
    )
    prod_b = ProductData(
        barcode="222",
        product_name="High Sodium Cereal",
        nutrition=NutritionData(serving_size_g=100.0, sodium_mg=600.0),
        source_url="http://test/b",
    )
    comp_res = compare_products(prod_a, prod_b)

    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        '{"available": true, "summary": "Product A has significantly lower sodium.", '
        '"nutritional_highlights": ["Product A contains 500mg less sodium per 100g."], '
        '"ingredient_highlights": [], "allergen_highlights": [], "uncertain_items": []}'
    )

    analysis = analyze_comparison(comp_res, conversation=[], provider=mock_provider)
    assert analysis.available is True
    assert "lower sodium" in analysis.summary
    assert len(analysis.nutritional_highlights) == 1


@patch("app.graph.nodes.lookup_product_by_barcode")
def test_comparison_workflow_integration(mock_lookup):
    prod_a = ProductData(
        barcode="3017620422003",
        product_name="Nutella Hazelnut Spread",
        nutrition=NutritionData(serving_size_g=30.0, calories=160.0, total_sugar_g=17.0),
        source_url="http://test/a",
    )
    prod_b = ProductData(
        barcode="0041196910759",
        product_name="Skippy Creamy Peanut Butter",
        nutrition=NutritionData(serving_size_g=32.0, calories=190.0, total_sugar_g=3.0),
        source_url="http://test/b",
    )

    def mock_lookup_side_effect(barcode):
        if "3017620422003" in barcode:
            return ProductLookupResult(found=True, product=prod_a)
        if "0041196910759" in barcode:
            return ProductLookupResult(found=True, product=prod_b)
        return ProductLookupResult(found=False)

    mock_lookup.side_effect = mock_lookup_side_effect

    mock_provider = MagicMock()
    mock_provider.generate_response.side_effect = [
        '{"intent": "PRODUCT_COMPARISON", "confidence": 0.95, "reason": "User asks to compare two products"}',
        '{"available": true, "summary": "Nutella has much higher sugar than Skippy.", "nutritional_highlights": ["Nutella: 17g sugar, Skippy: 3g sugar"], "ingredient_highlights": [], "allergen_highlights": [], "uncertain_items": []}',
        "Here is the comparison between Nutella and Skippy:\n\nSkippy Peanut Butter has significantly less sugar (3g vs 17g per serving).",
    ]

    app = build_workflow(provider=mock_provider)
    result = app.invoke(
        {
            "messages": [HumanMessage(content="Compare 3017620422003 and 0041196910759")],
            "comparison_barcodes": ["3017620422003", "0041196910759"],
        }
    )

    assert "messages" in result
    last_msg = result["messages"][-1].content
    assert "Skippy" in last_msg or "Nutella" in last_msg


@patch("httpx.Client.get")
def test_product_lookup_and_comparison_off_v2_nutrition(mock_get):
    from app.tools.product_lookup import lookup_product_by_barcode

    mock_resp_nutella = MagicMock()
    mock_resp_nutella.status_code = 200
    mock_resp_nutella.json.return_value = {
        "status": 1,
        "product": {
            "code": "3017620422003",
            "product_name": "Nutella",
            "brands": "Ferrero",
            "ingredients_text": "Sugar, Palm Oil, Hazelnuts, Cocoa, Skimmed Milk Powder",
            "allergens_tags": ["en:milk", "en:nuts", "en:soybeans"],
            "nutriments": {
                "energy-kcal_100g": 539.0,
                "sugars_100g": 56.3,
                "proteins_100g": 6.3,
                "sodium_100g": 0.0428,
                "fat_100g": 30.9,
            },
        },
    }

    mock_resp_pb = MagicMock()
    mock_resp_pb.status_code = 200
    mock_resp_pb.json.return_value = {
        "status": 1,
        "product": {
            "code": "0051500255162",
            "product_name": "Creamy Peanut Butter",
            "brands": "Jif",
            "serving_size": "1 serving (33 g)",
            "ingredients_text": "Made From Roasted Peanuts And Sugar",
            "allergens_tags": ["en:peanuts"],
            "nutriments": {
                "energy-kcal_serving": 190.0,
                "energy-kcal_100g": 575.76,
                "sugars_serving": 3.0,
                "sugars_100g": 9.09,
                "proteins_serving": 7.0,
                "proteins_100g": 21.21,
                "sodium_serving": 0.14,
                "sodium_100g": 0.424,
                "fat_serving": 16.0,
                "fat_100g": 48.48,
            },
        },
    }

    def get_side_effect(url, **kwargs):
        if "3017620422003" in url:
            return mock_resp_nutella
        if "0051500255162" in url:
            return mock_resp_pb
        return MagicMock(status_code=404, json=lambda: {"status": 0})

    mock_get.side_effect = get_side_effect

    lookup_a = lookup_product_by_barcode("3017620422003")
    lookup_b = lookup_product_by_barcode("0051500255162")

    assert lookup_a.found is True
    assert lookup_b.found is True
    assert lookup_a.product.nutrition is not None
    assert lookup_b.product.nutrition is not None

    comp = compare_products(lookup_a.product, lookup_b.product)
    assert comp.nutrients["total_sugar_g"].status == "COMPARED"
    assert comp.nutrients["protein_g"].status == "COMPARED"
    assert comp.nutrients["sodium_mg"].status == "COMPARED"

