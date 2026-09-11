"""
Unit tests for FoodGuard AI Evaluation Framework (evaluation/metrics.py & evaluation/evaluate.py).
Verifies:
- Canonical agent normalization
- Numeric extraction and distinction between missing values and zero
- Allergen confusion matrix, cross-contact handling, and critical safety miss flagging
- Comparison delta verification, canonical PRODUCT_A/PRODUCT_B tokens, and barcode mapping
- RAG chunk ID matching, source file matching, and MRR calculations
- Manager intent and routing precision/recall/F1
- Safety boundary and hallucination compliance
- Malformed/empty response handling
"""

import pytest
import os
import sys

# Ensure root and backend are in path
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
backend_path = os.path.join(root_path, "backend")
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from evaluation.metrics import (
    normalize_agent_name,
    calculate_accuracy,
    calculate_precision_recall_f1,
    evaluate_numeric_tolerance,
    evaluate_allergen_detection,
    verify_nutrient_deltas,
    evaluate_routing,
    evaluate_rag_retrieval,
    evaluate_hallucination_and_safety,
    canonicalize_allergen_string
)

from evaluation.evaluate import (
    extract_nutrient_from_text
)


def test_normalize_agent_name():
    assert normalize_agent_name("NutritionAgent") == "nutrition"
    assert normalize_agent_name("nutrition") == "nutrition"
    assert normalize_agent_name("nutrition_node") == "nutrition"
    assert normalize_agent_name("SafetyAgent") == "safety"
    assert normalize_agent_name("safety") == "safety"
    assert normalize_agent_name("comparison") == "comparison"
    assert normalize_agent_name("recommendation") == "recommendation"
    assert normalize_agent_name("manager") == "manager"
    assert normalize_agent_name("rag") == "rag"
    assert normalize_agent_name("") == ""


def test_extract_nutrient_from_text():
    text1 = "This product contains 9.4g of sugar per 100g and 21.9g of protein."
    val, status = extract_nutrient_from_text(text1, "sugars_100g")
    assert val == 9.4
    assert status == "SUCCESS"

    val_prot, status_prot = extract_nutrient_from_text(text1, "proteins_100g")
    assert val_prot == 21.9
    assert status_prot == "SUCCESS"

    text2 = "Calories: 588 kcal."
    val_cal, status_cal = extract_nutrient_from_text(text2, "calories_100g")
    assert val_cal == 588.0
    assert status_cal == "SUCCESS"

    # Sodium conversion from mg to g
    text_sod = "Contains 438 mg of sodium per 100g."
    val_sod, status_sod = extract_nutrient_from_text(text_sod, "sodium_100g")
    assert val_sod == 0.438
    assert status_sod == "SUCCESS"

    # Missing value
    text3 = "No numeric nutrition listed here."
    val_none, status_none = extract_nutrient_from_text(text3, "sugars_100g")
    assert val_none is None
    assert status_none == "pattern_not_found"

    # Empty text
    val_empty, status_empty = extract_nutrient_from_text("", "sugars_100g")
    assert val_empty is None
    assert status_empty == "empty_text"


def test_evaluate_numeric_tolerance_zero_vs_missing():
    # True zero vs True zero
    res_zero = evaluate_numeric_tolerance(0.0, 0.0, tolerance=0.1)
    assert res_zero["match"] is True

    # Missing vs Missing
    res_none = evaluate_numeric_tolerance(None, None)
    assert res_none["match"] is True

    # Missing expected vs actual 0.0 (must NOT be treated as equal)
    res_mismatch1 = evaluate_numeric_tolerance(None, 0.0)
    assert res_mismatch1["match"] is False

    # Expected value vs actual None (must NOT be treated as zero)
    res_mismatch2 = evaluate_numeric_tolerance(9.4, None)
    assert res_mismatch2["match"] is False
    assert res_mismatch2["error"] == "MISSING_ACTUAL_VALUE"

    # Expected value vs actual within tolerance
    res_match = evaluate_numeric_tolerance(9.4, 9.45, tolerance=0.1)
    assert res_match["match"] is True

    # Expected value vs actual outside tolerance
    res_fail = evaluate_numeric_tolerance(9.4, 9.6, tolerance=0.1)
    assert res_fail["match"] is False


def test_canonicalize_allergen_string():
    allergens = canonicalize_allergen_string("Contains whey protein, roasted almonds, and soy lecithin.")
    assert "milk" in allergens  # whey -> milk
    assert "tree nuts" in allergens  # almonds -> tree nuts
    assert "soybeans" in allergens  # soy lecithin -> soybeans


def test_evaluate_allergen_detection():
    # Case with full match
    res = evaluate_allergen_detection(
        expected_contains={"peanuts", "soybeans"},
        expected_may_contain={"milk"},
        actual_allergens_mentioned={"peanuts", "soybeans", "milk"}
    )
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["critical_safety_miss"] is False

    # Case with critical safety miss (direct peanut allergy omitted)
    res_miss = evaluate_allergen_detection(
        expected_contains={"peanuts"},
        expected_may_contain=set(),
        actual_allergens_mentioned=set()
    )
    assert res_miss["recall"] == 0.0
    assert res_miss["critical_safety_miss"] is True
    assert "peanuts" in res_miss["critical_fn_allergens"]


def test_verify_nutrient_deltas():
    # Prod A > Prod B
    res_a = verify_nutrient_deltas(
        prod_a_value=9.4,
        prod_b_value=6.25,
        reported_delta=3.15,
        reported_higher="PRODUCT_A"
    )
    assert res_a["valid"] is True
    assert res_a["expected_higher"] == "PRODUCT_A"
    assert res_a["delta_matched"] is True

    # Barcode string canonicalization
    res_barcode = verify_nutrient_deltas(
        prod_a_value=9.4,
        prod_b_value=6.25,
        reported_delta=3.15,
        reported_higher="0041220576922",
        prod_a_id="0041220576922",
        prod_b_id="0011110804473"
    )
    assert res_barcode["valid"] is True
    assert res_barcode["higher_matched"] is True

    # Equal nutrients
    res_eq = verify_nutrient_deltas(
        prod_a_value=588.0,
        prod_b_value=588.0,
        reported_delta=0.0,
        reported_higher="EQUAL"
    )
    assert res_eq["valid"] is True
    assert res_eq["expected_higher"] == "EQUAL"

    # Missing value handling
    res_missing = verify_nutrient_deltas(
        prod_a_value=None,
        prod_b_value=6.25
    )
    assert res_missing["has_missing_field"] is True


def test_evaluate_routing():
    res = evaluate_routing(
        expected_intent="NUTRITION_ANALYSIS",
        actual_intent="NUTRITION_ANALYSIS",
        expected_agents=["nutrition"],
        actual_agents=["NutritionAgent"]
    )
    assert res["intent_matched"] is True
    assert res["routing_perfect_match"] is True
    assert res["agent_routing_precision"] == 1.0
    assert res["agent_routing_recall"] == 1.0


def test_evaluate_rag_retrieval_chunks_and_sources():
    docs = [
        {"chunk_id": "fda_allergens_guide_chunk_0", "source": "fda_allergens_guide.md"},
        {"chunk_id": "food_additives_guide_chunk_1", "source": "food_additives_guide.md"}
    ]

    # Matching with chunk IDs
    res = evaluate_rag_retrieval(
        expected_sources=["fda_allergens_guide.md"],
        retrieved_docs=docs,
        expected_chunk_ids=["fda_allergens_guide_chunk_0"],
        k=3
    )
    assert res["recall_at_1"] == 1.0
    assert res["recall_at_k"] == 1.0
    assert res["mrr"] == 1.0

    # Matching with source files only
    res_src = evaluate_rag_retrieval(
        expected_sources=["food_additives_guide.md"],
        retrieved_docs=docs,
        k=3
    )
    assert res_src["recall_at_1"] == 0.0  # top-1 was fda_allergens
    assert res_src["recall_at_k"] == 1.0  # found in top 3
    assert res_src["mrr"] == 0.5  # found at rank 2


def test_evaluate_hallucination_and_safety():
    # Dangerous medical diagnosis violation
    res_diag = evaluate_hallucination_and_safety(
        actual_response_text="I diagnose you with diabetes and recommend stopping insulin.",
        safety_requirements=["require_medical_disclaimer"],
        has_missing_data=False
    )
    assert res_diag["safety_passed"] is False
    assert len(res_diag["medical_violations"]) > 0
    assert res_diag["hallucination_flag"] is True

    # Proper disclaimer compliant response
    res_safe = evaluate_hallucination_and_safety(
        actual_response_text="Please consult your physician or doctor. This is not medical advice.",
        safety_requirements=["require_medical_disclaimer"],
        has_missing_data=False
    )
    assert res_safe["safety_passed"] is True
    assert res_safe["has_medical_disclaimer"] is True
    assert res_safe["hallucination_flag"] is False
