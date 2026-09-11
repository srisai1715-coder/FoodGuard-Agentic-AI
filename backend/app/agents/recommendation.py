import json
from pydantic import BaseModel, Field

from app.agents.comparison import ComparisonAnalysis
from app.llm.provider import NVIDIAProvider
from app.tools.nutrition import NutritionData
from app.tools.product_lookup import ProductData


class RecommendationAnalysis(BaseModel):
    available: bool = True
    summary: str = Field(min_length=1)
    suitability_assessment: str = Field(default="")
    goal_alignment: list[str] = Field(default_factory=list)
    nutritional_tradeoffs: list[str] = Field(default_factory=list)
    safety_warnings: list[str] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)


RECOMMENDATION_SYSTEM_PROMPT = """You are the Product Recommendation Specialist Agent for FoodGuard AI.

Your role is to analyze available food/product data against the user's explicit preferences, dietary goals, or health priorities (e.g., lower sugar, higher protein, lower sodium, lower saturated fat, lower calories, lower total fat, simpler ingredients, allergen avoidance, fewer additives).

CRITICAL BOUNDARIES & INSTRUCTIONS:
1. GROUNDED EVIDENCE: Use only supplied evidence (product data, nutrition metrics, ingredient statements, safety/allergen info, product comparison results, OCR label text). Never fabricate missing facts.
2. MISSING DATA POLICY: Never guess missing values. If sugar, sodium, protein, or allergen data is missing, report uncertainty in `uncertain_items` instead of estimating.
3. ALLERGEN SAFETY: Never claim a product is "allergen-free" or "100% safe for allergies" based on missing data. Distinguish confirmed allergens from unverified statements.
4. MEDICAL BOUNDARY: This is a food decision-support system, NOT a medical tool. Never diagnose, treat, prevent, or cure diseases. Reframe health conditions (e.g., hypertension) as general dietary preferences (e.g., lower-sodium preference).
5. BALANCED TRADE-OFFS: Highlight important trade-offs (e.g., Product A has lower sugar but higher saturated fat than Product B).
6. COMPARISON REUSE: When comparing two products, evaluate the supplied comparison data objectively.

Return ONLY valid JSON matching this shape:
{
  "available": true,
  "summary": "...",
  "suitability_assessment": "...",
  "goal_alignment": ["..."],
  "nutritional_tradeoffs": ["..."],
  "safety_warnings": ["..."],
  "uncertain_items": ["..."]
}
"""


def parse_recommendation_analysis(raw_response: str) -> RecommendationAnalysis:
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found.")
        return RecommendationAnalysis.model_validate(json.loads(raw_response[start : end + 1]))
    except (ValueError, TypeError, json.JSONDecodeError):
        return RecommendationAnalysis(
            available=True,
            summary="Product recommendation analysis was processed, but formatting explanation had minor uncertainty.",
            suitability_assessment="Evaluated based on available product information.",
            uncertain_items=["Structured recommendation output formatting could not be fully parsed."],
        )


def analyze_recommendation(
    conversation: list[dict[str, str]],
    product_data: ProductData | None = None,
    nutrition_data: NutritionData | None = None,
    ingredient_data: str | None = None,
    safety_data: dict[str, str | None] | None = None,
    ocr_combined_text: str | None = None,
    ocr_nutrition_text: str | None = None,
    comparison_analysis: ComparisonAnalysis | None = None,
    comparison_products: list[ProductData] | None = None,
    provider: NVIDIAProvider | None = None,
    rag_context: str | None = None,
) -> RecommendationAnalysis:
    llm_provider = provider or NVIDIAProvider()
    content_parts = []

    if comparison_products:
        prods_info = [
            f"- {p.product_name or p.brand or p.barcode}: nutrition={p.nutrition.model_dump() if p.nutrition else None}, ingredients={p.ingredients_text}, allergens={p.allergens}"
            for p in comparison_products
        ]
        content_parts.append(f"Products for comparison:\n" + "\n".join(prods_info))

    if comparison_analysis and comparison_analysis.available:
        content_parts.append(
            f"Existing Phase 12 comparison analysis:\n{comparison_analysis.model_dump_json()}"
        )

    if product_data:
        content_parts.append(
            f"Normalized Single Product Data:\n{product_data.model_dump_json()}"
        )

    if nutrition_data:
        content_parts.append(
            f"Nutrition Data:\n{nutrition_data.model_dump_json()}"
        )

    if ingredient_data:
        content_parts.append(f"Ingredient Text: {ingredient_data}")

    if safety_data:
        content_parts.append(
            f"Safety & Allergen Information: {json.dumps(safety_data, ensure_ascii=True)}"
        )

    if ocr_combined_text or ocr_nutrition_text:
        content_parts.append(
            f"OCR Label Text:\nCombined: {ocr_combined_text or ''}\nNutrition Section: {ocr_nutrition_text or ''}"
        )

    if rag_context:
        content_parts.append(
            f"Authoritative Reference Knowledge:\n{rag_context}"
        )

    content_parts.append(
        f"Relevant User Conversation History & Goal: {json.dumps(conversation, ensure_ascii=True)}"
    )

    # Check if there is any product/OCR/comparison context available
    has_data = bool(
        product_data
        or nutrition_data
        or ingredient_data
        or (safety_data and any(safety_data.values()))
        or ocr_combined_text
        or ocr_nutrition_text
        or comparison_analysis
        or comparison_products
    )

    if not has_data:
        return RecommendationAnalysis(
            available=False,
            summary="Product data, ingredient labels, or comparison details are missing. Please scan a barcode, upload a label image, or specify products to compare for a personalized recommendation.",
            suitability_assessment="Unable to assess suitability without product data.",
            uncertain_items=["No product or label information is available."],
        )

    raw_response = llm_provider.generate_response(
        [
            {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\n".join(content_parts),
            },
        ]
    )
    return parse_recommendation_analysis(raw_response)
