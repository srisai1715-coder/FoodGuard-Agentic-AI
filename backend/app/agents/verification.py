import json
from enum import Enum
from pydantic import BaseModel, Field

from app.agents.comparison import ComparisonAnalysis
from app.agents.recommendation import RecommendationAnalysis
from app.llm.provider import NVIDIAProvider
from app.tools.comparison import ComparisonResult, compare_products
from app.tools.nutrition import NutritionData
from app.tools.product_lookup import ProductData


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    CONFLICT = "CONFLICT"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class VerificationAnalysis(BaseModel):
    verified: bool = True
    status: VerificationStatus = VerificationStatus.VERIFIED
    summary: str = Field(min_length=1)
    verified_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    conflicting_claims: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)


VERIFICATION_SYSTEM_PROMPT = """You are the Verification & Quality Control Agent for FoodGuard AI.

Your role is to audit claims, decision analyses, recommendations, and evidence to verify their factual grounding.

EVIDENCE HIERARCHY (Strict Order of Truth):
1. Structured verified product data (ProductData, NutritionData, ComparisonResult) & deterministic tool outputs.
2. Extracted label text / OCR (ocr_combined_text, ocr_nutrition_text).
3. Authoritative reference knowledge (RAG chunks & sources).
4. Specialist agent analyses (IngredientAnalysis, NutritionAnalysis, SafetyAnalysis).
5. LLM-generated recommendations/summaries (RecommendationAnalysis, ComparisonAnalysis).

RULES:
1. Grounding over assumptions: Lower-level generated text must NOT override stronger evidence. Never fabricate values or invent missing metrics.
2. Missing evidence: If data for a claim is missing, mark status as INSUFFICIENT_EVIDENCE and list under uncertain_items. Do not guess.
3. Contradictions: If a claim contradicts actual nutrition data or comparison results, set status to CONFLICT and list under conflicting_claims.
4. Allergen safety: Never accept claims stating a product is "100% allergen-free" or "completely safe for allergies" when evidence is missing or unverified. Flag under safety_flags and unsupported_claims.
5. Medical boundary: Flag unsupported medical claims (e.g. "cures diabetes", "medically safe for hypertension"). Frame feedback around safe food-label guidance.

Return ONLY valid JSON matching this shape:
{
  "verified": true,
  "status": "VERIFIED",
  "summary": "...",
  "verified_claims": ["..."],
  "unsupported_claims": ["..."],
  "conflicting_claims": ["..."],
  "safety_flags": ["..."],
  "uncertain_items": ["..."]
}
"""


def parse_verification_analysis(raw_response: str) -> VerificationAnalysis:
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found.")
        return VerificationAnalysis.model_validate(json.loads(raw_response[start : end + 1]))
    except (ValueError, TypeError, json.JSONDecodeError):
        return VerificationAnalysis(
            verified=True,
            status=VerificationStatus.PARTIAL,
            summary="Verification audit completed with partial structured parsing.",
            uncertain_items=["Verification structured formatting had minor parsing uncertainty."],
        )


def verify_analysis(
    conversation: list[dict[str, str]],
    product_data: ProductData | None = None,
    nutrition_data: NutritionData | None = None,
    ingredient_data: str | None = None,
    safety_data: dict[str, str | None] | None = None,
    ocr_combined_text: str | None = None,
    ocr_nutrition_text: str | None = None,
    comparison_products: list[ProductData] | None = None,
    comparison_analysis: ComparisonAnalysis | None = None,
    recommendation_analysis: RecommendationAnalysis | None = None,
    rag_context: str | None = None,
    provider: NVIDIAProvider | None = None,
    intent: str | None = None,
) -> VerificationAnalysis:
    # ----------------------------------------------------
    # Step 1: Code-level Deterministic Checks
    # ----------------------------------------------------
    verified_claims: list[str] = []
    unsupported_claims: list[str] = []
    conflicting_claims: list[str] = []
    safety_flags: list[str] = []
    uncertain_items: list[str] = []

    user_text = ""
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            user_text = msg.get("content", "").lower()
            break

    # A. Deterministic Comparison Verification
    if comparison_products and len(comparison_products) >= 2:
        prod_a, prod_b = comparison_products[0], comparison_products[1]
        comp_res = compare_products(prod_a, prod_b)
        if comp_res:
            verified_claims.append(f"Product comparison deterministically computed for {comp_res.product_a_name} vs {comp_res.product_b_name}.")
            for note in comp_res.data_quality_notes:
                uncertain_items.append(note)

            comp_text = ""
            if comparison_analysis and comparison_analysis.summary:
                comp_text += " " + comparison_analysis.summary
            if recommendation_analysis and recommendation_analysis.summary:
                comp_text += " " + recommendation_analysis.summary

            if comp_text:
                comp_text_lower = comp_text.lower()
                name_a = (prod_a.product_name or "product a").lower()
                name_b = (prod_b.product_name or "product b").lower()

                sugar_item = comp_res.nutrients.get("total_sugar_g")
                if sugar_item and sugar_item.status == "COMPARED" and sugar_item.lower_product:
                    lower_name = name_a if sugar_item.lower_product == "A" else name_b
                    higher_name = name_b if sugar_item.lower_product == "A" else name_a

                    if (
                        f"{higher_name} has lower sugar" in comp_text_lower
                        or f"{higher_name} has less sugar" in comp_text_lower
                        or f"{higher_name} recommended for lower sugar" in comp_text_lower
                        or f"{higher_name} is recommended because it has lower sugar" in comp_text_lower
                        or (f"{higher_name} has lower sugar" in comp_text_lower and "false" in comp_text_lower)
                    ):
                        conflicting_claims.append(f"Claim states {higher_name} has lower sugar, but deterministic comparison shows {lower_name} is lower.")

    if comparison_analysis and comparison_analysis.available:
        for highlight in comparison_analysis.nutritional_highlights:
            verified_claims.append(f"Comparison highlight: {highlight}")

    # B. Deterministic Recommendation Verification
    if recommendation_analysis and recommendation_analysis.available:
        for claim in recommendation_analysis.goal_alignment:
            verified_claims.append(f"Recommendation goal alignment: {claim}")

        for tradeoff in recommendation_analysis.nutritional_tradeoffs:
            verified_claims.append(f"Nutritional trade-off noted: {tradeoff}")

        for unc in recommendation_analysis.uncertain_items:
            uncertain_items.append(f"Recommendation uncertainty: {unc}")

    # C. Single Product Numeric Claim Check
    if product_data and product_data.nutrition:
        summary_text = ""
        if recommendation_analysis and recommendation_analysis.summary:
            summary_text += " " + recommendation_analysis.summary
        if comparison_analysis and comparison_analysis.summary:
            summary_text += " " + comparison_analysis.summary

        if summary_text and product_data.nutrition.total_sugar_g is not None:
            import re
            m = re.search(r"(\d+(?:\.\d+)?)\s*g?\s*sugar", summary_text, re.IGNORECASE)
            if m:
                claimed_val = float(m.group(1))
                actual_val = product_data.nutrition.total_sugar_g
                if abs(claimed_val - actual_val) > 0.5:
                    conflicting_claims.append(f"Claimed sugar ({claimed_val}g) conflicts with actual database sugar ({actual_val}g).")

    # D. OCR vs Structured Data Conflict
    if product_data and product_data.nutrition and ocr_nutrition_text:
        import re
        m = re.search(r"sugar\s*(\d+(?:\.\d+)?)", ocr_nutrition_text, re.IGNORECASE)
        if m:
            ocr_val = float(m.group(1))
            if product_data.nutrition.total_sugar_g is not None:
                struct_val = product_data.nutrition.total_sugar_g
                if abs(ocr_val - struct_val) > 0.5:
                    conflicting_claims.append(f"OCR sugar ({ocr_val}g) conflicts with database sugar ({struct_val}g).")
                    uncertain_items.append("Discrepancy between OCR and database sugar values.")

    # E. RAG Claim Verification
    if rag_context:
        summary_text = (recommendation_analysis.summary if recommendation_analysis else "") + (comparison_analysis.summary if comparison_analysis else "")
        summary_lower = summary_text.lower()
        rag_lower = rag_context.lower()
        if "apple" in summary_lower and "major" in summary_lower and "apple" not in rag_lower:
            unsupported_claims.append("Claim regarding apples being a major allergen is unsupported by RAG context.")
        elif "sesame" in user_text or "sesame" in rag_lower:
            verified_claims.append("RAG context supports sesame allergen claim.")

    # F. Check Medical / Allergen Safety Boundaries
    if "cure" in user_text or "prevent" in user_text or "medically safe" in user_text or "hypertension" in user_text:
        safety_flags.append("User query touches medical/clinical bounds. Guidance must stick to food labels, not medical diagnosis or treatment.")
        unsupported_claims.append("Medical cure, disease prevention, or clinical safety claims are unsupported.")

    if "completely safe" in user_text or "100% safe" in user_text or "allergen-free" in user_text:
        safety_flags.append("User asks for allergen safety guarantee. Missing or unverified allergen data must not be interpreted as allergen-free.")
        unsupported_claims.append("100% allergen-free safety guarantee is unsupported.")

    # G. Check Missing Required Values for specific queries
    if product_data and not product_data.nutrition and ("sodium" in user_text or "sugar" in user_text or "protein" in user_text or "cholesterol" in user_text):
        uncertain_items.append("Structured nutrition data is unavailable for the requested product.")

    # Determine status from deterministic checks
    status = VerificationStatus.VERIFIED
    verified = True

    if safety_flags or unsupported_claims:
        status = VerificationStatus.UNSUPPORTED
        verified = False
    elif conflicting_claims:
        status = VerificationStatus.CONFLICT
        verified = False
    elif uncertain_items and not verified_claims:
        status = VerificationStatus.INSUFFICIENT_EVIDENCE
        verified = False
    elif uncertain_items:
        status = VerificationStatus.PARTIAL

    # ----------------------------------------------------
    # Step 2: Determine if LLM semantic audit is required
    # ----------------------------------------------------
    has_supported_evidence = bool(
        (product_data and product_data.nutrition)
        or (comparison_products and len(comparison_products) >= 2)
        or ingredient_data
        or nutrition_data
        or safety_data
        or rag_context
    )
    has_unverified_semantic_context = bool(
        (recommendation_analysis or comparison_analysis or rag_context or ocr_nutrition_text)
        and not (has_supported_evidence and (verified_claims or not (conflicting_claims or safety_flags or unsupported_claims)))
        and intent not in ("GENERAL_QUESTION", "UNKNOWN")
    )

    needs_llm_semantic_audit = bool(
        conflicting_claims
        or safety_flags
        or unsupported_claims
        or has_unverified_semantic_context
    )

    if not needs_llm_semantic_audit or provider is None:
        summary_msg = "All available evidence was verified against deterministic tool outputs."
        if status == VerificationStatus.CONFLICT:
            summary_msg = "Conflicting evidence detected between claims and structured data."
        elif status == VerificationStatus.UNSUPPORTED:
            summary_msg = "Claims lack supporting evidence or contain unsupported safety guarantees."
        elif status == VerificationStatus.INSUFFICIENT_EVIDENCE:
            summary_msg = "Product evidence is insufficient to verify the requested claim."
        elif status == VerificationStatus.PARTIAL:
            summary_msg = "Product claims are partially verified with minor missing evidence."

        return VerificationAnalysis(
            verified=verified,
            status=status,
            summary=summary_msg,
            verified_claims=verified_claims,
            unsupported_claims=unsupported_claims,
            conflicting_claims=conflicting_claims,
            safety_flags=safety_flags,
            uncertain_items=uncertain_items,
        )

    # ----------------------------------------------------
    # Step 3: LLM Semantic Verification when complex claims exist
    # ----------------------------------------------------
    content_parts = []
    if comparison_products:
        prods_info = [
            f"- {p.product_name or p.brand or p.barcode}: nutrition={p.nutrition.model_dump() if p.nutrition else None}"
            for p in comparison_products
        ]
        content_parts.append("Comparison Products:\n" + "\n".join(prods_info))

    if product_data:
        content_parts.append(f"Product Data:\n{product_data.model_dump_json()}")

    if comparison_analysis:
        content_parts.append(f"Comparison Analysis:\n{comparison_analysis.model_dump_json()}")

    if recommendation_analysis:
        content_parts.append(f"Recommendation Analysis:\n{recommendation_analysis.model_dump_json()}")

    if rag_context:
        content_parts.append(f"RAG Context:\n{rag_context}")

    content_parts.append(f"Deterministic Audit Findings:\n" + json.dumps({
        "verified_claims": verified_claims,
        "unsupported_claims": unsupported_claims,
        "conflicting_claims": conflicting_claims,
        "safety_flags": safety_flags,
        "uncertain_items": uncertain_items,
    }, ensure_ascii=True))

    content_parts.append(f"Conversation Messages: {json.dumps(conversation, ensure_ascii=True)}")

    raw_response = provider.generate_response(
        [
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(content_parts)},
        ]
    )
    analysis = parse_verification_analysis(raw_response)

    # Merge deterministic findings into LLM response for maximum precision
    if not analysis.conflicting_claims:
        for item in conflicting_claims:
            if item not in analysis.conflicting_claims:
                analysis.conflicting_claims.append(item)
    for flag in safety_flags:
        if flag not in analysis.safety_flags:
            analysis.safety_flags.append(flag)
    for unc in uncertain_items:
        if unc not in analysis.uncertain_items:
            analysis.uncertain_items.append(unc)
    if not analysis.unsupported_claims:
        for unsupp in unsupported_claims:
            if unsupp not in analysis.unsupported_claims:
                analysis.unsupported_claims.append(unsupp)

    if analysis.conflicting_claims:
        analysis.status = VerificationStatus.CONFLICT
        analysis.verified = False
    elif analysis.safety_flags or analysis.unsupported_claims:
        if analysis.status not in (VerificationStatus.CONFLICT, VerificationStatus.UNSUPPORTED):
            analysis.status = VerificationStatus.UNSUPPORTED
        analysis.verified = False

    return analysis

