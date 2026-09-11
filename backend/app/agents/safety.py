import json
from enum import Enum

from pydantic import BaseModel, Field

from app.llm.provider import NVIDIAProvider
from app.tools.safety import (
    AllergenStatus,
    classify_allergen_status,
    detect_cross_contact_statement,
)


class OverallSafetyStatus(str, Enum):
    CLEAR_FROM_SUPPLIED_INFO = "CLEAR_FROM_SUPPLIED_INFO"
    ALLERGEN_CONFIRMED = "ALLERGEN_CONFIRMED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class AllergenFinding(BaseModel):
    name: str = Field(min_length=1)
    status: AllergenStatus
    evidence: str = Field(min_length=1)


class SafetyAnalysis(BaseModel):
    available: bool = True
    allergens: list[AllergenFinding] = Field(default_factory=list)
    cross_contact_warnings: list[str] = Field(default_factory=list)
    unknown_items: list[str] = Field(default_factory=list)
    overall_status: OverallSafetyStatus = OverallSafetyStatus.REVIEW_REQUIRED
    summary: str = Field(min_length=1)


SAFETY_SYSTEM_PROMPT = """You are the Safety and Allergen Analyst Agent for FoodGuard AI.

Analyze only the product and ingredient information supplied to you. Identify allergens explicitly supported by that information.
Distinguish CONFIRMED, NOT_FOUND, and UNKNOWN. Never treat missing evidence as proof that an allergen is absent.
Identify cross-contact warnings only when supported by the supplied label information.
Do not diagnose medical conditions, prescribe treatment, or guarantee that a product is medically safe.
When information is incomplete, clearly state that additional label information is required.

Return only valid JSON with allergens, cross_contact_warnings, unknown_items, overall_status, and summary.
"""

MISSING_SAFETY_DATA_MESSAGE = (
    "I can check the product for common allergens, but I need the ingredient list and/or "
    "allergen statement first. Without that information, the allergen status is unknown."
)


def missing_safety_analysis() -> SafetyAnalysis:
    return SafetyAnalysis(
        available=False,
        overall_status=OverallSafetyStatus.INSUFFICIENT_INFORMATION,
        unknown_items=["Ingredient and allergen information was not supplied."],
        summary=MISSING_SAFETY_DATA_MESSAGE,
    )


def _overall_status(findings: list[AllergenFinding], warnings: list[str]) -> OverallSafetyStatus:
    if any(item.status == AllergenStatus.CONFIRMED for item in findings):
        return OverallSafetyStatus.ALLERGEN_CONFIRMED
    if warnings or any(item.status == AllergenStatus.UNKNOWN for item in findings):
        return OverallSafetyStatus.REVIEW_REQUIRED
    return OverallSafetyStatus.CLEAR_FROM_SUPPLIED_INFO


def _deterministic_analysis(ingredient_text: str | None, allergen_statement: str | None) -> SafetyAnalysis:
    raw_findings = classify_allergen_status(ingredient_text, allergen_statement)
    findings = [AllergenFinding(name=name, **value) for name, value in raw_findings.items()]
    warnings = detect_cross_contact_statement(ingredient_text, allergen_statement)
    return SafetyAnalysis(
        allergens=findings,
        cross_contact_warnings=warnings,
        overall_status=_overall_status(findings, warnings),
        summary="The supplied label information was checked for explicit allergen evidence.",
    )


def parse_safety_analysis(raw_response: str, deterministic: SafetyAnalysis) -> SafetyAnalysis:
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found.")
        parsed = SafetyAnalysis.model_validate(json.loads(raw_response[start : end + 1]))
        parsed.allergens = deterministic.allergens
        parsed.cross_contact_warnings = deterministic.cross_contact_warnings
        parsed.overall_status = deterministic.overall_status
        return parsed
    except (ValueError, TypeError, json.JSONDecodeError):
        deterministic = deterministic.model_copy(deep=True)
        deterministic.unknown_items.append("The safety explanation could not be validated.")
        deterministic.overall_status = OverallSafetyStatus.REVIEW_REQUIRED
        deterministic.summary = "The explicit label findings are available, but the safety explanation is uncertain."
        return deterministic


def analyze_safety(
    ingredient_text: str | None,
    allergen_statement: str | None,
    cross_contact_statement: str | None,
    conversation: list[dict[str, str]],
    provider: NVIDIAProvider | None = None,
    rag_context: str | None = None,
) -> SafetyAnalysis:
    if not ingredient_text and not allergen_statement and not cross_contact_statement:
        return missing_safety_analysis()

    deterministic = _deterministic_analysis(
        ingredient_text,
        " ".join(value for value in (allergen_statement, cross_contact_statement) if value),
    )
    llm_provider = provider or NVIDIAProvider()
    content_parts = [
        f"Ingredient information: {ingredient_text or '[not supplied]'}",
        f"Allergen statement: {allergen_statement or '[not supplied]'}",
        f"Cross-contact statement: {cross_contact_statement or '[not supplied]'}",
        f"Deterministic findings: {deterministic.model_dump_json()}",
    ]
    if rag_context:
        content_parts.append(
            f"Authoritative FDA food allergen & safety reference context (from knowledge base):\n{rag_context}"
        )
    content_parts.append(
        f"Conversation context: {json.dumps(conversation, ensure_ascii=True)}"
    )

    raw_response = llm_provider.generate_response(
        [
            {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\n".join(content_parts),
            },
        ]
    )
    return parse_safety_analysis(raw_response, deterministic)