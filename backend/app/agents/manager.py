import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.llm.provider import NVIDIAProvider


class Intent(str, Enum):
    GENERAL_QUESTION = "GENERAL_QUESTION"
    PRODUCT_ANALYSIS = "PRODUCT_ANALYSIS"
    INGREDIENT_ANALYSIS = "INGREDIENT_ANALYSIS"
    NUTRITION_ANALYSIS = "NUTRITION_ANALYSIS"
    SAFETY_ALLERGEN = "SAFETY_ALLERGEN"
    PRODUCT_COMPARISON = "PRODUCT_COMPARISON"
    RECOMMENDATION = "RECOMMENDATION"
    UNKNOWN = "UNKNOWN"


class ManagerDecision(BaseModel):
    intent: Intent = Intent.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = "Unable to validate the manager decision."


MANAGER_SYSTEM_PROMPT = """You are the Manager Agent for FoodGuard AI.

Your job is to understand the user's request and classify it into exactly one supported intent.
You are an orchestrator, not a specialist analyst. Do not invent product information.

Classification Guidance:
- GENERAL_QUESTION: Use for general food safety rules, educational questions, food regulations, dietary guidelines, additive explanations, or general allergen inquiries (e.g., "Why is sesame an allergen?", "What are the Big 9?", "How much sodium per day?") when the user is NOT asking to analyze a specific product or label.
- RECOMMENDATION: Use when the user asks for product recommendation, choice advice, suitability assessment for a specific diet/goal/preference, or healthier alternative advice (e.g., "Which product is better?", "Which one should I choose?", "Which has less sugar?", "Which is better if I want lower sugar/sodium/more protein?", "Is this a good option based on my preference?", "Can you recommend one of these?").
- PRODUCT_ANALYSIS / INGREDIENT_ANALYSIS / NUTRITION_ANALYSIS / SAFETY_ALLERGEN: Use ONLY when the user is asking purely factual questions to analyze a specific product, specific ingredient list, uploaded label image, or barcode without asking for a choice/recommendation.
- PRODUCT_COMPARISON: Use when the user explicitly asks to compare two products without asking for a single recommended choice.

Supported intents:
GENERAL_QUESTION
PRODUCT_ANALYSIS
INGREDIENT_ANALYSIS
NUTRITION_ANALYSIS
SAFETY_ALLERGEN
PRODUCT_COMPARISON
RECOMMENDATION
UNKNOWN

Return only a JSON object with exactly these fields:
{"intent":"...","confidence":0.0,"reason":"..."}
"""


def _message_payload(messages: list[dict[str, str]]) -> str:
    return json.dumps(messages, ensure_ascii=True)


def _unknown_decision(reason: str) -> ManagerDecision:
    return ManagerDecision(intent=Intent.UNKNOWN, confidence=0.0, reason=reason)


def parse_manager_decision(raw_response: str) -> ManagerDecision:
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found.")
        return ManagerDecision.model_validate(json.loads(raw_response[start : end + 1]))
    except (ValueError, TypeError, json.JSONDecodeError):
        return _unknown_decision("The manager returned an invalid structured decision.")


def classify_intent(
    messages: list[dict[str, str]],
    provider: NVIDIAProvider | None = None,
) -> ManagerDecision:
    llm_provider = provider or NVIDIAProvider()
    raw_response = llm_provider.generate_response(
        [
            {"role": "system", "content": MANAGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify this conversation. Use the latest user request and recent context. "
                    f"Conversation messages: {_message_payload(messages)}"
                ),
            },
        ]
    )
    return parse_manager_decision(raw_response)


def decision_to_dict(decision: ManagerDecision) -> dict[str, Any]:
    return decision.model_dump(mode="json")