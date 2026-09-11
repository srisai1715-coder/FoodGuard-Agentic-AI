import json

from pydantic import BaseModel, Field

from app.llm.provider import NVIDIAProvider
from app.tools.nutrition import NutritionData, normalized_values


class NutritionAnalysis(BaseModel):
    available: bool = True
    normalized_values: dict[str, float] = Field(default_factory=dict)
    summary: str = Field(min_length=1)
    uncertain_items: list[str] = Field(default_factory=list)


NUTRITION_SYSTEM_PROMPT = """You are the Nutrition Analyst Agent for FoodGuard AI.

Analyze only the nutrition information supplied to you. Do not invent missing values.
Use the deterministic calculated values supplied in the input rather than performing important arithmetic yourself.
Explain nutrition information clearly for a normal consumer and distinguish label facts from interpretation.
Do not diagnose medical conditions, prescribe treatment, or claim that nutrition prevents or causes disease.
If required information is missing, clearly state that it is unavailable.

Return only valid JSON with this shape:
{"available":true,"normalized_values":{},"summary":"...","uncertain_items":[]}
"""

MISSING_NUTRITION_DATA_MESSAGE = (
    "I can analyze the nutrition, but I need the nutrition-label information first. "
    "Please provide the nutrition values or upload a readable nutrition label once image analysis is connected."
)


def missing_nutrition_analysis() -> NutritionAnalysis:
    return NutritionAnalysis(available=False, summary=MISSING_NUTRITION_DATA_MESSAGE)


def parse_nutrition_analysis(raw_response: str, calculated_values: dict[str, float]) -> NutritionAnalysis:
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found.")
        parsed = NutritionAnalysis.model_validate(json.loads(raw_response[start : end + 1]))
        parsed.normalized_values = calculated_values
        return parsed
    except (ValueError, TypeError, json.JSONDecodeError):
        return NutritionAnalysis(
            normalized_values=calculated_values,
            uncertain_items=["The nutrition analysis response could not be validated."],
            summary="I calculated the available values, but the explanatory analysis is uncertain.",
        )


def analyze_nutrition(
    data: NutritionData | None,
    conversation: list[dict[str, str]],
    provider: NVIDIAProvider | None = None,
    rag_context: str | None = None,
) -> NutritionAnalysis:
    if data is None:
        return missing_nutrition_analysis()

    calculated = normalized_values(data)
    llm_provider = provider or NVIDIAProvider()
    content_parts = [
        f"Supplied nutrition data: {data.model_dump_json()}",
        f"Deterministically calculated values: {json.dumps(calculated)}",
    ]
    if rag_context:
        content_parts.append(
            f"Authoritative dietary guidelines reference context (from knowledge base):\n{rag_context}"
        )
    content_parts.append(
        f"Relevant conversation context: {json.dumps(conversation, ensure_ascii=True)}"
    )

    raw_response = llm_provider.generate_response(
        [
            {"role": "system", "content": NUTRITION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\n".join(content_parts),
            },
        ]
    )
    return parse_nutrition_analysis(raw_response, calculated)