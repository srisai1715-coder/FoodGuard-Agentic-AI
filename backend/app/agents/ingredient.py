import json

from pydantic import BaseModel, Field

from app.llm.provider import NVIDIAProvider


class IngredientItem(BaseModel):
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    attention: bool = False


class IngredientAnalysis(BaseModel):
    available: bool = True
    ingredients: list[IngredientItem] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)


INGREDIENT_SYSTEM_PROMPT = """You are the Ingredient Analyst Agent for FoodGuard AI.

Analyze only the ingredient information supplied to you. Do not invent ingredients.
For each supplied ingredient, explain its general identity and functional role.
Identify common allergens only when supported by the supplied information.
If an ingredient is unfamiliar or ambiguous, mark it as uncertain instead of guessing.
Do not diagnose medical conditions or prescribe treatment.
Separate factual ingredient information from interpretation and use simple consumer language.

Return only valid JSON with this shape:
{
  "available": true,
  "ingredients": [
    {"name": "...", "role": "...", "explanation": "...", "attention": false}
  ],
  "allergens": [],
  "uncertain_items": [],
  "summary": "..."
}
"""


MISSING_INGREDIENT_DATA_MESSAGE = (
    "I can analyze the ingredients, but I need the ingredient list first. "
    "Please provide the ingredient text or upload a readable label once image analysis is available."
)


def missing_ingredient_analysis() -> IngredientAnalysis:
    return IngredientAnalysis(
        available=False,
        summary=MISSING_INGREDIENT_DATA_MESSAGE,
    )


def parse_ingredient_analysis(raw_response: str) -> IngredientAnalysis:
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found.")
        return IngredientAnalysis.model_validate(json.loads(raw_response[start : end + 1]))
    except (ValueError, TypeError, json.JSONDecodeError):
        return IngredientAnalysis(
            available=True,
            uncertain_items=["The ingredient analysis response could not be validated."],
            summary="I could not validate the ingredient analysis, so the result is uncertain.",
        )


def analyze_ingredients(
    ingredient_text: str | None,
    conversation: list[dict[str, str]],
    provider: NVIDIAProvider | None = None,
    rag_context: str | None = None,
) -> IngredientAnalysis:
    if not ingredient_text or not ingredient_text.strip():
        return missing_ingredient_analysis()

    llm_provider = provider or NVIDIAProvider()
    content_parts = [
        f"Ingredient information supplied by the user: {ingredient_text.strip()}"
    ]
    if rag_context:
        content_parts.append(
            f"Authoritative food & additive reference context (from knowledge base):\n{rag_context}"
        )
    content_parts.append(
        f"Relevant conversation context: {json.dumps(conversation, ensure_ascii=True)}"
    )

    raw_response = llm_provider.generate_response(
        [
            {"role": "system", "content": INGREDIENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\n".join(content_parts),
            },
        ]
    )
    return parse_ingredient_analysis(raw_response)