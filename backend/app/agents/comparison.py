import json
from pydantic import BaseModel, Field

from app.llm.provider import NVIDIAProvider
from app.tools.comparison import ComparisonResult


class ComparisonAnalysis(BaseModel):
    available: bool = True
    summary: str = Field(min_length=1)
    nutritional_highlights: list[str] = Field(default_factory=list)
    ingredient_highlights: list[str] = Field(default_factory=list)
    allergen_highlights: list[str] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)


COMPARISON_SYSTEM_PROMPT = """You are the Product Comparison Specialist Agent for FoodGuard AI.

Analyze only the structured comparison results supplied to you. Do not invent missing values or calculate numbers yourself.
Highlight significant nutritional contrasts (lower sodium, lower sugar, higher protein, lower saturated fat).
Mention serving size differences clearly when serving sizes do not match.
Contrast ingredient and allergen differences objectively.
Do not claim one product is medically superior or diagnose health conditions.

Return only valid JSON with this shape:
{
  "available": true,
  "summary": "...",
  "nutritional_highlights": ["..."],
  "ingredient_highlights": ["..."],
  "allergen_highlights": ["..."],
  "uncertain_items": []
}
"""


def parse_comparison_analysis(raw_response: str) -> ComparisonAnalysis:
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found.")
        return ComparisonAnalysis.model_validate(json.loads(raw_response[start : end + 1]))
    except (ValueError, TypeError, json.JSONDecodeError):
        return ComparisonAnalysis(
            available=True,
            summary="Product comparison was completed deterministically, but explanatory analysis formatting was uncertain.",
            uncertain_items=["The structured comparison explanation could not be validated."],
        )


def analyze_comparison(
    result: ComparisonResult,
    conversation: list[dict[str, str]],
    provider: NVIDIAProvider | None = None,
    rag_context: str | None = None,
) -> ComparisonAnalysis:
    llm_provider = provider or NVIDIAProvider()
    content_parts = [
        f"Deterministic product comparison result:\n{result.model_dump_json()}"
    ]

    if rag_context:
        content_parts.append(
            f"Authoritative reference guidelines (from knowledge base):\n{rag_context}"
        )

    content_parts.append(
        f"Relevant conversation context: {json.dumps(conversation, ensure_ascii=True)}"
    )

    raw_response = llm_provider.generate_response(
        [
            {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\n".join(content_parts),
            },
        ]
    )
    return parse_comparison_analysis(raw_response)
