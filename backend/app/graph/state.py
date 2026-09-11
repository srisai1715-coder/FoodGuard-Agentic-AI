from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agents.manager import ManagerDecision
from app.agents.ingredient import IngredientAnalysis
from app.agents.nutrition import NutritionAnalysis
from app.agents.safety import SafetyAnalysis
from app.agents.comparison import ComparisonAnalysis
from app.agents.recommendation import RecommendationAnalysis
from app.agents.verification import VerificationAnalysis
from app.tools.nutrition import NutritionData
from app.tools.product_lookup import ProductData


class FoodGuardState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    manager_decision: ManagerDecision | None

    product_lookup_request: str | None
    product_data: ProductData | None
    product_lookup_error: str | None

    ocr_combined_text: str | None
    ocr_status: str | None
    ocr_warnings: list[str]
    ocr_nutrition_text: str | None

    # Individual OCR results for each uploaded image.
    #
    # Example:
    # image 1 -> ocr_results[0] -> Product 1
    # image 2 -> ocr_results[1] -> Product 2
    #
    # Keeping these results separately is required for
    # image-to-image product comparison.
    ocr_results: list[dict[str, Any]]

    ingredient_data: str | None
    ingredient_analysis: IngredientAnalysis | None

    nutrition_data: NutritionData | None
    nutrition_analysis: NutritionAnalysis | None

    safety_data: dict[str, str | None] | None
    safety_analysis: SafetyAnalysis | None

    rag_context: list[dict[str, Any]] | None
    rag_sources: list[str] | None

    comparison_barcodes: list[str] | None
    comparison_products: list[ProductData] | None
    comparison_analysis: ComparisonAnalysis | None

    recommendation_analysis: RecommendationAnalysis | None

    verification_analysis: VerificationAnalysis | None