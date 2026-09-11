from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from app.agents.manager import Intent
from app.graph.nodes import (
    comparison_node,
    ingredient_node,
    manager_node,
    nutrition_node,
    product_lookup_node,
    rag_node,
    recommendation_node,
    response_node,
    safety_node,
    verification_node,
)
from app.graph.state import FoodGuardState
from app.llm.provider import NVIDIAProvider


def _route_after_rag(state: FoodGuardState) -> str:
    """
    Determine which specialist should handle the request.

    Deterministic image-comparison rule:
    If two or more individual OCR results are available and the user
    explicitly asks to compare/comparison/side-by-side, route directly
    to the Comparison Agent.

    This prevents the Manager LLM from incorrectly choosing
    RECOMMENDATION for prompts such as:

        "Compare these two products and tell me which is better
        for reducing sugar."

    A pure recommendation request without an explicit comparison
    instruction can still follow the Manager's normal routing.
    """

    messages = state.get("messages") or []

    user_query = ""

    for message in reversed(messages):
        if message.type == "human":
            user_query = str(message.content)
            break

    normalized_query = user_query.lower()

    comparison_keywords = (
        "compare",
        "comparison",
        "side-by-side",
        "side by side",
    )

    explicit_comparison_request = any(
        keyword in normalized_query
        for keyword in comparison_keywords
    )

    ocr_results = state.get("ocr_results") or []

    if (
        len(ocr_results) >= 2
        and explicit_comparison_request
    ):
        return "comparison_node"

    decision = state.get("manager_decision")

    if not decision:
        return "verification_node"

    intent = decision.intent

    if intent == Intent.RECOMMENDATION:
        return "recommendation_node"

    if intent == Intent.PRODUCT_COMPARISON:
        return "comparison_node"

    if intent == Intent.INGREDIENT_ANALYSIS:
        return "ingredient_node"

    if intent == Intent.NUTRITION_ANALYSIS:
        return "nutrition_node"

    if intent == Intent.SAFETY_ALLERGEN:
        return "safety_node"

    if (
        intent == Intent.PRODUCT_ANALYSIS
        and state.get("product_lookup_request")
    ):
        return "product_lookup_node"

    return "verification_node"


def build_workflow(
    provider: NVIDIAProvider | None = None,
) -> Callable:
    graph = StateGraph(FoodGuardState)

    if provider is None:
        graph.add_node("manager_node", manager_node)
        graph.add_node("rag_node", rag_node)
        graph.add_node("comparison_node", comparison_node)
        graph.add_node("recommendation_node", recommendation_node)
        graph.add_node("ingredient_node", ingredient_node)
        graph.add_node("nutrition_node", nutrition_node)
        graph.add_node("safety_node", safety_node)
        graph.add_node("product_lookup_node", product_lookup_node)
        graph.add_node("verification_node", verification_node)
        graph.add_node("response_node", response_node)

    else:
        graph.add_node(
            "manager_node",
            lambda state: manager_node(
                state,
                provider=provider,
            ),
        )

        graph.add_node(
            "rag_node",
            rag_node,
        )

        graph.add_node(
            "comparison_node",
            lambda state: comparison_node(
                state,
                provider=provider,
            ),
        )

        graph.add_node(
            "recommendation_node",
            lambda state: recommendation_node(
                state,
                provider=provider,
            ),
        )

        graph.add_node(
            "ingredient_node",
            lambda state: ingredient_node(
                state,
                provider=provider,
            ),
        )

        graph.add_node(
            "nutrition_node",
            lambda state: nutrition_node(
                state,
                provider=provider,
            ),
        )

        graph.add_node(
            "safety_node",
            lambda state: safety_node(
                state,
                provider=provider,
            ),
        )

        graph.add_node(
            "product_lookup_node",
            product_lookup_node,
        )

        graph.add_node(
            "verification_node",
            lambda state: verification_node(
                state,
                provider=provider,
            ),
        )

        graph.add_node(
            "response_node",
            lambda state: response_node(
                state,
                provider=provider,
            ),
        )

    graph.add_edge(
        START,
        "manager_node",
    )

    graph.add_edge(
        "manager_node",
        "rag_node",
    )

    graph.add_conditional_edges(
        "rag_node",
        _route_after_rag,
        {
            "recommendation_node": "recommendation_node",
            "comparison_node": "comparison_node",
            "ingredient_node": "ingredient_node",
            "nutrition_node": "nutrition_node",
            "safety_node": "safety_node",
            "product_lookup_node": "product_lookup_node",
            "verification_node": "verification_node",
        },
    )

    graph.add_edge(
        "recommendation_node",
        "verification_node",
    )

    graph.add_edge(
        "comparison_node",
        "verification_node",
    )

    graph.add_edge(
        "ingredient_node",
        "verification_node",
    )

    graph.add_edge(
        "nutrition_node",
        "verification_node",
    )

    graph.add_edge(
        "safety_node",
        "verification_node",
    )

    graph.add_edge(
        "product_lookup_node",
        "verification_node",
    )

    graph.add_edge(
        "verification_node",
        "response_node",
    )

    graph.add_edge(
        "response_node",
        END,
    )

    return graph.compile()


workflow = build_workflow()