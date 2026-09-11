import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage

from app.agents.comparison import ComparisonAnalysis, analyze_comparison
from app.agents.ingredient import IngredientAnalysis, analyze_ingredients
from app.agents.manager import ManagerDecision, classify_intent, decision_to_dict
from app.agents.nutrition import NutritionAnalysis, analyze_nutrition
from app.agents.recommendation import RecommendationAnalysis, analyze_recommendation
from app.agents.safety import SafetyAnalysis, analyze_safety
from app.agents.verification import VerificationAnalysis, verify_analysis
from app.graph.state import FoodGuardState
from app.llm.provider import NVIDIAProvider
from app.rag.retriever import Retriever
from app.tools.comparison import compare_products
from app.tools.nutrition import NutritionData
from app.tools.product_lookup import ProductData, ProductLookupError, lookup_product_by_barcode


def _provider_messages(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    roles = {"human": "user", "ai": "assistant", "system": "system"}
    return [
        {"role": roles[message.type], "content": str(message.content)}
        for message in messages
    ]


def manager_node(
    state: FoodGuardState,
    provider: NVIDIAProvider | None = None,
) -> FoodGuardState:
    return {
        "manager_decision": classify_intent(
            _provider_messages(state["messages"]),
            provider=provider,
        )
    }


def rag_node(
    state: FoodGuardState,
    retriever: Retriever | None = None,
) -> FoodGuardState:
    messages = state.get("messages") or []
    user_query = ""

    for msg in reversed(messages):
        if msg.type == "human":
            user_query = str(msg.content)
            break

    if not user_query:
        return {"rag_context": [], "rag_sources": []}

    rag_retriever = retriever or Retriever()
    chunks = rag_retriever.retrieve(user_query)
    sources = rag_retriever.extract_sources(chunks)

    return {
        "rag_context": chunks,
        "rag_sources": sources,
    }


def _allergens_from_ocr_text(text: str | None) -> list[str]:
    if not text:
        return []

    normalized = text.lower()
    patterns = {
        "peanuts": r"\bpeanuts?\b",
        "milk": r"\b(?:milk|dairy|whey|casein)\b",
        "eggs": r"\beggs?\b",
        "soybeans": r"\b(?:soy|soybeans?|soya|soy lecithin|tofu)\b",
        "wheat": r"\b(?:wheat|flour|gluten)\b",
        "sesame": r"\b(?:sesame|tahini)\b",
        "fish": r"\b(?:fish|salmon|tuna|cod|flounder|tilapia|anchov(?:y|ies))\b",
        "crustacean shellfish": r"\b(?:shellfish|crustacean|shrimp|prawn(?:s)?|crab|lobster)\b",
        "tree nuts": r"\b(?:tree[- ]nuts?|almonds?|cashews?|walnuts?|pecans?|pistachios?|hazelnuts?|macadamia|brazil[- ]nuts?)\b",
    }

    found: list[str] = []
    for allergen, pattern in patterns.items():
        if re.search(pattern, normalized):
            found.append(allergen)

    return found


def _extract_number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _extract_ocr_nutrition_data(text: str | None) -> NutritionData | None:
    """Extract common Nutrition Facts values directly from OCR text.

    This is intentionally conservative and only extracts values when the
    nutrient label is explicitly present in the OCR text. It is a fallback
    for photographs where the OCR section detector did not isolate the
    nutrition panel.
    """
    if not text:
        return None

    normalized = str(text).replace("\u00a0", " ")

    serving_size_g = _extract_number(
        normalized,
        r"serving\s+size[^\n]{0,80}?(\d+(?:\.\d+)?)\s*g\b",
    )

    servings_per_container = _extract_number(
        normalized,
        r"(?:about\s+)?(\d+(?:\.\d+)?)\s+servings?\s+per\s+container",
    )

    calories = _extract_number(
        normalized,
        r"calories(?:\s+per\s+serving)?[^\d]{0,30}(\d+(?:\.\d+)?)",
    )

    fat_g = _extract_number(
        normalized,
        r"(?:total\s+fat|fat)\s*(?:\([^)]*\))?[^\d]{0,20}(\d+(?:\.\d+)?)\s*g\b",
    )

    saturated_fat_g = _extract_number(
        normalized,
        r"(?:saturated\s+fat|sat\.?\s*fat)[^\d]{0,20}(\d+(?:\.\d+)?)\s*g\b",
    )

    trans_fat_g = _extract_number(
        normalized,
        r"trans\s+fat[^\d]{0,20}(\d+(?:\.\d+)?)\s*g\b",
    )

    cholesterol_mg = _extract_number(
        normalized,
        r"cholesterol[^\d]{0,20}(\d+(?:\.\d+)?)\s*mg\b",
    )

    sodium_mg = _extract_number(
        normalized,
        r"sodium[^\d]{0,20}(\d+(?:\.\d+)?)\s*mg\b",
    )

    carbohydrates_g = _extract_number(
        normalized,
        r"(?:total\s+carbohydrate|total\s+carb\.?|carbohydrate|carb\.?)\s*[^\d]{0,20}(\d+(?:\.\d+)?)\s*g\b",
    )

    fiber_g = _extract_number(
        normalized,
        r"(?:dietary\s+fiber|fiber)\s*[^\d]{0,20}(\d+(?:\.\d+)?)\s*g\b",
    )

    total_sugar_g = _extract_number(
        normalized,
        r"total\s+sugars?\s*[^\d]{0,20}(\d+(?:\.\d+)?)\s*g\b",
    )

    added_sugar_g = _extract_number(
        normalized,
        r"(?:includes?|including)\s+(\d+(?:\.\d+)?)\s*g\s+added\s+sugars?",
    )

    protein_g = _extract_number(
        normalized,
        r"protein\s*[^\d]{0,20}(\d+(?:\.\d+)?)\s*g\b",
    )

    values = [
        serving_size_g,
        calories,
        fat_g,
        saturated_fat_g,
        sodium_mg,
        carbohydrates_g,
        total_sugar_g,
        protein_g,
    ]

    if not any(value is not None for value in values):
        return None

    return NutritionData(
        serving_size_g=serving_size_g,
        servings_per_container=servings_per_container,
        calories=calories,
        protein_g=protein_g,
        carbohydrates_g=carbohydrates_g,
        total_sugar_g=total_sugar_g,
        added_sugar_g=added_sugar_g,
        fat_g=fat_g,
        saturated_fat_g=saturated_fat_g,
        trans_fat_g=trans_fat_g,
        fiber_g=fiber_g,
        sodium_mg=sodium_mg,
        cholesterol_mg=cholesterol_mg,
    )


def _ocr_result_to_product(
    ocr_result: dict,
    product_number: int,
) -> ProductData:
    """Convert one uploaded image's OCR result into ProductData.

    OCR section parsing can miss a Nutrition Facts heading on a real-world
    label. For comparison, fall back to parsing the complete OCR text so
    that plainly labelled nutrition values are still preserved.
    """
    sections = ocr_result.get("sections") or {}
    full_text = str(ocr_result.get("text") or "")

    nutrition_payload = sections.get("nutrition_data")

    nutrition_data = None
    if isinstance(nutrition_payload, dict):
        try:
            nutrition_data = NutritionData.model_validate(
                nutrition_payload
            )
        except (TypeError, ValueError):
            nutrition_data = None

    # Fallback: parse the complete OCR text when section extraction did not
    # identify a Nutrition Facts block. This is especially useful for labels
    # whose OCR text does not contain the exact heading "Nutrition Facts".
    if nutrition_data is None:
        nutrition_data = _extract_ocr_nutrition_data(
            sections.get("nutrition_text") or full_text
        )

    ingredients_text = sections.get("ingredients_text")
    allergen_statement = sections.get("allergen_text")

    # Fallback extraction for labels where section parsing did not isolate
    # the ingredients/contains lines. Keep the extracted text bounded so the
    # entire nutrition panel is not incorrectly treated as ingredients.
    if not ingredients_text and full_text:
        ingredient_match = re.search(
            r"\bingredients?\s*[:\-]?\s*(.+?)(?=\b(?:contains|allergens?|nutrition(?: facts| information)?|serving size)\b|$)",
            full_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if ingredient_match:
            ingredients_text = ingredient_match.group(1).strip(" \n.;:") or None

    if not allergen_statement and full_text:
        allergen_match = re.search(
            r"\b(?:contains|allergens?)\s*[:\-]?\s*(.+?)(?=\b(?:may contain|nutrition(?: facts| information)?|serving size)\b|$)",
            full_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if allergen_match:
            allergen_statement = (
                allergen_match.group(0).strip(" \n.;:")
                or None
            )

    source_text = "\n".join(
        str(value)
        for value in (
            ingredients_text,
            allergen_statement,
            sections.get("other_text"),
            full_text,
        )
        if value
    )

    allergens = _allergens_from_ocr_text(
        "\n".join(
            str(value)
            for value in (
                ingredients_text,
                allergen_statement,
            )
            if value
        )
    )

    may_contain: list[str] = []
    if allergen_statement and re.search(
        r"\bmay contain\b|\btraces?\b|\bprocessed in\b|\bfacility\b",
        str(allergen_statement),
        flags=re.IGNORECASE,
    ):
        may_contain = _allergens_from_ocr_text(
            str(allergen_statement)
        )

    filename = str(
        ocr_result.get("filename")
        or f"uploaded-image-{product_number}"
    )

    product_name = f"Uploaded Product {product_number}"

    return ProductData(
        barcode=f"OCR-UPLOAD-{product_number}",
        product_name=product_name,
        brand=None,
        ingredients_text=ingredients_text,
        allergen_statement=allergen_statement,
        allergens=allergens,
        traces=may_contain,
        nutrition=nutrition_data,
        image_url=None,
        categories=[],
        labels=[],
        source="ocr_upload",
        source_url=f"ocr://{filename}",
        data_quality="PARTIAL",
    )


def comparison_node(
    state: FoodGuardState,
    provider: NVIDIAProvider | None = None,
) -> FoodGuardState:
    barcodes: list[str] = []

    messages = state.get("messages") or []

    # ---------------------------------------------------------------
    # Priority 1: explicit barcodes in the latest user message.
    # ---------------------------------------------------------------
    if messages:
        last_msg = messages[-1]

        if last_msg.type == "human":
            matches = re.findall(
                r"\b\d{8,14}\b",
                str(last_msg.content),
            )

            if len(matches) >= 2:
                barcodes = matches[:2]

    # ---------------------------------------------------------------
    # Priority 2: comparison barcodes already stored in state.
    # ---------------------------------------------------------------
    if not barcodes:
        state_barcodes = list(
            state.get("comparison_barcodes") or []
        )

        if len(state_barcodes) >= 2:
            barcodes = state_barcodes[:2]

    # ---------------------------------------------------------------
    # Priority 3: recover two barcodes from conversation history.
    # ---------------------------------------------------------------
    if len(barcodes) < 2:
        for msg in reversed(messages):
            if msg.type == "human":
                matches = re.findall(
                    r"\b\d{8,14}\b",
                    str(msg.content),
                )

                for match in matches:
                    if match not in barcodes:
                        barcodes.append(match)

                if len(barcodes) >= 2:
                    break

    # ---------------------------------------------------------------
    # IMAGE-TO-IMAGE COMPARISON
    # ---------------------------------------------------------------
    # When two or more images were uploaded, main.py preserves the
    # individual OCR results in state["ocr_results"].
    #
    # Do NOT use the combined OCR text here because that loses the
    # boundary between Product A and Product B.
    # ---------------------------------------------------------------
    ocr_results = list(state.get("ocr_results") or [])

    if len(barcodes) < 2 and len(ocr_results) >= 2:
        product_a = _ocr_result_to_product(
            ocr_results[0],
            1,
        )
        product_b = _ocr_result_to_product(
            ocr_results[1],
            2,
        )

        comp_result = compare_products(
            product_a,
            product_b,
        )

        rag_chunks = state.get("rag_context") or []
        rag_str = (
            Retriever().format_rag_context(rag_chunks)
            if rag_chunks
            else None
        )

        analysis = analyze_comparison(
            comp_result,
            _provider_messages(state["messages"]),
            provider=provider,
            rag_context=rag_str,
        )

        return {
            "comparison_barcodes": [
                product_a.barcode,
                product_b.barcode,
            ],
            "comparison_products": [
                product_a,
                product_b,
            ],
            "comparison_analysis": analysis,
        }

    # ---------------------------------------------------------------
    # Existing barcode-to-barcode comparison path.
    # ---------------------------------------------------------------
    if len(barcodes) < 2:
        return {
            "comparison_products": [],
            "comparison_analysis": ComparisonAnalysis(
                available=False,
                summary=(
                    "To perform a comparison, please provide two product "
                    "barcodes or upload product labels for both products."
                ),
            ),
        }

    prod_a_res = lookup_product_by_barcode(
        barcodes[0]
    )
    prod_b_res = lookup_product_by_barcode(
        barcodes[1]
    )

    if (
        not prod_a_res.found
        or not prod_a_res.product
        or not prod_b_res.found
        or not prod_b_res.product
    ):
        missing = []

        if not prod_a_res.found or not prod_a_res.product:
            missing.append(barcodes[0])

        if not prod_b_res.found or not prod_b_res.product:
            missing.append(barcodes[1])

        return {
            "comparison_products": [],
            "comparison_analysis": ComparisonAnalysis(
                available=False,
                summary=(
                    f"I couldn't retrieve product data for barcode(s): "
                    f"{', '.join(missing)} from the product database."
                ),
            ),
        }

    prod_a = prod_a_res.product
    prod_b = prod_b_res.product

    comp_result = compare_products(
        prod_a,
        prod_b,
    )

    rag_chunks = state.get("rag_context") or []
    rag_str = (
        Retriever().format_rag_context(rag_chunks)
        if rag_chunks
        else None
    )

    analysis = analyze_comparison(
        comp_result,
        _provider_messages(state["messages"]),
        provider=provider,
        rag_context=rag_str,
    )

    return {
        "comparison_barcodes": barcodes,
        "comparison_products": [prod_a, prod_b],
        "comparison_analysis": analysis,
    }


def recommendation_node(
    state: FoodGuardState,
    provider: NVIDIAProvider | None = None,
) -> FoodGuardState:
    barcodes: list[str] = []

    messages = state.get("messages") or []

    if messages:
        last_msg = messages[-1]

        if last_msg.type == "human":
            matches = re.findall(
                r"\b\d{8,14}\b",
                str(last_msg.content),
            )

            if len(matches) >= 2:
                barcodes = matches[:2]

    if not barcodes:
        state_barcodes = list(state.get("comparison_barcodes") or [])

        if len(state_barcodes) >= 2:
            barcodes = state_barcodes[:2]

    if len(barcodes) < 2 and state.get("product_lookup_request"):
        barcodes = [state["product_lookup_request"]]

    if len(barcodes) < 2:
        for msg in reversed(messages):
            if msg.type == "human":
                matches = re.findall(
                    r"\b\d{8,14}\b",
                    str(msg.content),
                )

                for m in matches:
                    if m not in barcodes:
                        barcodes.append(m)

    comp_products = state.get("comparison_products")
    comp_analysis = state.get("comparison_analysis")

    if len(barcodes) >= 2 and not comp_products:
        prod_a_res = lookup_product_by_barcode(barcodes[0])
        prod_b_res = lookup_product_by_barcode(barcodes[1])

        if (
            prod_a_res.found
            and prod_a_res.product
            and prod_b_res.found
            and prod_b_res.product
        ):
            prod_a = prod_a_res.product
            prod_b = prod_b_res.product

            comp_products = [prod_a, prod_b]
            comp_result = compare_products(prod_a, prod_b)

            rag_chunks = state.get("rag_context") or []
            rag_str = (
                Retriever().format_rag_context(rag_chunks)
                if rag_chunks
                else None
            )

            comp_analysis = analyze_comparison(
                comp_result,
                _provider_messages(state["messages"]),
                provider=provider,
                rag_context=rag_str,
            )

    product_data = state.get("product_data")

    if not product_data and state.get("product_lookup_request"):
        res = lookup_product_by_barcode(
            state["product_lookup_request"]
        )

        if res.found and res.product:
            product_data = res.product

    rag_chunks = state.get("rag_context") or []
    rag_str = (
        Retriever().format_rag_context(rag_chunks)
        if rag_chunks
        else None
    )

    rec_analysis = analyze_recommendation(
        conversation=_provider_messages(state["messages"]),
        product_data=product_data,
        nutrition_data=(
            state.get("nutrition_data")
            or (product_data.nutrition if product_data else None)
        ),
        ingredient_data=(
            state.get("ingredient_data")
            or (
                product_data.ingredients_text
                if product_data
                else None
            )
        ),
        safety_data=state.get("safety_data"),
        ocr_combined_text=state.get("ocr_combined_text"),
        ocr_nutrition_text=state.get("ocr_nutrition_text"),
        comparison_analysis=comp_analysis,
        comparison_products=comp_products,
        provider=provider,
        rag_context=rag_str,
    )

    update_dict: dict = {
        "recommendation_analysis": rec_analysis,
    }

    if len(barcodes) >= 2:
        update_dict["comparison_barcodes"] = barcodes

    if product_data and not state.get("product_data"):
        update_dict["product_data"] = product_data

    if comp_products and not state.get("comparison_products"):
        update_dict["comparison_products"] = comp_products

    if comp_analysis and not state.get("comparison_analysis"):
        update_dict["comparison_analysis"] = comp_analysis

    return update_dict


def verification_node(
    state: FoodGuardState,
    provider: NVIDIAProvider | None = None,
) -> FoodGuardState:
    rag_chunks = state.get("rag_context") or []
    rag_str = (
        Retriever().format_rag_context(rag_chunks)
        if rag_chunks
        else None
    )

    decision = state.get("manager_decision")
    intent_val = None

    if decision and hasattr(decision, "intent"):
        intent_val = (
            decision.intent.value
            if hasattr(decision.intent, "value")
            else str(decision.intent)
        )

    analysis = verify_analysis(
        conversation=_provider_messages(state["messages"]),
        product_data=state.get("product_data"),
        nutrition_data=state.get("nutrition_data"),
        ingredient_data=state.get("ingredient_data"),
        safety_data=state.get("safety_data"),
        ocr_combined_text=state.get("ocr_combined_text"),
        ocr_nutrition_text=state.get("ocr_nutrition_text"),
        comparison_products=state.get("comparison_products"),
        comparison_analysis=state.get("comparison_analysis"),
        recommendation_analysis=state.get("recommendation_analysis"),
        rag_context=rag_str,
        provider=provider,
        intent=intent_val,
    )

    return {
        "verification_analysis": analysis,
    }


def _build_deterministic_safety_facts(
    safety_analysis: SafetyAnalysis,
) -> tuple[list[str], list[str], list[str], list[str]]:
    confirmed = [
        item.name
        for item in safety_analysis.allergens
        if item.status.value == "CONFIRMED"
    ]

    not_found = [
        item.name
        for item in safety_analysis.allergens
        if item.status.value == "NOT_FOUND"
    ]

    unknown = [
        item.name
        for item in safety_analysis.allergens
        if item.status.value == "UNKNOWN"
    ]

    warnings = list(safety_analysis.cross_contact_warnings)

    return confirmed, not_found, unknown, warnings


def _format_safety_fact_name(name: str) -> str:
    replacements = {
        "tree_nuts": "tree nuts",
        "peanut": "peanuts",
        "soy": "soy",
        "fish": "fish",
        "shellfish": "shellfish",
        "egg": "eggs",
    }

    return replacements.get(name, name.replace("_", " "))


def _append_deterministic_safety_facts(
    response: str,
    safety_analysis: SafetyAnalysis,
) -> str:
    """
    Preserve authoritative Safety Agent facts in the final answer.

    The LLM explains the facts, but deterministic allergen findings must not
    disappear from the final response.
    """

    confirmed, _, _, warnings = _build_deterministic_safety_facts(
        safety_analysis
    )

    if not confirmed and not warnings:
        return response

    normalized_response = response.lower()

    missing_confirmed = []

    for allergen in confirmed:
        aliases = {
            "milk": ["milk", "dairy"],
            "egg": ["egg", "eggs"],
            "peanut": ["peanut", "peanuts"],
            "tree_nuts": [
                "tree nut",
                "tree nuts",
                "almond",
                "cashew",
                "walnut",
                "pecan",
                "pistachio",
                "hazelnut",
            ],
            "wheat": ["wheat"],
            "soy": ["soy", "soya", "soybean", "soybeans"],
            "fish": [
                "fish",
                "salmon",
                "tuna",
                "cod",
                "anchovy",
            ],
            "shellfish": [
                "shellfish",
                "shrimp",
                "crab",
                "lobster",
                "prawn",
            ],
            "sesame": ["sesame", "tahini"],
        }

        terms = aliases.get(
            allergen,
            [allergen.replace("_", " ")],
        )

        if not any(term in normalized_response for term in terms):
            missing_confirmed.append(allergen)

    if not missing_confirmed:
        return response

    lines = [
        "",
        "",
        "### Deterministic safety facts",
        (
            "Confirmed allergens in the supplied product data: "
            + ", ".join(
                _format_safety_fact_name(name)
                for name in confirmed
            )
            + "."
        ),
    ]

    if warnings:
        lines.append(
            "Cross-contact warning: "
            + " ".join(warnings)
        )

    return response.rstrip() + "\n" + "\n".join(lines)


def response_node(
    state: FoodGuardState,
    provider: NVIDIAProvider | None = None,
) -> FoodGuardState:
    decision = state.get("manager_decision") or ManagerDecision()
    ingredient_analysis = state.get("ingredient_analysis")
    nutrition_analysis = state.get("nutrition_analysis")
    safety_analysis = state.get("safety_analysis")
    comparison_analysis = state.get("comparison_analysis")
    recommendation_analysis = state.get("recommendation_analysis")
    verification_analysis = state.get("verification_analysis")
    product_data = state.get("product_data")
    product_lookup_error = state.get("product_lookup_error")
    ocr_combined_text = state.get("ocr_combined_text")
    ocr_nutrition_text = state.get("ocr_nutrition_text")
    rag_chunks = state.get("rag_context") or []

    if (
        recommendation_analysis is not None
        and not recommendation_analysis.available
        and not rag_chunks
    ):
        return {
            "messages": [
                AIMessage(
                    content=recommendation_analysis.summary
                )
            ]
        }

    if (
        comparison_analysis is not None
        and not comparison_analysis.available
    ):
        return {
            "messages": [
                AIMessage(
                    content=comparison_analysis.summary
                )
            ]
        }

    if (
        ingredient_analysis is not None
        and not ingredient_analysis.available
        and not rag_chunks
    ):
        return {
            "messages": [
                AIMessage(
                    content=ingredient_analysis.summary
                )
            ]
        }

    if (
        nutrition_analysis is not None
        and not nutrition_analysis.available
        and not rag_chunks
    ):
        return {
            "messages": [
                AIMessage(
                    content=nutrition_analysis.summary
                )
            ]
        }

    if (
        safety_analysis is not None
        and not safety_analysis.available
        and not rag_chunks
    ):
        return {
            "messages": [
                AIMessage(
                    content=safety_analysis.summary
                )
            ]
        }

    if (
        product_lookup_error
        == ProductLookupError.PRODUCT_NOT_FOUND.value
    ):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I couldn't find that barcode in the "
                        "product database."
                    )
                )
            ]
        }

    if product_lookup_error:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I couldn't retrieve product data from "
                        "the product database right now."
                    )
                )
            ]
        }

    llm_provider = provider or NVIDIAProvider()
    response_messages = _provider_messages(
        state["messages"]
    )

    verification_context = ""

    if verification_analysis is not None:
        verification_context = (
            "\n\nQuality Control Verification Audit:\n"
            f"{verification_analysis.model_dump_json()}\n"
            "INSTRUCTIONS FOR VERIFICATION:\n"
            "1. If verification status is CONFLICT or conflicting_claims "
            "exist, DO NOT present contradicted claims as fact. Highlight "
            "evidence disparities.\n"
            "2. If safety_flags exist, communicate safety limitations "
            "clearly without medical diagnoses or guarantees.\n"
            "3. If uncertain_items or INSUFFICIENT_EVIDENCE exist, "
            "communicate missing data clearly."
        )

    recommendation_context = ""

    if recommendation_analysis is not None:
        recommendation_context = (
            "\n\nStructured Recommendation Analysis from "
            "Recommendation Agent:\n"
            f"{recommendation_analysis.model_dump_json()}\n"
            "Synthesize a clear, personalized recommendation addressing "
            "the user's preference/goal. Highlight goal alignment, "
            "trade-offs, safety warnings, and explicit missing "
            "data/uncertainties."
        )

    comparison_context = ""

    if comparison_analysis is not None:
        comparison_context = (
            "\n\nStructured Product Comparison Analysis:\n"
            f"{comparison_analysis.model_dump_json()}\n"
            "Present a clear comparison summary and markdown comparison "
            "table for the user."
        )

    ingredient_context = ""

    if ingredient_analysis is not None:
        ingredient_context = (
            "\n\nStructured ingredient analysis from the Ingredient Agent:\n"
            f"{ingredient_analysis.model_dump_json()}\n"
            "Use this analysis accurately. Do not expose raw JSON "
            "unless useful."
        )

    nutrition_context = ""

    if nutrition_analysis is not None:
        nutrition_context = (
            "\n\nStructured nutrition analysis from the Nutrition Agent:\n"
            f"{nutrition_analysis.model_dump_json()}\n"
            "Use the deterministic normalized values accurately. "
            "Do not expose raw JSON unless useful."
        )

    safety_context = ""

    if safety_analysis is not None:
        (
            confirmed_allergens,
            not_found_allergens,
            unknown_allergens,
            cross_contact_warnings,
        ) = _build_deterministic_safety_facts(
            safety_analysis
        )

        safety_facts = {
            "confirmed_allergens": confirmed_allergens,
            "not_found_allergens": not_found_allergens,
            "unknown_allergens": unknown_allergens,
            "cross_contact_warnings": cross_contact_warnings,
            "overall_status": (
                safety_analysis.overall_status.value
            ),
        }

        safety_context = (
            "\n\nStructured safety and allergen analysis from "
            "the Safety Agent:\n"
            f"{safety_analysis.model_dump_json()}\n\n"
            "DETERMINISTIC SAFETY FACTS — AUTHORITATIVE:\n"
            f"{safety_facts}\n\n"
            "SAFETY RESPONSE RULES:\n"
            "1. Treat confirmed_allergens as authoritative.\n"
            "2. When answering an allergen/product safety question, "
            "do not omit a confirmed allergen that is relevant to "
            "the product.\n"
            "3. Distinguish confirmed allergens from cross-contact "
            "warnings.\n"
            "4. A 'may contain' or cross-contact warning does not "
            "mean that the allergen is a confirmed ingredient.\n"
            "5. Treat not_found_allergens as explicit absence from "
            "the supplied information, not a universal guarantee.\n"
            "6. Treat unknown_allergens as unknown.\n"
            "7. Never claim guaranteed allergen-free or medically "
            "safe status.\n"
            "8. Never contradict deterministic safety facts."
        )

    product_context = ""

    if product_data is not None:
        product_context = (
            "\n\nNormalized product data retrieved from "
            "Open Food Facts:\n"
            f"{product_data.model_dump_json()}\n"
            "Attribute this as community-contributed Open Food Facts "
            "data, not manufacturer certification. Use only fields "
            "that are present and say when information is missing."
        )

    ocr_context = ""

    if ocr_combined_text or ocr_nutrition_text:
        ocr_context = (
            "\n\nOCR-derived label text (imperfect extraction, "
            "not verified product data):\n"
            f"{ocr_combined_text or ''}\n"
            f"Nutrition section: "
            f"{ocr_nutrition_text or '[not detected]'}\n"
            "Do not invent or treat missing OCR information as absent."
        )

    rag_context = ""

    if rag_chunks:
        retriever = Retriever()
        rag_str = retriever.format_rag_context(
            rag_chunks
        )

        rag_context = (
            "\n\nAuthoritative Reference Knowledge "
            "(retrieved from RAG Knowledge Base):\n"
            f"{rag_str}\n"
            "INSTRUCTIONS FOR REFERENCE KNOWLEDGE:\n"
            "1. Use reference knowledge to clarify regulations, "
            "daily values, or additive functions.\n"
            "2. DO NOT allow general reference knowledge to overwrite "
            "specific product label, OCR, or barcode data.\n"
            "3. Cite source titles (e.g. [Source: FDA Allergen Guide]) "
            "when using reference knowledge.\n"
            "4. Do not invent sources or fabricate claims."
        )

    response_messages.insert(
        0,
        {
            "role": "system",
            "content": (
                "You are FoodGuard AI, a helpful conversational "
                "food-analysis assistant. For now, answer general "
                "food and nutrition questions clearly. Do not diagnose "
                "medical conditions or provide medical treatment. "
                "When product-specific data is not available, say so "
                "instead of inventing information.\n\n"
                f"Manager routing decision: "
                f"{decision_to_dict(decision)}\n"
                "CONVERSATIONAL MULTI-TURN INSTRUCTIONS:\n"
                "1. For follow-up questions (e.g., 'What about the "
                "sugar?', 'Is that high?', 'What about protein?'), "
                "provide a direct, concise answer addressing the "
                "specific question. DO NOT repeat full ingredient "
                "lists or complete nutrition tables unless requested.\n"
                "2. Maintain natural context across turns "
                "(referring to the active product, active comparison "
                "products, or preceding recommendation).\n"
                "3. If a product-specific question is asked without "
                "active product/barcode/OCR context, politely ask the "
                "user to provide a barcode or upload a label image.\n"
                "4. Communicate missing or conflicting data clearly "
                "in plain language rather than exposing raw "
                "code/enums.\n"
                "5. Safety boundaries: NEVER make medical diagnoses, "
                "disease cure/prevention claims, or guaranteed "
                "'100% allergen-free' claims."
                f"{verification_context}"
                f"{recommendation_context}"
                f"{comparison_context}"
                f"{ingredient_context}"
                f"{nutrition_context}"
                f"{safety_context}"
                f"{product_context}"
                f"{ocr_context}"
                f"{rag_context}"
            ),
        },
    )

    response = llm_provider.generate_response(
        response_messages
    )

    # Final deterministic safety guard.
    # The LLM explains the result, but authoritative confirmed
    # allergens must not be silently omitted.
    if safety_analysis is not None:
        response = _append_deterministic_safety_facts(
            response,
            safety_analysis,
        )

    return {
        "messages": [
            AIMessage(content=response)
        ]
    }


def ingredient_node(
    state: FoodGuardState,
    provider: NVIDIAProvider | None = None,
) -> FoodGuardState:
    rag_chunks = state.get("rag_context") or []
    rag_str = (
        Retriever().format_rag_context(rag_chunks)
        if rag_chunks
        else None
    )

    ingredient_data = state.get("ingredient_data")
    product_data = state.get("product_data")

    if (
        not ingredient_data
        and not product_data
        and state.get("product_lookup_request")
    ):
        res = lookup_product_by_barcode(
            state["product_lookup_request"]
        )

        if res.found and res.product:
            product_data = res.product
            ingredient_data = product_data.ingredients_text

    analysis = analyze_ingredients(
        ingredient_data,
        _provider_messages(state["messages"]),
        provider=provider,
        rag_context=rag_str,
    )

    ret_dict: dict = {
        "ingredient_analysis": analysis,
    }

    if product_data and not state.get("product_data"):
        ret_dict["product_data"] = product_data
        ret_dict["ingredient_data"] = ingredient_data

    return ret_dict


def nutrition_node(
    state: FoodGuardState,
    provider: NVIDIAProvider | None = None,
) -> FoodGuardState:
    rag_chunks = state.get("rag_context") or []
    rag_str = (
        Retriever().format_rag_context(rag_chunks)
        if rag_chunks
        else None
    )

    nutrition_data = state.get("nutrition_data")
    product_data = state.get("product_data")

    if (
        not nutrition_data
        and not product_data
        and state.get("product_lookup_request")
    ):
        res = lookup_product_by_barcode(
            state["product_lookup_request"]
        )

        if res.found and res.product:
            product_data = res.product
            nutrition_data = product_data.nutrition

    analysis = analyze_nutrition(
        nutrition_data,
        _provider_messages(state["messages"]),
        provider=provider,
        rag_context=rag_str,
    )

    ret_dict: dict = {
        "nutrition_analysis": analysis,
    }

    if product_data and not state.get("product_data"):
        ret_dict["product_data"] = product_data
        ret_dict["nutrition_data"] = nutrition_data

    return ret_dict


def safety_node(
    state: FoodGuardState,
    provider: NVIDIAProvider | None = None,
) -> FoodGuardState:
    safety_data = state.get("safety_data") or {}
    product_data = state.get("product_data")

    if (
        not safety_data.get("ingredient_text")
        and not product_data
        and state.get("product_lookup_request")
    ):
        res = lookup_product_by_barcode(
            state["product_lookup_request"]
        )

        if res.found and res.product:
            product_data = res.product

            safety_data = {
                "ingredient_text": product_data.ingredients_text,
                "allergen_statement": product_data.allergen_statement,
                "cross_contact_statement": (
                    ", ".join(product_data.traces)
                    or None
                ),
            }

    rag_chunks = state.get("rag_context") or []
    rag_str = (
        Retriever().format_rag_context(rag_chunks)
        if rag_chunks
        else None
    )

    analysis = analyze_safety(
        safety_data.get("ingredient_text"),
        safety_data.get("allergen_statement"),
        safety_data.get("cross_contact_statement"),
        _provider_messages(state["messages"]),
        provider=provider,
        rag_context=rag_str,
    )

    ret_dict: dict = {
        "safety_analysis": analysis,
    }

    if product_data and not state.get("product_data"):
        ret_dict["product_data"] = product_data
        ret_dict["safety_data"] = safety_data

    return ret_dict


def product_lookup_node(
    state: FoodGuardState,
) -> FoodGuardState:
    barcode = state.get("product_lookup_request")

    if not barcode:
        messages = state.get("messages") or []

        for msg in reversed(messages):
            if msg.type == "human":
                matches = re.findall(
                    r"\b\d{8,14}\b",
                    str(msg.content),
                )

                if matches:
                    barcode = matches[0]
                    break

    if not barcode:
        return {
            "product_lookup_error": (
                ProductLookupError.INVALID_BARCODE.value
            )
        }

    result = lookup_product_by_barcode(barcode)

    if not result.found or result.product is None:
        return {
            "product_lookup_error": (
                result.error.value
                if result.error
                else ProductLookupError.MISSING_PRODUCT_DATA.value
            )
        }

    product = result.product

    return {
        "product_data": product,
        "product_lookup_error": None,
        "ingredient_data": product.ingredients_text,
        "nutrition_data": product.nutrition,
        "safety_data": {
            "ingredient_text": product.ingredients_text,
            "allergen_statement": product.allergen_statement,
            "cross_contact_statement": (
                ", ".join(product.traces)
                or None
            ),
        },
    }