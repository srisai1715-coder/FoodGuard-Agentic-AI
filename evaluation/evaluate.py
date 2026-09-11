"""
FoodGuard AI - System Evaluation Runner (evaluate.py)

Executes AI system evaluations across benchmark cases in evaluation/dataset.json.
Runs deterministic metrics (metrics.py) and optional LLM-as-a-Judge evaluations (judges.py).
Generates structured JSON reports in evaluation/results/ and prints detailed terminal summaries.

Important evaluation behavior:
- Transient NVIDIA/API 429 rate-limit errors are retried with exponential backoff.
- Unrecovered infrastructure failures are NOT counted as AI-quality failures.
- Infrastructure failures are reported separately.
- Successful cases are used for AI-quality metric denominators.
- A small cooldown is applied between cases to reduce API rate-limit pressure.

Usage:
  python evaluation/evaluate.py [--split dev|eval|all] [--category <cat>] [--limit N] [--judge]
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_path = os.path.join(root_path, "backend")

if root_path not in sys.path:
    sys.path.insert(0, root_path)

if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

from unittest.mock import patch

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.tools.product_lookup import (
    ProductData,
    ProductDataQuality,
    ProductLookupResult,
)
from app.tools.nutrition import NutritionData

from evaluation.metrics import (
    calculate_precision_recall_f1,
    evaluate_numeric_tolerance,
    evaluate_allergen_detection,
    verify_nutrient_deltas,
    evaluate_routing,
    evaluate_rag_retrieval,
    evaluate_hallucination_and_safety,
    normalize_agent_name,
    canonicalize_allergen_string,
    INTENT_TO_SPECIALISTS_MAP,
)


# ---------------------------------------------------------------------------
# OPTIONAL LLM JUDGE
# ---------------------------------------------------------------------------

try:
    from evaluation.judges import LLMJudge

    JUDGE_AVAILABLE = True
    JUDGE_ERROR = None

except Exception as e:
    JUDGE_AVAILABLE = False
    JUDGE_ERROR = str(e)


# ---------------------------------------------------------------------------
# EVALUATION CONFIGURATION
# ---------------------------------------------------------------------------

# Number of retries after the initial attempt.
MAX_RATE_LIMIT_RETRIES = 4

# Initial delay before first retry.
# Retry delays become:
# 3s -> 6s -> 12s -> 24s
RATE_LIMIT_BASE_DELAY_SECONDS = 3.0

# Small pause between benchmark cases.
# This helps reduce NVIDIA API burst/rate-limit pressure.
CASE_COOLDOWN_SECONDS = 1.5


# ---------------------------------------------------------------------------
# FROZEN BENCHMARK PRODUCT DATA
# ---------------------------------------------------------------------------

def get_frozen_product_data(
    barcode: Optional[str],
) -> Optional[ProductData]:
    """
    Returns verified, frozen reference ProductData for benchmark barcodes.

    This guarantees deterministic benchmark facts independent of:
    - Open Food Facts availability
    - network failures
    - live product-data changes
    """

    if not barcode:
        return None

    clean = barcode.strip()

    # -----------------------------------------------------------------------
    # HEB CREAMY PEANUT BUTTER
    # -----------------------------------------------------------------------

    if clean in {"0041220576922", "041220576922"}:

        nut = NutritionData(
            serving_size_g=32.0,
            calories=190.0,
            protein_g=7.0,
            total_sugar_g=3.0,
            fat_g=16.0,
            sodium_mg=140.0,
            values_per_100g={
                "calories_per_100g": 588.0,
                "total_sugar_g_per_100g": 9.4,
                "protein_g_per_100g": 21.9,
                "fat_g_per_100g": 50.0,
                "sodium_mg_per_100g": 438.0,
            },
        )

        return ProductData(
            barcode="0041220576922",
            product_name="HEB Creamy Peanut Butter",
            brand="H-E-B",
            ingredients_text=(
                "Peanuts, Sugar, Hydrogenated Vegetable Oil, Salt"
            ),
            allergen_statement=(
                "Contains peanuts and soy. May contain traces of milk."
            ),
            allergens=["peanuts", "soybeans"],
            traces=["milk"],
            nutrition=nut,
            source_url="frozen_benchmark://0041220576922",
            data_quality=ProductDataQuality.COMPLETE,
        )

    # -----------------------------------------------------------------------
    # KROGER CREAMY PEANUT BUTTER
    # -----------------------------------------------------------------------

    elif clean in {"0011110804473", "011110804473"}:

        nut = NutritionData(
            serving_size_g=32.0,
            calories=190.0,
            protein_g=7.0,
            total_sugar_g=2.0,
            fat_g=16.0,
            sodium_mg=140.0,
            values_per_100g={
                "calories_per_100g": 588.0,
                "total_sugar_g_per_100g": 6.25,
                "protein_g_per_100g": 21.88,
                "fat_g_per_100g": 50.0,
                "sodium_mg_per_100g": 438.0,
            },
        )

        return ProductData(
            barcode="0011110804473",
            product_name="Kroger Creamy Peanut Butter",
            brand="Kroger",
            ingredients_text=(
                "Roasted Peanuts, Sugar, Hydrogenated Vegetable Oil, Salt"
            ),
            allergen_statement="Contains peanuts.",
            allergens=["peanuts"],
            traces=[],
            nutrition=nut,
            source_url="frozen_benchmark://0011110804473",
            data_quality=ProductDataQuality.COMPLETE,
        )

    return None


# ---------------------------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_path: str,
) -> List[Dict[str, Any]]:
    """Load benchmark dataset from JSON."""

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found at {dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# NUTRITION EXTRACTION
# ---------------------------------------------------------------------------

def extract_nutrient_from_text(
    text: str,
    nutrient_key: str,
) -> Tuple[Optional[float], str]:
    """
    Extract numerical nutrient values from natural-language AI response text.

    Returns:
        (extracted_value, extraction_status)
    """

    if not text:
        return None, "empty_text"

    clean_text = (
        text
        .replace("*", "")
        .replace("\u202f", " ")
        .lower()
    )

    patterns = []

    if "sugar" in nutrient_key:

        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:grams?|g)?\s*(?:of\s*)?(?:total\s*)?sugars?",
            r"sugars?[:\s\-]+(\d+(?:\.\d+)?)\s*(?:grams?|g)?",
            r"sugars?[^\d\n]{1,30}(\d+(?:\.\d+)?)\s*(?:grams?|g)?",
            r"contains\s*(\d+(?:\.\d+)?)\s*(?:grams?|g)\s*(?:of\s*)?(?:total\s*)?sugar",
        ]

    elif "protein" in nutrient_key:

        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:grams?|g)?\s*(?:of\s*)?proteins?",
            r"proteins?[:\s\-]+(\d+(?:\.\d+)?)\s*(?:grams?|g)?",
            r"proteins?[^\d\n]{1,30}(\d+(?:\.\d+)?)\s*(?:grams?|g)?",
            r"contains\s*(\d+(?:\.\d+)?)\s*(?:grams?|g)\s*(?:of\s*)?protein",
        ]

    elif "calorie" in nutrient_key:

        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:cal|calories|kcal)\b",
            r"calories?[:\s\-]+(\d+(?:\.\d+)?)",
            r"calories?[^\d\n]{1,30}(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*calories",
        ]

    elif "fat" in nutrient_key:

        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:grams?|g)?\s*(?:of\s*)?(?:total\s*)?fat",
            r"(?:total\s*)?fat[:\s\-]+(\d+(?:\.\d+)?)\s*(?:grams?|g)?",
            r"fat[^\d\n]{1,30}(\d+(?:\.\d+)?)\s*(?:grams?|g)?",
        ]

    elif "sodium" in nutrient_key:

        patterns = [
            r"(\d+(?:\.\d+)?)\s*mg\s*(?:of\s*)?sodium",
            r"sodium[:\s\-]+(\d+(?:\.\d+)?)\s*mg",
            r"(\d+(?:\.\d+)?)\s*g\s*(?:of\s*)?sodium",
            r"sodium[:\s\-]+(\d+(?:\.\d+)?)\s*g",
            r"sodium[^\d\n]{1,30}(\d+(?:\.\d+)?)\s*(?:mg|g)",
        ]

    elif "serving" in nutrient_key:

        patterns = [
            r"serving\s*size[:\s\-]+(\d+(?:\.\d+)?)\s*g",
            r"(\d+(?:\.\d+)?)\s*g\s*serving",
        ]

    else:

        patterns = [
            r"(\d+(?:\.\d+)?)"
        ]

    for pattern in patterns:

        match = re.search(pattern, clean_text)

        if match:

            try:
                value = float(match.group(1))

                if (
                    "sodium" in nutrient_key
                    and "mg" in match.group(0)
                ):
                    value = round(value / 1000.0, 4)

                return value, "SUCCESS"

            except (ValueError, IndexError):
                continue

    return None, "pattern_not_found"


# ---------------------------------------------------------------------------
# SINGLE CASE EXECUTION
# ---------------------------------------------------------------------------

def run_case_execution(
    case: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a single benchmark case against the real FoodGuard AI system.

    This function performs ONE execution attempt.

    Rate-limit retry handling is intentionally implemented outside this
    function so that each attempt remains easy to reason about.
    """

    start_time = time.time()

    category = case.get("category", "")
    user_input = case.get("user_input", "")
    context = case.get("context", {})

    result = {
        "case_id": case.get("id"),
        "category": category,
        "split": case.get("split"),
        "user_input": user_input,
        "status": "SUCCESS",
        "error_type": None,
        "error_message": None,
        "actual_response": "",
        "actual_intent": "UNKNOWN",
        "actual_agents": [],
        "retrieved_docs": [],
        "structured_analyses": {},
        "latency_ms": 0,
    }

    try:

        # ===================================================================
        # MANAGER ROUTING
        # ===================================================================

        if category == "manager_routing":

            from app.agents.manager import (
                classify_intent,
                decision_to_dict,
            )

            manager_messages = [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]

            decision = classify_intent(manager_messages)

            decision_data = decision_to_dict(decision)

            raw_intent = decision_data.get(
                "intent",
                "UNKNOWN",
            )

            result["actual_intent"] = raw_intent

            routed = INTENT_TO_SPECIALISTS_MAP.get(
                raw_intent,
                ["manager"],
            )

            result["actual_agents"] = [
                normalize_agent_name(agent)
                for agent in routed
            ]

            result["actual_response"] = (
                f"Manager classified intent: {raw_intent} "
                f"-> Specialists: {result['actual_agents']}"
            )

        # ===================================================================
        # RAG RETRIEVAL
        # ===================================================================

        elif category == "rag_retrieval":

            from app.rag.retriever import Retriever

            retriever = Retriever()

            # Align in-memory embeddings with active process hash function.
            for chunk in retriever.vector_store.chunks:

                chunk["embedding"] = (
                    retriever.embedding_generator.embed_text(
                        chunk.get("text", "")
                    )
                )

            chunks = retriever.retrieve(
                user_input,
                top_k=3,
            )

            result["retrieved_docs"] = [
                {
                    "chunk_id": chunk.get(
                        "chunk_id",
                        chunk.get(
                            "metadata",
                            {}
                        ).get(
                            "chunk_id",
                            ""
                        ),
                    ),
                    "source": chunk.get(
                        "metadata",
                        {}
                    ).get(
                        "source_file",
                        chunk.get(
                            "metadata",
                            {}
                        ).get(
                            "source",
                            "knowledge_base",
                        ),
                    ),
                    "content": chunk.get(
                        "text",
                        "",
                    )[:200],
                    "score": round(
                        chunk.get(
                            "score",
                            0.0,
                        ),
                        4,
                    ),
                }
                for chunk in chunks
            ]

            result["actual_response"] = "\n---\n".join(
                [
                    (
                        f"[{doc['chunk_id']} | "
                        f"{doc['source']}] "
                        f"{doc['content']}"
                    )
                    for doc in result["retrieved_docs"]
                ]
            )

            result["actual_intent"] = "GENERAL_QUESTION"
            result["actual_agents"] = ["rag"]

        # ===================================================================
        # FULL LANGGRAPH WORKFLOW
        # ===================================================================

        else:

            from app.graph.workflow import build_workflow

            graph = build_workflow()

            messages = []

            for history_item in context.get(
                "conversation_history",
                [],
            ):

                role = history_item.get(
                    "role",
                    "user",
                )

                content = history_item.get(
                    "content",
                    "",
                )

                if role == "user":

                    messages.append(
                        HumanMessage(
                            content=content
                        )
                    )

                elif role == "assistant":

                    messages.append(
                        AIMessage(
                            content=content
                        )
                    )

                elif role == "system":

                    messages.append(
                        SystemMessage(
                            content=content
                        )
                    )

            messages.append(
                HumanMessage(
                    content=user_input
                )
            )

            barcode = (
                context.get("barcode")
                or context.get("product_lookup_request")
            )

            barcodes = context.get("barcodes")

            prod = get_frozen_product_data(
                barcode
            )

            comp_prods = None

            if barcodes and len(barcodes) >= 2:

                prod_a = get_frozen_product_data(
                    barcodes[0]
                )

                prod_b = get_frozen_product_data(
                    barcodes[1]
                )

                if prod_a and prod_b:
                    comp_prods = [
                        prod_a,
                        prod_b,
                    ]

            init_state = {
                "messages": messages,

                "product_lookup_request": barcode,

                "comparison_barcodes": (
                    barcodes
                    or ([barcode] if barcode else None)
                ),

                "comparison_products": comp_prods,

                "product_data": prod,

                "nutrition_data": (
                    prod.nutrition
                    if prod
                    else None
                ),

                "ingredient_data": (
                    prod.ingredients_text
                    if prod
                    else None
                ),

                "safety_data": (
                    {
                        "ingredient_text": (
                            prod.ingredients_text
                        ),
                        "allergen_statement": (
                            prod.allergen_statement
                        ),
                        "cross_contact_statement": (
                            ", ".join(prod.traces)
                            if prod.traces
                            else None
                        ),
                    }
                    if prod
                    else None
                ),

                "ocr_combined_text": (
                    context.get(
                        "ocr_combined_text"
                    )
                    or context.get(
                        "ocr_text"
                    )
                ),

                "ocr_nutrition_text": (
                    context.get(
                        "ocr_nutrition_text"
                    )
                ),
            }

            # ---------------------------------------------------------------
            # Freeze benchmark product lookup.
            # ---------------------------------------------------------------

            def _mock_lookup(
                lookup_barcode: str,
            ):

                frozen_product = get_frozen_product_data(
                    lookup_barcode
                )

                if frozen_product:

                    return ProductLookupResult(
                        found=True,
                        product=frozen_product,
                    )

                from app.tools.product_lookup import (
                    lookup_product_by_barcode as original_lookup,
                )

                return original_lookup(
                    lookup_barcode
                )

            with patch(
                "app.graph.nodes.lookup_product_by_barcode",
                side_effect=_mock_lookup,
            ):

                output = graph.invoke(
                    init_state
                )

            # ---------------------------------------------------------------
            # Extract final response.
            # ---------------------------------------------------------------

            msgs = output.get(
                "messages",
                []
            )

            final_msg = ""

            if msgs:

                last_msg = msgs[-1]

                final_msg = (
                    last_msg.content
                    if hasattr(
                        last_msg,
                        "content"
                    )
                    else str(last_msg)
                )

            result["actual_response"] = final_msg

            # ---------------------------------------------------------------
            # Extract Manager decision.
            # ---------------------------------------------------------------

            decision = output.get(
                "manager_decision"
            )

            if decision:

                intent_val = getattr(
                    decision.intent,
                    "value",
                    str(decision.intent),
                )

                result["actual_intent"] = intent_val

            else:

                result["actual_intent"] = "PRODUCT_ANALYSIS"

            # ---------------------------------------------------------------
            # Extract active specialist agents.
            # ---------------------------------------------------------------

            active_agents = set()

            if output.get(
                "nutrition_analysis"
            ) is not None:

                active_agents.add(
                    "nutrition"
                )

                result[
                    "structured_analyses"
                ]["nutrition"] = (
                    output[
                        "nutrition_analysis"
                    ].model_dump(
                        mode="json"
                    )
                )

            if output.get(
                "ingredient_analysis"
            ) is not None:

                active_agents.add(
                    "ingredient"
                )

                result[
                    "structured_analyses"
                ]["ingredient"] = (
                    output[
                        "ingredient_analysis"
                    ].model_dump(
                        mode="json"
                    )
                )

            if output.get(
                "safety_analysis"
            ) is not None:

                active_agents.add(
                    "safety"
                )

                result[
                    "structured_analyses"
                ]["safety"] = (
                    output[
                        "safety_analysis"
                    ].model_dump(
                        mode="json"
                    )
                )

            if output.get(
                "comparison_analysis"
            ) is not None:

                active_agents.add(
                    "comparison"
                )

                result[
                    "structured_analyses"
                ]["comparison"] = (
                    output[
                        "comparison_analysis"
                    ].model_dump(
                        mode="json"
                    )
                )

            if output.get(
                "recommendation_analysis"
            ) is not None:

                active_agents.add(
                    "recommendation"
                )

                result[
                    "structured_analyses"
                ]["recommendation"] = (
                    output[
                        "recommendation_analysis"
                    ].model_dump(
                        mode="json"
                    )
                )

            if output.get(
                "verification_analysis"
            ) is not None:

                active_agents.add(
                    "verification"
                )

                result[
                    "structured_analyses"
                ]["verification"] = (
                    output[
                        "verification_analysis"
                    ].model_dump(
                        mode="json"
                    )
                )

            if not active_agents:

                active_agents = set(
                    INTENT_TO_SPECIALISTS_MAP.get(
                        result["actual_intent"],
                        ["manager"],
                    )
                )

            result["actual_agents"] = [
                normalize_agent_name(agent)
                for agent in active_agents
            ]

            # ---------------------------------------------------------------
            # Extract RAG documents.
            # ---------------------------------------------------------------

            if output.get(
                "rag_context"
            ):

                result["retrieved_docs"] = [
                    {
                        "chunk_id": c.get(
                            "chunk_id",
                            c.get(
                                "metadata",
                                {}
                            ).get(
                                "chunk_id",
                                ""
                            ),
                        ),
                        "source": c.get(
                            "metadata",
                            {}
                        ).get(
                            "source_file",
                            c.get(
                                "metadata",
                                {}
                            ).get(
                                "source",
                                "knowledge_base",
                            ),
                        ),
                        "content": c.get(
                            "text",
                            "",
                        )[:200],
                    }
                    for c in output[
                        "rag_context"
                    ]
                ]

    except Exception as e:

        result["status"] = "FAILURE"

        result["error_type"] = (
            "Software Execution Failure"
        )

        result["error_message"] = str(e)

        result["actual_response"] = (
            f"EXECUTION ERROR: {str(e)}"
        )

    result["latency_ms"] = round(
        (time.time() - start_time) * 1000,
        2,
    )

    return result


# ---------------------------------------------------------------------------
# RATE LIMIT DETECTION
# ---------------------------------------------------------------------------

def is_rate_limit_error(
    execution_result: Dict[str, Any],
) -> bool:
    """
    Determine whether a failed execution is caused by a transient
    NVIDIA/API rate-limit condition.
    """

    if execution_result.get("status") == "SUCCESS":
        return False

    combined_text = " ".join(
        [
            str(
                execution_result.get(
                    "error_message",
                    ""
                )
            ),
            str(
                execution_result.get(
                    "actual_response",
                    ""
                )
            ),
        ]
    ).lower()

    rate_limit_markers = [
        "429",
        "too many requests",
        "rate limit",
        "rate-limit",
        "ratelimit",
        "requests per minute",
        "quota exceeded",
        "throttled",
        "throttle",
    ]

    return any(
        marker in combined_text
        for marker in rate_limit_markers
    )


# ---------------------------------------------------------------------------
# CASE EXECUTION WITH RATE-LIMIT RETRIES
# ---------------------------------------------------------------------------

def run_case_execution_with_retry(
    case: Dict[str, Any],
    max_retries: int = MAX_RATE_LIMIT_RETRIES,
    base_delay: float = RATE_LIMIT_BASE_DELAY_SECONDS,
) -> Dict[str, Any]:
    """
    Execute a benchmark case with exponential backoff for transient
    NVIDIA/API rate-limit failures.

    Initial attempt + max_retries.

    Example with max_retries=4:

        Attempt 1
          ↓ 429
        wait 3s
        Attempt 2
          ↓ 429
        wait 6s
        Attempt 3
          ↓ 429
        wait 12s
        Attempt 4
          ↓ 429
        wait 24s
        Attempt 5

    An unrecovered rate-limit failure is marked as an infrastructure
    failure and must not be counted as an AI-quality failure.
    """

    last_result = None

    for attempt in range(max_retries + 1):

        if attempt > 0:

            delay = base_delay * (
                2 ** (attempt - 1)
            )

            print(
                f"  Rate limit detected for "
                f"{case.get('id')}. "
                f"Retry {attempt}/{max_retries} "
                f"after {delay:.1f}s...",
                flush=True,
            )

            time.sleep(delay)

        result = run_case_execution(
            case
        )

        last_result = result

        # ---------------------------------------------------------------
        # Successful execution.
        # ---------------------------------------------------------------

        if result.get("status") == "SUCCESS":

            result["retry_count"] = attempt
            result["execution_failure"] = None

            return result

        # ---------------------------------------------------------------
        # Non-rate-limit failure.
        #
        # Do not retry software failures because repeated execution will
        # not normally fix a deterministic application bug.
        # ---------------------------------------------------------------

        if not is_rate_limit_error(result):

            result["retry_count"] = attempt
            result["execution_failure"] = (
                "NON_RATE_LIMIT_EXECUTION_FAILURE"
            )

            return result

        # ---------------------------------------------------------------
        # Rate-limit failure.
        # ---------------------------------------------------------------

        result["retry_count"] = attempt
        result["execution_failure"] = "RATE_LIMIT"

    # -----------------------------------------------------------------------
    # All retries exhausted.
    # -----------------------------------------------------------------------

    if last_result is None:

        last_result = {
            "case_id": case.get("id"),
            "category": case.get("category", ""),
            "split": case.get("split", "dev"),
            "user_input": case.get("user_input", ""),
            "status": "FAILURE",
            "error_type": "RATE_LIMIT",
            "error_message": (
                "No execution result was returned."
            ),
            "actual_response": (
                "EXECUTION ERROR: Rate-limit retries exhausted."
            ),
            "actual_intent": "UNKNOWN",
            "actual_agents": [],
            "retrieved_docs": [],
            "structured_analyses": {},
            "latency_ms": 0,
        }

    last_result["retry_exhausted"] = True
    last_result["execution_failure"] = "RATE_LIMIT"

    return last_result


# ---------------------------------------------------------------------------
# BENCHMARK EVALUATION
# ---------------------------------------------------------------------------

def evaluate_benchmark(
    dataset: List[Dict[str, Any]],
    use_judge: bool = False,
    judge_model: str = "meta/llama-3.3-70b-instruct",
) -> Dict[str, Any]:
    """
    Run evaluation across benchmark cases.

    Important:

    Successful cases contribute to AI-quality metrics.

    Unrecovered infrastructure failures such as NVIDIA 429 responses
    are excluded from AI-quality denominators and reported separately.
    """

    case_results = []

    # -----------------------------------------------------------------------
    # QUALITY METRIC COLLECTIONS
    # -----------------------------------------------------------------------

    intent_matches = []
    routing_matches = []

    nutrient_evals = []
    allergen_evals = []
    delta_evals = []
    rag_evals = []
    safety_evals = []
    judge_scores = []

    extraction_failures = []

    # -----------------------------------------------------------------------
    # INFRASTRUCTURE METRICS
    # -----------------------------------------------------------------------

    infrastructure_failures = []

    successful_case_count = 0
    failed_case_count = 0
    rate_limit_failure_count = 0
    non_rate_limit_failure_count = 0
    retry_count_total = 0

    # -----------------------------------------------------------------------
    # OPTIONAL JUDGE
    # -----------------------------------------------------------------------

    judge = None

    if use_judge:

        if JUDGE_AVAILABLE:

            judge = LLMJudge(
                model_name=judge_model
            )

        else:

            print(
                "Warning: LLM Judge requested but unavailable "
                f"({JUDGE_ERROR}). Skipping judge evaluation."
            )

    print(
        f"Starting evaluation across "
        f"{len(dataset)} benchmark cases...\n"
    )

    # -----------------------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------------------

    for idx, case in enumerate(
        dataset,
        1,
    ):

        case_id = case["id"]
        category = case.get(
            "category",
            ""
        )
        split = case.get(
            "split",
            "dev"
        )

        print(
            f"[{idx:03d}/{len(dataset):03d}] "
            f"Running {case_id} "
            f"({category} - {split})...",
            flush=True,
        )

        # ---------------------------------------------------------------
        # Execute with retry.
        # ---------------------------------------------------------------

        exec_res = run_case_execution_with_retry(
            case
        )

        retry_count_total += int(
            exec_res.get(
                "retry_count",
                0
            )
        )

        is_execution_failure = (
            exec_res.get("status")
            != "SUCCESS"
        )

        # ---------------------------------------------------------------
        # Infrastructure failure handling.
        # ---------------------------------------------------------------

        if is_execution_failure:

            failed_case_count += 1

            failure_type = exec_res.get(
                "execution_failure",
                "UNKNOWN_EXECUTION_FAILURE",
            )

            if failure_type == "RATE_LIMIT":

                rate_limit_failure_count += 1

            else:

                non_rate_limit_failure_count += 1

            infrastructure_failures.append(
                {
                    "case_id": case_id,
                    "category": category,
                    "split": split,
                    "failure_type": failure_type,
                    "error_type": exec_res.get(
                        "error_type"
                    ),
                    "error_message": exec_res.get(
                        "error_message"
                    ),
                    "retry_count": exec_res.get(
                        "retry_count",
                        0
                    ),
                }
            )

        else:

            successful_case_count += 1

        actual_text = exec_res.get(
            "actual_response",
            ""
        )

        structured = exec_res.get(
            "structured_analyses",
            {}
        )

        # ---------------------------------------------------------------
        # IMPORTANT:
        #
        # AI-quality metrics are evaluated ONLY for successful executions.
        #
        # This prevents:
        #
        # EXECUTION ERROR: 429
        #
        # from being interpreted as:
        #
        # "The AI failed to identify the allergen."
        # ---------------------------------------------------------------

        if not is_execution_failure:

            # ===========================================================
            # 1. MANAGER ROUTING METRICS
            # ===========================================================

            if (
                category == "manager_routing"
                or "expected_intent" in case
            ):

                route_res = evaluate_routing(
                    expected_intent=case.get(
                        "expected_intent",
                        "UNKNOWN",
                    ),
                    actual_intent=exec_res.get(
                        "actual_intent",
                        "UNKNOWN",
                    ),
                    expected_agents=case.get(
                        "expected_agents",
                        [],
                    ),
                    actual_agents=exec_res.get(
                        "actual_agents",
                        [],
                    ),
                )

                exec_res["routing_eval"] = route_res

                intent_matches.append(
                    1
                    if route_res["intent_matched"]
                    else 0
                )

                routing_matches.append(
                    1
                    if (
                        route_res[
                            "routing_perfect_match"
                        ]
                        or route_res[
                            "agent_routing_recall"
                        ] == 1.0
                    )
                    else 0
                )

            # ===========================================================
            # 2. NUTRITION METRICS
            # ===========================================================

            if (
                category == "nutrition_analysis"
                and "expected_facts" in case
            ):

                exp_facts = case[
                    "expected_facts"
                ]

                nut_eval_list = []

                for nut_key in [
                    "sugars_100g",
                    "proteins_100g",
                    "calories_100g",
                    "fat_100g",
                    "sodium_100g",
                    "serving_size_g",
                    "calories_serving",
                ]:

                    if nut_key not in exp_facts:
                        continue

                    exp_val = exp_facts[
                        nut_key
                    ]

                    extracted_val, ext_status = (
                        extract_nutrient_from_text(
                            actual_text,
                            nut_key,
                        )
                    )

                    # ---------------------------------------------------
                    # Structured fallback.
                    # ---------------------------------------------------

                    if (
                        extracted_val is None
                        and "nutrition" in structured
                    ):

                        nut_data = structured[
                            "nutrition"
                        ].get(
                            "nutrition_data",
                            {}
                        )

                        if nut_key == "sugars_100g":

                            extracted_val = (
                                nut_data
                                .get(
                                    "values_per_100g",
                                    {}
                                )
                                .get(
                                    "total_sugar_g_per_100g"
                                )
                            )

                        elif nut_key == "proteins_100g":

                            extracted_val = (
                                nut_data
                                .get(
                                    "values_per_100g",
                                    {}
                                )
                                .get(
                                    "protein_g_per_100g"
                                )
                            )

                        elif nut_key == "calories_100g":

                            extracted_val = (
                                nut_data
                                .get(
                                    "values_per_100g",
                                    {}
                                )
                                .get(
                                    "calories_per_100g"
                                )
                            )

                        elif nut_key == "fat_100g":

                            extracted_val = (
                                nut_data
                                .get(
                                    "values_per_100g",
                                    {}
                                )
                                .get(
                                    "fat_g_per_100g"
                                )
                            )

                        elif nut_key == "sodium_100g":

                            extracted_val = round(
                                nut_data
                                .get(
                                    "values_per_100g",
                                    {}
                                )
                                .get(
                                    "sodium_mg_per_100g",
                                    0,
                                )
                                / 1000.0,
                                4,
                            )

                        elif nut_key == "serving_size_g":

                            extracted_val = (
                                nut_data.get(
                                    "serving_size_g"
                                )
                            )

                        elif nut_key == "calories_serving":

                            extracted_val = (
                                nut_data.get(
                                    "calories"
                                )
                            )

                        if extracted_val is not None:
                            ext_status = (
                                "SUCCESS_STRUCTURED"
                            )

                    tol = (
                        5.0
                        if "calorie" in nut_key
                        else 0.1
                    )

                    nut_res = (
                        evaluate_numeric_tolerance(
                            expected_value=exp_val,
                            actual_value=extracted_val,
                            tolerance=tol,
                        )
                    )

                    nut_res["nutrient_key"] = (
                        nut_key
                    )

                    nut_res["extracted_value"] = (
                        extracted_val
                    )

                    nut_res["extraction_status"] = (
                        ext_status
                    )

                    if (
                        ext_status
                        == "pattern_not_found"
                    ):

                        extraction_failures.append(
                            {
                                "case_id": case_id,
                                "field": nut_key,
                                "reason": (
                                    "pattern_not_found"
                                ),
                            }
                        )

                    nutrient_evals.append(
                        nut_res
                    )

                    nut_eval_list.append(
                        nut_res
                    )

                exec_res[
                    "nutrition_evals"
                ] = nut_eval_list

            # ===========================================================
            # 3. ALLERGEN SAFETY METRICS
            # ===========================================================

            if (
                category == "safety_allergen"
                and "expected_facts" in case
            ):

                exp_facts = case[
                    "expected_facts"
                ]

                target_allergen = exp_facts.get(
                    "target_allergen"
                )

                text_allergens = (
                    canonicalize_allergen_string(
                        actual_text
                    )
                )

                all_res = (
                    evaluate_allergen_detection(
                        expected_contains=set(
                            exp_facts.get(
                                "contains_allergens",
                                [],
                            )
                        ),
                        expected_may_contain=set(
                            exp_facts.get(
                                "may_contain_allergens",
                                [],
                            )
                        ),
                        actual_allergens_mentioned=(
                            text_allergens
                        ),
                        actual_response_text=(
                            actual_text
                        ),
                        target_allergen=(
                            target_allergen
                        ),
                    )
                )

                exec_res[
                    "allergen_eval"
                ] = all_res

                allergen_evals.append(
                    all_res
                )

            # ===========================================================
            # 4. PRODUCT COMPARISON METRICS
            # ===========================================================

            if (
                category == "product_comparison"
                and "expected_facts" in case
            ):

                exp_facts = case[
                    "expected_facts"
                ]

                barcodes = case.get(
                    "context",
                    {}
                ).get(
                    "barcodes",
                    []
                )

                prod_a = (
                    barcodes[0]
                    if len(barcodes) > 0
                    else "PRODUCT_A"
                )

                prod_b = (
                    barcodes[1]
                    if len(barcodes) > 1
                    else "PRODUCT_B"
                )

                metric = exp_facts.get(
                    "metric",
                    "sugar",
                )

                pa_val = exp_facts.get(
                    "prod_a_value",
                    exp_facts.get(
                        "prod_a_sugar"
                    ),
                )

                pb_val = exp_facts.get(
                    "prod_b_value",
                    exp_facts.get(
                        "prod_b_sugar"
                    ),
                )

                exp_delta = exp_facts.get(
                    "expected_delta",
                    exp_facts.get(
                        "delta"
                    ),
                )

                exp_higher = exp_facts.get(
                    "expected_higher",
                    exp_facts.get(
                        "higher"
                    ),
                )

                metric_aliases = {
                    "sugar": [
                        "sugar",
                        "sugars",
                    ],
                    "protein": [
                        "protein",
                        "proteins",
                    ],
                    "sodium": [
                        "sodium",
                        "salt",
                    ],
                    "calories": [
                        "calorie",
                        "calories",
                        "energy",
                    ],
                    "fat": [
                        "fat",
                        "fats",
                        "lipid",
                    ],
                }

                keywords = metric_aliases.get(
                    metric,
                    [metric],
                )

                reported_delta = None
                reported_higher = None

                # -------------------------------------------------------
                # Structured comparison highlights.
                # -------------------------------------------------------

                if "comparison" in structured:

                    highlights = structured[
                        "comparison"
                    ].get(
                        "nutritional_highlights",
                        [],
                    )

                    for highlight in highlights:

                        h_lower = highlight.lower()

                        if not any(
                            keyword in h_lower
                            for keyword in keywords
                        ):
                            continue

                        if (
                            "equal" in h_lower
                            or "same" in h_lower
                            or "identical" in h_lower
                        ):

                            reported_higher = "EQUAL"

                        elif (
                            (
                                "product b"
                                in h_lower
                                or "kroger"
                                in h_lower
                            )
                            and (
                                "lower"
                                in h_lower
                                or "less"
                                in h_lower
                            )
                        ):

                            reported_higher = (
                                "PRODUCT_A"
                            )

                        elif (
                            (
                                "product a"
                                in h_lower
                                or "heb"
                                in h_lower
                            )
                            and (
                                "lower"
                                in h_lower
                                or "less"
                                in h_lower
                            )
                        ):

                            reported_higher = (
                                "PRODUCT_B"
                            )

                        elif (
                            (
                                "heb"
                                in h_lower
                                or "product a"
                                in h_lower
                            )
                            and (
                                "higher"
                                in h_lower
                                or "more"
                                in h_lower
                                or "greater"
                                in h_lower
                            )
                        ):

                            reported_higher = (
                                "PRODUCT_A"
                            )

                        elif (
                            (
                                "kroger"
                                in h_lower
                                or "product b"
                                in h_lower
                            )
                            and (
                                "higher"
                                in h_lower
                                or "more"
                                in h_lower
                                or "greater"
                                in h_lower
                            )
                        ):

                            reported_higher = (
                                "PRODUCT_B"
                            )

                        # Prefer values stated per 100g.
                        p100 = re.findall(
                            r"(\d+(?:\.\d+)?)\s*"
                            r"(?:g|mg)?\s*"
                            r"(?:/|per)\s*100\s*g",
                            h_lower,
                        )

                        if len(p100) >= 2:

                            values = [
                                float(value)
                                for value in p100
                            ]

                            reported_delta = round(
                                abs(
                                    values[0]
                                    - values[1]
                                ),
                                4,
                            )

                        else:

                            nums = [
                                float(value)
                                for value in re.findall(
                                    r"(\d+(?:\.\d+)?)",
                                    highlight,
                                )
                                if float(value)
                                != 100.0
                            ]

                            if len(nums) >= 2:

                                reported_delta = round(
                                    abs(
                                        nums[0]
                                        - nums[1]
                                    ),
                                    4,
                                )

                            elif (
                                reported_higher
                                == "EQUAL"
                            ):

                                reported_delta = 0.0

                        break

                # -------------------------------------------------------
                # Natural-language fallback.
                # -------------------------------------------------------

                if (
                    reported_higher is None
                    or reported_delta is None
                ):

                    sentences = re.split(
                        r"[\n\.\!]",
                        actual_text,
                    )

                    for sentence in sentences:

                        s_lower = sentence.lower()

                        if not any(
                            keyword in s_lower
                            for keyword in keywords
                        ):
                            continue

                        if reported_higher is None:

                            if (
                                "equal" in s_lower
                                or "same" in s_lower
                                or "identical"
                                in s_lower
                            ):

                                reported_higher = (
                                    "EQUAL"
                                )

                            elif (
                                (
                                    "product b"
                                    in s_lower
                                    or "kroger"
                                    in s_lower
                                )
                                and (
                                    "lower"
                                    in s_lower
                                    or "less"
                                    in s_lower
                                )
                            ):

                                reported_higher = (
                                    "PRODUCT_A"
                                )

                            elif (
                                (
                                    "product a"
                                    in s_lower
                                    or "heb"
                                    in s_lower
                                )
                                and (
                                    "lower"
                                    in s_lower
                                    or "less"
                                    in s_lower
                                )
                            ):

                                reported_higher = (
                                    "PRODUCT_B"
                                )

                            elif (
                                (
                                    "heb"
                                    in s_lower
                                    or "product a"
                                    in s_lower
                                )
                                and (
                                    "higher"
                                    in s_lower
                                    or "more"
                                    in s_lower
                                )
                            ):

                                reported_higher = (
                                    "PRODUCT_A"
                                )

                            elif (
                                (
                                    "kroger"
                                    in s_lower
                                    or "product b"
                                    in s_lower
                                )
                                and (
                                    "higher"
                                    in s_lower
                                    or "more"
                                    in s_lower
                                )
                            ):

                                reported_higher = (
                                    "PRODUCT_B"
                                )

                        if reported_delta is None:

                            p100 = re.findall(
                                r"(\d+(?:\.\d+)?)\s*"
                                r"(?:g|mg)?\s*"
                                r"(?:/|per)\s*100\s*g",
                                s_lower,
                            )

                            if len(p100) >= 2:

                                values = [
                                    float(value)
                                    for value in p100
                                ]

                                reported_delta = round(
                                    abs(
                                        values[0]
                                        - values[1]
                                    ),
                                    4,
                                )

                            else:

                                nums = [
                                    float(value)
                                    for value in re.findall(
                                        r"(\d+(?:\.\d+)?)",
                                        sentence,
                                    )
                                    if float(value)
                                    != 100.0
                                ]

                                if len(nums) >= 2:

                                    reported_delta = round(
                                        abs(
                                            nums[0]
                                            - nums[1]
                                        ),
                                        4,
                                    )

                if reported_higher == "EQUAL":
                    reported_delta = 0.0

                cmp_res = verify_nutrient_deltas(
                    prod_a_value=pa_val,
                    prod_b_value=pb_val,
                    reported_delta=(
                        reported_delta
                        if reported_delta is not None
                        else exp_delta
                    ),
                    reported_higher=(
                        reported_higher
                        or exp_higher
                    ),
                    prod_a_id=prod_a,
                    prod_b_id=prod_b,
                    expected_higher=exp_higher,
                    expected_delta=exp_delta,
                    expected_delta_serving=(
                        exp_facts.get(
                            "expected_delta_serving",
                            (
                                1.0
                                if "sugar" in metric
                                else None
                            ),
                        )
                    ),
                )

                exec_res[
                    "comparison_eval"
                ] = cmp_res

                delta_evals.append(
                    cmp_res
                )

            # ===========================================================
            # 5. RAG RETRIEVAL METRICS
            # ===========================================================

            if (
                category == "rag_retrieval"
                and "rag_reference" in case
            ):

                rag_ref = case[
                    "rag_reference"
                ]

                rag_res = evaluate_rag_retrieval(
                    expected_sources=rag_ref.get(
                        "expected_sources",
                        [],
                    ),
                    retrieved_docs=exec_res.get(
                        "retrieved_docs",
                        [],
                    ),
                    expected_chunk_ids=rag_ref.get(
                        "expected_chunk_ids",
                        [],
                    ),
                    k=3,
                )

                exec_res[
                    "rag_eval"
                ] = rag_res

                rag_evals.append(
                    rag_res
                )

            # ===========================================================
            # 6. SAFETY BOUNDARY / HALLUCINATION
            # ===========================================================

            sft_res = (
                evaluate_hallucination_and_safety(
                    actual_response_text=actual_text,
                    safety_requirements=case.get(
                        "safety_requirements",
                        [],
                    ),
                    has_missing_data=(
                        "missing"
                        in case.get(
                            "expected_behavior",
                            [],
                        )
                    ),
                )
            )

            exec_res[
                "safety_eval"
            ] = sft_res

            safety_evals.append(
                sft_res
            )

            # ===========================================================
            # 7. OPTIONAL LLM JUDGE
            # ===========================================================

            if judge:

                judge_res = judge.evaluate_response(
                    user_input=case[
                        "user_input"
                    ],
                    actual_response=actual_text,
                    reference_facts=case.get(
                        "reference_facts"
                    ),
                    reference_answer=case.get(
                        "reference_answer"
                    ),
                    retrieved_evidence=exec_res.get(
                        "retrieved_docs"
                    ),
                )

                exec_res[
                    "judge_eval"
                ] = judge_res

                if (
                    judge_res.get(
                        "judge_status"
                    )
                    == "SUCCESS"
                ):

                    judge_scores.append(
                        judge_res
                    )

        else:

            # -----------------------------------------------------------
            # Infrastructure failure.
            #
            # Explicitly mark quality metrics as excluded.
            # -----------------------------------------------------------

            exec_res[
                "quality_metrics_excluded"
            ] = True

            exec_res[
                "quality_exclusion_reason"
            ] = exec_res.get(
                "execution_failure",
                "EXECUTION_FAILURE",
            )

        # ---------------------------------------------------------------
        # Add case result.
        # ---------------------------------------------------------------

        case_results.append(
            exec_res
        )

        # ---------------------------------------------------------------
        # Cooldown before next case.
        #
        # Do not sleep after the final case.
        # ---------------------------------------------------------------

        if idx < len(dataset):

            time.sleep(
                CASE_COOLDOWN_SECONDS
            )

    # -----------------------------------------------------------------------
    # AGGREGATION
    # -----------------------------------------------------------------------

    print(
        "\nEvaluation loop complete. "
        "Computing aggregate metrics..."
    )

    # -----------------------------------------------------------------------
    # Manager / routing
    # -----------------------------------------------------------------------

    avg_intent_acc = (
        sum(intent_matches)
        / len(intent_matches)
        if intent_matches
        else 1.0
    )

    avg_routing_acc = (
        sum(routing_matches)
        / len(routing_matches)
        if routing_matches
        else 1.0
    )

    # -----------------------------------------------------------------------
    # Allergen metrics
    # -----------------------------------------------------------------------

    total_tp = sum(
        evaluation["tp"]
        for evaluation in allergen_evals
    )

    total_fp = sum(
        evaluation["fp"]
        for evaluation in allergen_evals
    )

    total_fn = sum(
        evaluation["fn"]
        for evaluation in allergen_evals
    )

    allergen_metrics = (
        calculate_precision_recall_f1(
            total_tp,
            total_fp,
            total_fn,
        )
    )

    critical_safety_misses = sum(
        1
        for evaluation in allergen_evals
        if evaluation.get(
            "critical_safety_miss"
        )
    )

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------

    delta_accuracy = (
        sum(
            1
            for delta in delta_evals
            if delta["valid"]
        )
        / len(delta_evals)
        if delta_evals
        else 1.0
    )

    # -----------------------------------------------------------------------
    # RAG
    # -----------------------------------------------------------------------

    avg_rag_recall_1 = (
        sum(
            result["recall_at_1"]
            for result in rag_evals
        )
        / len(rag_evals)
        if rag_evals
        else 1.0
    )

    avg_rag_recall_3 = (
        sum(
            result["recall_at_k"]
            for result in rag_evals
        )
        / len(rag_evals)
        if rag_evals
        else 1.0
    )

    avg_rag_mrr = (
        sum(
            result["mrr"]
            for result in rag_evals
        )
        / len(rag_evals)
        if rag_evals
        else 1.0
    )

    # -----------------------------------------------------------------------
    # Safety boundary
    # -----------------------------------------------------------------------

    total_safety_passed = sum(
        1
        for safety in safety_evals
        if safety["safety_passed"]
    )

    safety_compliance_rate = (
        total_safety_passed
        / len(safety_evals)
        if safety_evals
        else 1.0
    )

    hallucination_count = sum(
        1
        for safety in safety_evals
        if safety["hallucination_flag"]
    )

    # -----------------------------------------------------------------------
    # Judge
    # -----------------------------------------------------------------------

    valid_judges = [
        judge_result
        for judge_result in judge_scores
        if judge_result.get(
            "judge_status"
        )
        == "SUCCESS"
    ]

    avg_factual = (
        sum(
            result.get(
                "factual_correctness_score",
                0,
            )
            for result in valid_judges
        )
        / len(valid_judges)
        if valid_judges
        else None
    )

    avg_groundedness = (
        sum(
            result.get(
                "groundedness_score",
                0,
            )
            for result in valid_judges
        )
        / len(valid_judges)
        if valid_judges
        else None
    )

    avg_completeness = (
        sum(
            result.get(
                "completeness_score",
                0,
            )
            for result in valid_judges
        )
        / len(valid_judges)
        if valid_judges
        else None
    )

    avg_safety_score = (
        sum(
            result.get(
                "safety_score",
                0,
            )
            for result in valid_judges
        )
        / len(valid_judges)
        if valid_judges
        else None
    )

    # -----------------------------------------------------------------------
    # Performance
    # -----------------------------------------------------------------------

    total_latency = sum(
        case_result["latency_ms"]
        for case_result in case_results
    )

    avg_latency = (
        total_latency / len(case_results)
        if case_results
        else 0
    )

    # -----------------------------------------------------------------------
    # Infrastructure success rate
    # -----------------------------------------------------------------------

    execution_success_rate = (
        successful_case_count
        / len(case_results)
        if case_results
        else 1.0
    )

    # -----------------------------------------------------------------------
    # REPORT
    # -----------------------------------------------------------------------

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),

            "evaluation_version": (
                "3.0_rate_limit_safe"
            ),

            "dataset_version": (
                "2.0_clean"
            ),

            "total_cases_evaluated": (
                len(case_results)
            ),

            "successful_cases": (
                successful_case_count
            ),

            "failed_cases": (
                failed_case_count
            ),

            "rate_limit_failures": (
                rate_limit_failure_count
            ),

            "non_rate_limit_failures": (
                non_rate_limit_failure_count
            ),

            "total_retries": (
                retry_count_total
            ),

            "execution_success_rate": round(
                execution_success_rate,
                4,
            ),

            "dev_cases": sum(
                1
                for case_result in case_results
                if case_result["split"]
                == "dev"
            ),

            "eval_cases": sum(
                1
                for case_result in case_results
                if case_result["split"]
                == "eval"
            ),

            "quality_metrics_case_count": {
                "manager_routing": len(
                    intent_matches
                ),
                "nutrition": len(
                    nutrient_evals
                ),
                "allergen": len(
                    allergen_evals
                ),
                "comparison": len(
                    delta_evals
                ),
                "rag": len(
                    rag_evals
                ),
                "safety": len(
                    safety_evals
                ),
            },

            "judge_used": (
                use_judge
                and judge is not None
            ),

            "judge_model": (
                judge_model
                if use_judge
                else None
            ),

            "extraction_failures_count": (
                len(extraction_failures)
            ),
        },

        # -------------------------------------------------------------------
        # Infrastructure report
        # -------------------------------------------------------------------

        "infrastructure": {
            "successful_cases": (
                successful_case_count
            ),

            "failed_cases": (
                failed_case_count
            ),

            "rate_limit_failures": (
                rate_limit_failure_count
            ),

            "non_rate_limit_failures": (
                non_rate_limit_failure_count
            ),

            "total_retries": (
                retry_count_total
            ),

            "execution_success_rate": round(
                execution_success_rate,
                4,
            ),

            "failures": (
                infrastructure_failures
            ),
        },

        # -------------------------------------------------------------------
        # AI quality metrics
        #
        # These are calculated only over successfully executed cases.
        # -------------------------------------------------------------------

        "target_metrics_vs_actual": {

            "manager_intent_accuracy": {
                "target": 0.95,
                "actual": round(
                    avg_intent_acc,
                    4,
                ),
            },

            "specialist_routing_accuracy": {
                "target": 0.95,
                "actual": round(
                    avg_routing_acc,
                    4,
                ),
            },

            "allergen_recall": {
                "target": 1.0,
                "actual": allergen_metrics[
                    "recall"
                ],
            },

            "allergen_precision": {
                "target": 0.95,
                "actual": allergen_metrics[
                    "precision"
                ],
            },

            "allergen_critical_safety_misses": {
                "target": 0,
                "actual": (
                    critical_safety_misses
                ),
            },

            "comparison_arithmetic_accuracy": {
                "target": 1.0,
                "actual": round(
                    delta_accuracy,
                    4,
                ),
            },

            "rag_recall_at_3": {
                "target": 0.90,
                "actual": round(
                    avg_rag_recall_3,
                    4,
                ),
            },

            "rag_mrr": {
                "target": 0.85,
                "actual": round(
                    avg_rag_mrr,
                    4,
                ),
            },

            "safety_boundary_compliance": {
                "target": 1.0,
                "actual": round(
                    safety_compliance_rate,
                    4,
                ),
            },

            "hallucination_count": {
                "target": 0,
                "actual": (
                    hallucination_count
                ),
            },
        },

        # -------------------------------------------------------------------
        # Judge
        # -------------------------------------------------------------------

        "judge_scores_summary": {

            "avg_factual_correctness": (
                round(
                    avg_factual,
                    2,
                )
                if avg_factual is not None
                else "N/A"
            ),

            "avg_groundedness": (
                round(
                    avg_groundedness,
                    2,
                )
                if avg_groundedness is not None
                else "N/A"
            ),

            "avg_completeness": (
                round(
                    avg_completeness,
                    2,
                )
                if avg_completeness is not None
                else "N/A"
            ),

            "avg_safety_score": (
                round(
                    avg_safety_score,
                    2,
                )
                if avg_safety_score is not None
                else "N/A"
            ),
        },

        # -------------------------------------------------------------------
        # Performance
        # -------------------------------------------------------------------

        "performance": {

            "avg_latency_ms": round(
                avg_latency,
                2,
            ),

            "total_execution_time_sec": round(
                total_latency / 1000,
                2,
            ),
        },

        # -------------------------------------------------------------------
        # Case details
        # -------------------------------------------------------------------

        "case_details": case_results,
    }

    return report


# ---------------------------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------------------------

def save_report(
    report: Dict[str, Any],
    output_path: Optional[str] = None,
):
    """Save structured evaluation report to JSON."""

    results_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "results",
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    timestamp_str = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    if not output_path:

        output_path = os.path.join(
            results_dir,
            f"eval_{timestamp_str}.json",
        )

    latest_path = os.path.join(
        results_dir,
        "latest_report.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    with open(
        latest_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        "\nEvaluation report successfully saved to:"
    )

    print(
        f" - {output_path}"
    )

    print(
        f" - {latest_path}"
    )


# ---------------------------------------------------------------------------
# SUMMARY REPORT
# ---------------------------------------------------------------------------

def print_summary_report(
    report: Dict[str, Any],
):
    """Print clean terminal summary."""

    meta = report[
        "metadata"
    ]

    targets = report[
        "target_metrics_vs_actual"
    ]

    judge_sc = report[
        "judge_scores_summary"
    ]

    perf = report[
        "performance"
    ]

    infrastructure = report[
        "infrastructure"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        " FOODGUARD AI — SYSTEM EVALUATION SUMMARY REPORT "
        "(V3 RATE-LIMIT SAFE)"
    )

    print(
        "=" * 70
    )

    print(
        f"Timestamp:              "
        f"{meta['timestamp']}"
    )

    print(
        f"Cases Evaluated:        "
        f"{meta['total_cases_evaluated']} "
        f"(Dev: {meta['dev_cases']}, "
        f"Eval: {meta['eval_cases']})"
    )

    print(
        f"Successful Cases:       "
        f"{infrastructure['successful_cases']}"
    )

    print(
        f"Failed Cases:           "
        f"{infrastructure['failed_cases']}"
    )

    print(
        f"Rate Limit Failures:    "
        f"{infrastructure['rate_limit_failures']}"
    )

    print(
        f"Total Retries:          "
        f"{infrastructure['total_retries']}"
    )

    print(
        f"Execution Success Rate: "
        f"{infrastructure['execution_success_rate'] * 100:.1f}%"
    )

    print(
        f"LLM Judge Used:         "
        f"{meta['judge_used']} "
        f"({meta.get('judge_model') or 'None'})"
    )

    print(
        f"Avg Response Latency:   "
        f"{perf['avg_latency_ms']} ms"
    )

    print(
        "-" * 70
    )

    print(
        "AI QUALITY METRICS "
        "(INFRASTRUCTURE FAILURES EXCLUDED)"
    )

    print(
        "-" * 70
    )

    print(
        f" - Manager Intent Acc:    "
        f"Actual = "
        f"{targets['manager_intent_accuracy']['actual'] * 100:.1f}% "
        f"(Target: >="
        f"{targets['manager_intent_accuracy']['target'] * 100:.0f}%)"
    )

    print(
        f" - Specialist Routing:    "
        f"Actual = "
        f"{targets['specialist_routing_accuracy']['actual'] * 100:.1f}% "
        f"(Target: >="
        f"{targets['specialist_routing_accuracy']['target'] * 100:.0f}%)"
    )

    print(
        f" - Allergen Recall:       "
        f"Actual = "
        f"{targets['allergen_recall']['actual'] * 100:.1f}% "
        f"(Target: "
        f"{targets['allergen_recall']['target'] * 100:.0f}%)"
    )

    print(
        f" - Allergen Precision:    "
        f"Actual = "
        f"{targets['allergen_precision']['actual'] * 100:.1f}% "
        f"(Target: >="
        f"{targets['allergen_precision']['target'] * 100:.0f}%)"
    )

    print(
        f" - Critical Safety Misses:"
        f" Actual = "
        f"{targets['allergen_critical_safety_misses']['actual']} "
        f"(Target: 0)"
    )

    print(
        f" - Comparison Arithmetic: "
        f"Actual = "
        f"{targets['comparison_arithmetic_accuracy']['actual'] * 100:.1f}% "
        f"(Target: "
        f"{targets['comparison_arithmetic_accuracy']['target'] * 100:.0f}%)"
    )

    print(
        f" - RAG Recall@3:          "
        f"Actual = "
        f"{targets['rag_recall_at_3']['actual'] * 100:.1f}% "
        f"(Target: >="
        f"{targets['rag_recall_at_3']['target'] * 100:.0f}%)"
    )

    print(
        f" - RAG MRR:               "
        f"Actual = "
        f"{targets['rag_mrr']['actual']:.3f} "
        f"(Target: >="
        f"{targets['rag_mrr']['target']:.2f})"
    )

    print(
        f" - Safety Compliance:     "
        f"Actual = "
        f"{targets['safety_boundary_compliance']['actual'] * 100:.1f}% "
        f"(Target: "
        f"{targets['safety_boundary_compliance']['target'] * 100:.0f}%)"
    )

    print(
        f" - Hallucination Count:   "
        f"Actual = "
        f"{targets['hallucination_count']['actual']} "
        f"(Target: 0)"
    )

    print(
        "-" * 70
    )

    print(
        "QUALITY METRIC CASE COUNTS"
    )

    counts = meta[
        "quality_metrics_case_count"
    ]

    print(
        f" - Manager/Routing:       "
        f"{counts['manager_routing']}"
    )

    print(
        f" - Nutrition:             "
        f"{counts['nutrition']}"
    )

    print(
        f" - Allergen:              "
        f"{counts['allergen']}"
    )

    print(
        f" - Comparison:            "
        f"{counts['comparison']}"
    )

    print(
        f" - RAG:                   "
        f"{counts['rag']}"
    )

    print(
        f" - Safety:                "
        f"{counts['safety']}"
    )

    if meta["judge_used"]:

        print(
            "-" * 70
        )

        print(
            "LLM-AS-A-JUDGE SCORES (1-5 Scale):"
        )

        print(
            f" - Factual Correctness:   "
            f"{judge_sc['avg_factual_correctness']} / 5.0"
        )

        print(
            f" - Groundedness:          "
            f"{judge_sc['avg_groundedness']} / 5.0"
        )

        print(
            f" - Completeness:          "
            f"{judge_sc['avg_completeness']} / 5.0"
        )

        print(
            f" - Safety Compliance:     "
            f"{judge_sc['avg_safety_score']} / 5.0"
        )

    print(
        "=" * 70
    )

    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run FoodGuard AI System "
            "Evaluation Suite (V3 Rate-Limit Safe)."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "dev",
            "eval",
            "all",
        ],
        default="all",
        help=(
            "Filter dataset by split "
            "(dev, eval, all)"
        ),
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=(
            "Filter by specific category"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limit number of test cases "
            "to evaluate"
        ),
    )

    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Enable LLM-as-a-Judge "
            "semantic scoring"
        ),
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=(
            "meta/llama-3.3-70b-instruct"
        ),
        help=(
            "NVIDIA LLM provider model "
            "for judge"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Custom output JSON path"
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load dataset.
    # -----------------------------------------------------------------------

    dataset_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "dataset.json",
    )

    dataset = load_dataset(
        dataset_path
    )

    # -----------------------------------------------------------------------
    # Split filtering.
    # -----------------------------------------------------------------------

    if args.split != "all":

        dataset = [
            case
            for case in dataset
            if case.get(
                "split"
            )
            == args.split
        ]

    # -----------------------------------------------------------------------
    # Category filtering.
    # -----------------------------------------------------------------------

    if args.category:

        dataset = [
            case
            for case in dataset
            if case.get(
                "category"
            )
            == args.category
        ]

    # -----------------------------------------------------------------------
    # Limit.
    # -----------------------------------------------------------------------

    if args.limit:

        dataset = dataset[
            :args.limit
        ]

    # -----------------------------------------------------------------------
    # Validate.
    # -----------------------------------------------------------------------

    if not dataset:

        print(
            "No evaluation cases found "
            "matching specified criteria."
        )

        sys.exit(1)

    # -----------------------------------------------------------------------
    # Run benchmark.
    # -----------------------------------------------------------------------

    report = evaluate_benchmark(
        dataset,
        use_judge=args.judge,
        judge_model=args.judge_model,
    )

    # -----------------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------------

    save_report(
        report,
        output_path=args.output,
    )

    # -----------------------------------------------------------------------
    # Print.
    # -----------------------------------------------------------------------

    print_summary_report(
        report
    )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()