from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

from app import config
from app.agents.ingredient import (
    IngredientAnalysis,
    analyze_ingredients,
    missing_ingredient_analysis,
    parse_ingredient_analysis,
)
from app.agents.manager import Intent, classify_intent, parse_manager_decision
from app.agents.nutrition import (
    NutritionAnalysis,
    analyze_nutrition,
    missing_nutrition_analysis,
    parse_nutrition_analysis,
)
from app.agents.safety import (
    OverallSafetyStatus,
    SafetyAnalysis,
    analyze_safety,
    missing_safety_analysis,
    parse_safety_analysis,
)
from app.graph.nodes import ingredient_node, manager_node, nutrition_node, response_node
from app.graph.nodes import safety_node
from app.graph.workflow import build_workflow
from app.main import app
from app.llm import provider
from app.tools.nutrition import (
    NutritionData,
    calculate_percentage_difference,
    compare_nutrients,
    normalize_nutrient_per_100g,
    normalize_nutrient_per_serving,
)
from app.tools.barcode import InvalidBarcodeError, normalize_barcode
from app.tools.product_lookup import (
    ProductLookupError,
    ProductDataQuality,
    lookup_product_by_barcode,
)
from app.tools import product_lookup
from app.tools.safety import (
    AllergenStatus,
    classify_allergen_status,
    detect_cross_contact_statement,
    detect_explicit_allergens,
)


client = TestClient(app)


def test_health_still_works() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "FoodGuard AI API"}


def test_chat_requires_message_or_history() -> None:
    response = client.post("/chat", json={})

    assert response.status_code == 422
    assert "Provide a message or conversation history." in response.text


def test_missing_nvidia_configuration_returns_useful_error(monkeypatch) -> None:
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")

    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "NVIDIA_API_KEY is not configured."}


def test_provider_can_be_mocked_without_network(monkeypatch) -> None:
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(provider.config, "NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(provider.config, "NVIDIA_MODEL", "test-model")
    monkeypatch.setattr(provider.config, "NVIDIA_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(provider, "OpenAI", FakeClient)

    provider.NVIDIAProvider()

    assert calls == [{"api_key": "test-key", "base_url": "https://example.test/v1"}]


@pytest.mark.parametrize(
    ("user_message", "intent"),
    [
        ("What is protein?", Intent.GENERAL_QUESTION),
        ("Analyze this chips packet.", Intent.PRODUCT_ANALYSIS),
        ("What do these ingredients mean?", Intent.INGREDIENT_ANALYSIS),
        ("How much sugar does it contain?", Intent.NUTRITION_ANALYSIS),
        ("Does it contain allergens?", Intent.SAFETY_ALLERGEN),
        ("Compare these two products.", Intent.PRODUCT_COMPARISON),
        ("Which product should I choose?", Intent.RECOMMENDATION),
    ],
)
def test_manager_classifies_supported_intents(user_message, intent) -> None:
    class FakeProvider:
        def generate_response(self, messages):
            assert user_message in messages[-1]["content"]
            return f'{{"intent":"{intent.value}","confidence":0.92,"reason":"mocked"}}'

    decision = classify_intent(
        [{"role": "user", "content": user_message}],
        provider=FakeProvider(),
    )

    assert decision.intent == intent
    assert decision.confidence == 0.92


def test_invalid_manager_output_falls_back_to_unknown() -> None:
    decision = parse_manager_decision("not valid JSON")

    assert decision.intent == Intent.UNKNOWN
    assert decision.confidence == 0.0


def test_ingredient_agent_returns_valid_structured_analysis() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            return (
                '{"available":true,"ingredients":['
                '{"name":"sunflower oil","role":"oil/fat source",'
                '"explanation":"A plant-based oil used to provide fat and texture.","attention":false},'
                '{"name":"salt","role":"salt/mineral",'
                '"explanation":"Used for flavor; high intake may deserve attention.","attention":true}'
                '],"allergens":[],"uncertain_items":[],"summary":"A simple snack ingredient list."}'
            )

    analysis = analyze_ingredients(
        "potato, sunflower oil, salt",
        [{"role": "user", "content": "Explain these ingredients."}],
        provider=FakeProvider(),
    )

    assert isinstance(analysis, IngredientAnalysis)
    assert [item.name for item in analysis.ingredients] == ["sunflower oil", "salt"]
    assert analysis.ingredients[0].role == "oil/fat source"
    assert analysis.ingredients[1].attention is True


def test_missing_ingredient_data_is_handled_without_provider_call() -> None:
    class FailingProvider:
        def generate_response(self, messages):
            raise AssertionError("provider should not be called without ingredient data")

    analysis = analyze_ingredients(None, [], provider=FailingProvider())

    assert analysis == missing_ingredient_analysis()
    assert analysis.available is False
    assert "ingredient list" in analysis.summary


def test_invalid_ingredient_output_returns_controlled_uncertainty() -> None:
    analysis = parse_ingredient_analysis("not valid structured output")

    assert analysis.available is True
    assert analysis.ingredients == []
    assert analysis.uncertain_items


def test_ingredient_node_stores_analysis_in_graph_state() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            return '{"available":true,"ingredients":[],"allergens":[],"uncertain_items":[],"summary":"Reviewed."}'

    result = ingredient_node(
        {
            "messages": [HumanMessage(content="Explain these ingredients.")],
            "ingredient_data": "potato, salt",
        },
        provider=FakeProvider(),
    )

    assert result["ingredient_analysis"].summary == "Reviewed."


def test_general_question_does_not_route_to_ingredient_node() -> None:
    calls = []

    class FakeProvider:
        def generate_response(self, messages):
            calls.append(messages)
            if len(calls) == 1:
                return '{"intent":"GENERAL_QUESTION","confidence":0.9,"reason":"general"}'
            return "General response."

    result = build_workflow(provider=FakeProvider()).invoke(
        {
            "messages": [HumanMessage(content="What is protein?")],
            "ingredient_data": "potato, salt",
            "ingredient_analysis": None,
        }
    )

    assert result["ingredient_analysis"] is None
    assert result["messages"][-1].content == "General response."
    assert len(calls) == 2


def test_ingredient_intent_routes_through_ingredient_node() -> None:
    calls = []

    class FakeProvider:
        def generate_response(self, messages):
            calls.append(messages)
            if len(calls) == 1:
                return '{"intent":"INGREDIENT_ANALYSIS","confidence":0.95,"reason":"ingredient request"}'
            if len(calls) == 2:
                return '{"available":true,"ingredients":[],"allergens":[],"uncertain_items":[],"summary":"Ingredient data reviewed."}'
            return "The ingredient data was reviewed."

    result = build_workflow(provider=FakeProvider()).invoke(
        {
            "messages": [HumanMessage(content="Explain these ingredients.")],
            "ingredient_data": "potato, salt",
            "ingredient_analysis": None,
        }
    )

    assert result["ingredient_analysis"].summary == "Ingredient data reviewed."
    assert result["messages"][-1].content == "The ingredient data was reviewed."
    assert len(calls) == 3


def test_ingredient_response_node_uses_analysis() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            assert "sunflower oil" in messages[0]["content"]
            return "Sunflower oil is an oil/fat source used for texture and cooking."

    result = response_node(
        {
            "messages": [HumanMessage(content="What does sunflower oil do?")],
            "manager_decision": None,
            "ingredient_analysis": IngredientAnalysis(
                ingredients=[
                    {
                        "name": "sunflower oil",
                        "role": "oil/fat source",
                        "explanation": "A plant-based oil.",
                    }
                ],
                summary="Reviewed ingredient.",
            ),
        },
        provider=FakeProvider(),
    )

    assert result["messages"][-1].content.startswith("Sunflower oil")


def test_chat_passes_ingredient_text_into_graph_state(monkeypatch) -> None:
    captured_state = {}

    class FakeWorkflow:
        def invoke(self, state):
            captured_state.update(state)
            return {"messages": [*state["messages"], AIMessage(content="Reviewed.")]}

    monkeypatch.setattr("app.main.workflow", FakeWorkflow())

    response = client.post(
        "/chat",
        json={
            "message": "Explain these ingredients.",
            "ingredient_text": "potato, sunflower oil, salt",
        },
    )

    assert response.status_code == 200
    assert captured_state["ingredient_data"] == "potato, sunflower oil, salt"


def test_normalize_per_serving_to_per_100g() -> None:
    assert normalize_nutrient_per_100g(3, 30) == 10
    assert normalize_nutrient_per_100g(4, 30) == pytest.approx(13.3333333)


def test_normalize_per_100g_to_per_serving() -> None:
    assert normalize_nutrient_per_serving(10, 30) == 3


def test_nutrition_tools_handle_zero_missing_and_invalid_values() -> None:
    assert normalize_nutrient_per_100g(3, 0) is None
    assert normalize_nutrient_per_100g(-1, 30) is None
    assert normalize_nutrient_per_100g("invalid", 30) is None
    assert normalize_nutrient_per_serving(None, 30) is None
    assert calculate_percentage_difference(0, 10) is None
    assert calculate_percentage_difference(None, 10) is None


def test_compare_nutrients_handles_values_and_missing_values() -> None:
    first = NutritionData(protein_g=3)
    second = NutritionData(protein_g=5)
    assert compare_nutrients(first, second, "protein_g") == {
        "first": 3,
        "second": 5,
        "difference": 2,
        "higher": "second",
    }
    assert compare_nutrients(NutritionData(), second, "protein_g")["higher"] == "unknown"


def test_calculate_percentage_difference() -> None:
    assert calculate_percentage_difference(4, 5) == 25


def test_nutrition_agent_returns_valid_analysis_and_calculated_values() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            assert '"protein_g":3.0' in messages[-1]["content"]
            assert 'protein_g_per_100g' in messages[-1]["content"]
            return '{"available":true,"normalized_values":{},"summary":"The label lists protein.","uncertain_items":[]}'

    analysis = analyze_nutrition(
        NutritionData(serving_size_g=30, protein_g=3, total_sugar_g=4),
        [{"role": "user", "content": "How much protein is there?"}],
        provider=FakeProvider(),
    )

    assert isinstance(analysis, NutritionAnalysis)
    assert analysis.normalized_values["protein_g_per_100g"] == 10
    assert analysis.normalized_values["total_sugar_g_per_100g"] == pytest.approx(13.3333333)


def test_missing_nutrition_data_is_handled_without_provider_call() -> None:
    class FailingProvider:
        def generate_response(self, messages):
            raise AssertionError("provider should not be called without nutrition data")

    analysis = analyze_nutrition(None, [], provider=FailingProvider())

    assert analysis == missing_nutrition_analysis()
    assert analysis.available is False
    assert "nutrition-label" in analysis.summary


def test_invalid_nutrition_output_keeps_calculated_values_and_marks_uncertainty() -> None:
    analysis = parse_nutrition_analysis("invalid", {"protein_g_per_100g": 10})

    assert analysis.normalized_values == {"protein_g_per_100g": 10}
    assert analysis.uncertain_items


def test_nutrition_node_stores_analysis_in_graph_state() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            return '{"available":true,"normalized_values":{},"summary":"Reviewed nutrition.","uncertain_items":[]}'

    result = nutrition_node(
        {
            "messages": [HumanMessage(content="How much protein?")],
            "nutrition_data": NutritionData(serving_size_g=30, protein_g=3),
        },
        provider=FakeProvider(),
    )

    assert result["nutrition_analysis"].summary == "Reviewed nutrition."
    assert result["nutrition_analysis"].normalized_values["protein_g_per_100g"] == 10


def test_nutrition_intent_routes_through_nutrition_node() -> None:
    calls = []

    class FakeProvider:
        def generate_response(self, messages):
            calls.append(messages)
            if len(calls) == 1:
                return '{"intent":"NUTRITION_ANALYSIS","confidence":0.95,"reason":"nutrition request"}'
            if len(calls) == 2:
                return '{"available":true,"normalized_values":{},"summary":"Nutrition reviewed.","uncertain_items":[]}'
            return "The product provides 3 g of protein per serving."

    result = build_workflow(provider=FakeProvider()).invoke(
        {
            "messages": [HumanMessage(content="How much protein?")],
            "nutrition_data": NutritionData(serving_size_g=30, protein_g=3),
            "nutrition_analysis": None,
        }
    )

    assert result["nutrition_analysis"].summary == "Nutrition reviewed."
    assert result["messages"][-1].content.startswith("The product provides")
    assert len(calls) == 3


def test_nutrition_response_node_uses_analysis() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            assert "protein_g_per_100g" in messages[0]["content"]
            return "This provides 3 g of protein per serving."

    result = response_node(
        {
            "messages": [HumanMessage(content="How much protein?")],
            "manager_decision": None,
            "nutrition_analysis": NutritionAnalysis(
                normalized_values={"protein_g_per_100g": 10},
                summary="Protein is listed.",
            ),
        },
        provider=FakeProvider(),
    )

    assert result["messages"][-1].content.startswith("This provides")


def test_safety_tools_detect_explicit_allergens() -> None:
    findings = detect_explicit_allergens(
        "wheat flour, milk powder, sugar, soy lecithin",
        "Contains: wheat, milk, soy",
    )

    assert findings["wheat"]["status"] == AllergenStatus.CONFIRMED.value
    assert findings["milk"]["status"] == AllergenStatus.CONFIRMED.value
    assert findings["soy"]["status"] == AllergenStatus.CONFIRMED.value


def test_safety_tools_distinguish_not_found_and_unknown() -> None:
    complete_findings = classify_allergen_status("potato, salt", None)
    unknown_findings = classify_allergen_status(None, None)

    assert complete_findings["milk"]["status"] == AllergenStatus.NOT_FOUND.value
    assert complete_findings["peanut"]["status"] == AllergenStatus.NOT_FOUND.value
    assert unknown_findings["milk"]["status"] == AllergenStatus.UNKNOWN.value


def test_cross_contact_is_separate_from_confirmed_allergen() -> None:
    statement = "May contain peanuts. Manufactured in a facility that also processes tree nuts."

    findings = detect_explicit_allergens(None, statement)
    warnings = detect_cross_contact_statement(statement)

    assert findings["peanut"]["status"] == AllergenStatus.UNKNOWN.value
    assert len(warnings) == 2


def test_safety_agent_returns_valid_analysis() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            assert "Contains: milk, soy" in messages[-1]["content"]
            return (
                '{"available":true,"allergens":[],"cross_contact_warnings":[],'
                '"unknown_items":[],"overall_status":"ALLERGEN_CONFIRMED",'
                '"summary":"Milk and soy are listed."}'
            )

    analysis = analyze_safety(
        "wheat flour, milk powder, sugar, soy lecithin",
        "Contains: milk, soy",
        None,
        [],
        provider=FakeProvider(),
    )

    assert isinstance(analysis, SafetyAnalysis)
    assert analysis.overall_status == OverallSafetyStatus.ALLERGEN_CONFIRMED
    assert {item.name for item in analysis.allergens if item.status == AllergenStatus.CONFIRMED} >= {
        "wheat",
        "milk",
        "soy",
    }


def test_missing_safety_data_is_handled_without_provider_call() -> None:
    class FailingProvider:
        def generate_response(self, messages):
            raise AssertionError("provider should not be called without safety data")

    analysis = analyze_safety(None, None, None, [], provider=FailingProvider())

    assert analysis == missing_safety_analysis()
    assert analysis.available is False
    assert analysis.overall_status == OverallSafetyStatus.INSUFFICIENT_INFORMATION


def test_invalid_safety_output_keeps_deterministic_findings() -> None:
    deterministic = SafetyAnalysis(
        allergens=[
            {"name": "milk", "status": AllergenStatus.CONFIRMED, "evidence": "Contains: milk"}
        ],
        summary="Checked.",
        overall_status=OverallSafetyStatus.ALLERGEN_CONFIRMED,
    )

    analysis = parse_safety_analysis("invalid", deterministic)

    assert analysis.allergens[0].status == AllergenStatus.CONFIRMED
    assert analysis.unknown_items
    assert analysis.overall_status == OverallSafetyStatus.REVIEW_REQUIRED


def test_safety_node_stores_analysis_in_graph_state() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            return '{"available":true,"allergens":[],"cross_contact_warnings":[],"unknown_items":[],"overall_status":"CLEAR_FROM_SUPPLIED_INFO","summary":"Checked."}'

    result = safety_node(
        {
            "messages": [HumanMessage(content="Does this contain allergens?")],
            "safety_data": {"ingredient_text": "potato, salt"},
        },
        provider=FakeProvider(),
    )

    assert result["safety_analysis"].summary == "Checked."
    assert result["safety_analysis"].allergens[0].status == AllergenStatus.NOT_FOUND


def test_safety_intent_routes_through_safety_node() -> None:
    calls = []

    class FakeProvider:
        def generate_response(self, messages):
            calls.append(messages)
            if len(calls) == 1:
                return '{"intent":"SAFETY_ALLERGEN","confidence":0.95,"reason":"allergen request"}'
            if len(calls) == 2:
                return '{"available":true,"allergens":[],"cross_contact_warnings":[],"unknown_items":[],"overall_status":"ALLERGEN_CONFIRMED","summary":"Safety checked."}'
            return "Milk is confirmed in the supplied information."

    result = build_workflow(provider=FakeProvider()).invoke(
        {
            "messages": [HumanMessage(content="Does this product contain allergens?")],
            "safety_data": {"ingredient_text": "wheat flour, milk powder"},
            "safety_analysis": None,
        }
    )

    assert result["safety_analysis"].summary == "Safety checked."
    assert result["messages"][-1].content.startswith("Milk is confirmed")
    assert len(calls) == 3


def test_safety_response_node_uses_analysis() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            assert "CONFIRMED" in messages[0]["content"]
            return "Milk is confirmed in the supplied label information."

    result = response_node(
        {
            "messages": [HumanMessage(content="Does this contain milk?")],
            "manager_decision": None,
            "safety_analysis": SafetyAnalysis(
                allergens=[
                    {"name": "milk", "status": AllergenStatus.CONFIRMED, "evidence": "Contains: milk"}
                ],
                overall_status=OverallSafetyStatus.ALLERGEN_CONFIRMED,
                summary="Checked.",
            ),
        },
        provider=FakeProvider(),
    )

    assert result["messages"][-1].content.startswith("Milk is confirmed")


def test_chat_passes_safety_data_into_graph_state(monkeypatch) -> None:
    captured_state = {}

    class FakeWorkflow:
        def invoke(self, state):
            captured_state.update(state)
            return {"messages": [*state["messages"], AIMessage(content="Checked.")]}

    monkeypatch.setattr("app.main.workflow", FakeWorkflow())

    response = client.post(
        "/chat",
        json={
            "message": "Does this contain milk?",
            "ingredient_text": "wheat flour, milk powder",
            "allergen_statement": "Contains: wheat, milk",
            "cross_contact_statement": "May contain peanuts.",
        },
    )

    assert response.status_code == 200
    assert captured_state["safety_data"] == {
        "ingredient_text": "wheat flour, milk powder",
        "allergen_statement": "Contains: wheat, milk",
        "cross_contact_statement": "May contain peanuts.",
    }


def test_barcode_validation_and_normalization() -> None:
    assert normalize_barcode("034000470693") == "0034000470693"
    assert normalize_barcode("3017624010701") == "3017624010701"
    with pytest.raises(InvalidBarcodeError):
        normalize_barcode("not-a-barcode")


def test_product_lookup_successfully_normalizes_open_food_facts(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "status": 1,
                "product": {
                    "code": "3017624010701",
                    "product_name": "Test Spread",
                    "brands": "Test Brand",
                    "ingredients_text": "sugar, milk powder, hazelnuts",
                    "allergens": "en:milk, en:nuts",
                    "allergens_tags": ["en:milk", "en:nuts"],
                    "traces_tags": ["en:peanuts"],
                    "nutriments": {
                        "energy-kcal_100g": 539,
                        "proteins_100g": 6.3,
                        "sodium_100g": 0.1,
                    },
                    "serving_size": "15 g",
                    "categories": "en:spreads",
                    "labels": "en:vegetarian",
                    "image_front_url": "https://example.test/image.jpg",
                },
            }

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params, headers):
            assert url.endswith("/product/3017624010701.json")
            assert "fields" in params
            assert headers["User-Agent"]
            return FakeResponse()

    monkeypatch.setattr(product_lookup.httpx, "Client", FakeClient)
    result = lookup_product_by_barcode("3017624010701")

    assert result.found is True
    assert result.product.product_name == "Test Spread"
    assert result.product.brand == "Test Brand"
    assert result.product.data_quality == ProductDataQuality.COMPLETE
    assert result.product.nutrition.values_per_100g["sodium_mg_per_100g"] == 100
    assert result.product.traces == ["peanuts"]


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(404, ProductLookupError.PRODUCT_NOT_FOUND), (503, ProductLookupError.UPSTREAM_UNAVAILABLE)],
)
def test_product_lookup_handles_http_errors(monkeypatch, status_code, expected) -> None:
    class FakeResponse:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            return {"status": 0}

    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def get(self, *args, **kwargs): return FakeResponse()

    monkeypatch.setattr(product_lookup.httpx, "Client", FakeClient)
    result = lookup_product_by_barcode("3017624010701")

    assert result.found is False
    assert result.error == expected


def test_product_lookup_handles_timeout_invalid_json_and_partial_data(monkeypatch) -> None:
    class TimeoutClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def get(self, *args, **kwargs): raise product_lookup.httpx.TimeoutException("timeout")

    monkeypatch.setattr(product_lookup.httpx, "Client", TimeoutClient)
    assert lookup_product_by_barcode("3017624010701").error == ProductLookupError.UPSTREAM_TIMEOUT

    class InvalidResponse:
        status_code = 200
        def json(self): raise ValueError("bad json")

    class InvalidJsonClient(TimeoutClient):
        def get(self, *args, **kwargs): return InvalidResponse()

    monkeypatch.setattr(product_lookup.httpx, "Client", InvalidJsonClient)
    assert lookup_product_by_barcode("3017624010701").error == ProductLookupError.INVALID_UPSTREAM_RESPONSE

    class PartialResponse:
        status_code = 200
        def json(self): return {"status": 1, "product": {"product_name": "Partial"}}

    class PartialClient(TimeoutClient):
        def get(self, *args, **kwargs): return PartialResponse()

    monkeypatch.setattr(product_lookup.httpx, "Client", PartialClient)
    result = lookup_product_by_barcode("3017624010701")
    assert result.found is True
    assert result.product.data_quality == ProductDataQuality.PARTIAL
    assert result.product.nutrition is None


def test_product_lookup_invalid_barcode_is_controlled() -> None:
    result = lookup_product_by_barcode("bad")

    assert result.found is False
    assert result.error == ProductLookupError.INVALID_BARCODE


def test_chat_passes_barcode_into_graph_state(monkeypatch) -> None:
    captured_state = {}

    class FakeWorkflow:
        def invoke(self, state):
            captured_state.update(state)
            return {"messages": [*state["messages"], AIMessage(content="Product checked.")]}

    monkeypatch.setattr("app.main.workflow", FakeWorkflow())
    response = client.post(
        "/chat",
        json={"message": "Analyze this product.", "barcode": "3017624010701"},
    )

    assert response.status_code == 200
    assert captured_state["product_lookup_request"] == "3017624010701"


def test_manager_node_stores_decision_and_response_node_adds_assistant() -> None:
    class FakeProvider:
        def generate_response(self, messages):
            if messages[0]["role"] == "system" and "Manager Agent" in messages[0]["content"]:
                return '{"intent":"NUTRITION_ANALYSIS","confidence":0.9,"reason":"sugar question"}'
            return "I need the product nutrition information for a specific assessment."

    provider_instance = FakeProvider()
    state = {"messages": [HumanMessage(content="How much sugar does it contain?")]}

    manager_result = manager_node(state, provider=provider_instance)
    state.update(manager_result)
    response_result = response_node(state, provider=provider_instance)

    assert state["manager_decision"].intent == Intent.NUTRITION_ANALYSIS
    assert response_result["messages"] == [
        AIMessage(content="I need the product nutrition information for a specific assessment.")
    ]


def test_workflow_can_be_constructed_and_invoked_with_mocked_provider() -> None:
    calls = []

    class FakeProvider:
        def generate_response(self, messages):
            calls.append(messages)
            if len(calls) == 1:
                return '{"intent":"GENERAL_QUESTION","confidence":0.8,"reason":"general"}'
            return "Mocked assistant response."

    compiled_workflow = build_workflow(provider=FakeProvider())
    result = compiled_workflow.invoke(
        {"messages": [HumanMessage(content="Hello")], "manager_decision": None}
    )

    assert result["manager_decision"].intent == Intent.GENERAL_QUESTION
    assert result["messages"][-1].content == "Mocked assistant response."
    assert len(calls) == 2


def test_chat_uses_langgraph_workflow_and_forwards_history(monkeypatch) -> None:
    captured_messages = []

    class FakeWorkflow:
        def invoke(self, state):
            captured_messages.extend(state["messages"])
            return {
                "messages": [
                    *state["messages"],
                    AIMessage(content="Protein supports growth and repair."),
                ],
                "manager_decision": {
                    "intent": Intent.GENERAL_QUESTION,
                    "confidence": 0.9,
                    "reason": "general question",
                },
            }

    monkeypatch.setattr("app.main.workflow", FakeWorkflow())

    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "What is protein?"},
                {"role": "assistant", "content": "It is a nutrient."},
            ],
            "message": "Why is it important?",
        },
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Protein supports growth and repair."
    assert captured_messages[0].type == "system"
    assert captured_messages[-1] == HumanMessage(content="Why is it important?")