import io

import pytest
from PIL import Image

from app.main import app
from app.tools import ocr
from app.tools.ocr import (
    ImageValidationError,
    OCRStatus,
    extract_label_sections,
    normalize_ocr_text,
    process_image_bytes,
    process_images,
)
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


client = TestClient(app)


def image_bytes(format_name="PNG") -> bytes:
    image = Image.new("RGB", (80, 40), "white")
    output = io.BytesIO()
    image.save(output, format=format_name)
    return output.getvalue()


def test_valid_image_decodes_and_ocr_success_is_normalized(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "_configure_tesseract", lambda: None)
    monkeypatch.setattr(ocr.pytesseract, "image_to_string", lambda image: " Ingredients: potato, salt\n\n")
    monkeypatch.setattr(ocr.pytesseract, "image_to_data", lambda image, output_type: {"conf": ["90", "80"]})

    result = process_image_bytes(image_bytes(), "label.png", "image/png")

    assert result.status == OCRStatus.SUCCESS
    assert result.text == "Ingredients: potato, salt"
    assert result.confidence == 0.85
    assert result.sections.ingredients_text == "Ingredients: potato, salt"


def test_invalid_and_unsupported_images_are_rejected() -> None:
    with pytest.raises(ImageValidationError) as invalid:
        process_image_bytes(b"not an image", "bad.png", "image/png")
    assert invalid.value.code == "INVALID_IMAGE"

    with pytest.raises(ImageValidationError) as unsupported:
        process_image_bytes(image_bytes("BMP"), "bad.bmp", "image/bmp")
    assert unsupported.value.code == "UNSUPPORTED_IMAGE"


def test_oversized_and_too_many_images_are_rejected(monkeypatch) -> None:
    with pytest.raises(ImageValidationError) as oversized:
        process_image_bytes(b"x" * (ocr.MAX_IMAGE_SIZE + 1), "large.png", "image/png")
    assert oversized.value.code == "IMAGE_TOO_LARGE"

    monkeypatch.setattr(ocr, "process_image_bytes", lambda *args, **kwargs: ocr.OCRImageResult(
        index=kwargs.get("index", 0), filename="x.png", status=OCRStatus.NO_TEXT
    ))
    with pytest.raises(ImageValidationError) as too_many:
        process_images([(b"x", "x.png", "image/png")] * 6)
    assert too_many.value.code == "TOO_MANY_IMAGES"


def test_ocr_unavailable_failure_no_text_and_partial_multi_image(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "_configure_tesseract", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    unavailable = process_image_bytes(image_bytes(), "one.png", "image/png")
    assert unavailable.status == OCRStatus.OCR_UNAVAILABLE

    monkeypatch.setattr(ocr, "_configure_tesseract", lambda: None)
    monkeypatch.setattr(ocr.pytesseract, "image_to_string", lambda image: "")
    monkeypatch.setattr(ocr.pytesseract, "image_to_data", lambda image, output_type: {"conf": []})
    no_text = process_image_bytes(image_bytes(), "empty.png", "image/png")
    assert no_text.status == OCRStatus.NO_TEXT

    calls = {"count": 0}
    def mixed_ocr(image):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("mock OCR failure")
        return "Ingredients: potato, salt"
    monkeypatch.setattr(ocr.pytesseract, "image_to_string", mixed_ocr)
    monkeypatch.setattr(ocr.pytesseract, "image_to_data", lambda image, output_type: {"conf": ["80"]})
    batch = process_images([(image_bytes(), "one.png", "image/png"), (image_bytes(), "two.png", "image/png")])
    assert batch.status == OCRStatus.PARTIAL
    assert batch.images[0].status == OCRStatus.SUCCESS
    assert batch.images[1].status == OCRStatus.OCR_FAILED
    assert batch.combined_text == "Ingredients: potato, salt"


def test_normalization_and_label_section_extraction() -> None:
    text = "Ingredients: potato, salt\nNutrition Facts\nServing size 30 g\nCalories 150\nProtein 3 g\nContains: milk"
    sections = extract_label_sections(text)

    assert normalize_ocr_text("a   b\n\n\n c") == "a b\n\nc"
    assert sections.ingredients_text
    assert sections.nutrition_text
    assert sections.serving_text == "Serving size 30 g"
    assert sections.allergen_text == "Contains: milk"
    assert sections.nutrition_data.protein_g == 3
    assert sections.nutrition_data.serving_size_g == 30


def test_multipart_chat_passes_ocr_sections_to_graph(monkeypatch) -> None:
    captured_state = {}

    class FakeWorkflow:
        def invoke(self, state):
            captured_state.update(state)
            return {"messages": [*state["messages"], AIMessage(content="OCR received.")]}

    monkeypatch.setattr("app.main.workflow", FakeWorkflow())
    monkeypatch.setattr(
        "app.main.process_images",
        lambda images: ocr.OCRBatchResult(
            images=[],
            combined_text="Ingredients: potato, salt",
            status=OCRStatus.SUCCESS,
            images_processed=1,
            sections=ocr.LabelSections(ingredients_text="Ingredients: potato, salt"),
        ),
    )

    response = client.post(
        "/chat",
        data={"message": "Explain these ingredients.", "messages": "[]"},
        files={"images": ("label.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["ocr_status"] == "SUCCESS"
    assert captured_state["ingredient_data"] == "Ingredients: potato, salt"
    assert "ocr_result" not in captured_state
    assert not any(isinstance(value, bytes) for value in captured_state.values())