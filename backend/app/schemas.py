from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.tools.nutrition import NutritionData


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str | None = Field(default=None, min_length=1)
    messages: list[ChatMessage] = Field(default_factory=list)

    # Product/ingredient context
    ingredient_text: str | None = Field(default=None, min_length=1)

    # Allergen and cross-contact context
    allergen_statement: str | None = Field(default=None, min_length=1)
    cross_contact_statement: str | None = Field(default=None, min_length=1)

    # Structured nutrition information
    nutrition_data: NutritionData | None = None

    # OCR context retained by the current conversation
    ocr_combined_text: str | None = Field(default=None, min_length=1)
    ocr_nutrition_text: str | None = Field(default=None, min_length=1)

    # Optional barcode(s) for Phase 9 product lookup & Phase 12 product comparison
    barcode: str | None = Field(default=None, min_length=1)
    barcodes: list[str] = Field(default_factory=list)

    # Ephemeral conversation identifier for Phase 15
    conversation_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_message_or_history(self) -> "ChatRequest":
        if not self.message and not self.messages:
            raise ValueError("Provide a message or conversation history.")
        return self


class ChatResponse(BaseModel):
    response: str

    # OCR processing information
    ocr_status: str | None = None
    ocr_warnings: list[str] | None = None

    # OCR context returned to the frontend so it can be reused
    # for follow-up questions.
    ocr_combined_text: str | None = None
    ocr_nutrition_text: str | None = None

    # Structured product information returned for conversation context
    ingredient_text: str | None = None
    allergen_statement: str | None = None
    cross_contact_statement: str | None = None
    nutrition_data: NutritionData | None = None

    # Phase 9 & Phase 15 active barcode context
    barcode: str | None = None

    # Phase 12 & Phase 15 active product comparison barcodes
    comparison_barcodes: list[str] | None = None

    # Ephemeral conversation identifier for Phase 15
    conversation_id: str | None = None
