import json
import logging
import re
import uuid

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.database.connection import AsyncSessionFactory, init_db_tables
from app.database.crud import (
    add_message,
    create_conversation,
    get_conversation,
    get_conversation_messages,
    update_conversation_context,
)
from app.graph.workflow import workflow
from app.llm.provider import LLMConfigurationError
from app.schemas import ChatRequest, ChatResponse
from app.tools.ocr import (
    ImageValidationError,
    MAX_IMAGE_SIZE,
    MAX_IMAGES,
    OCRStatus,
    process_images,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="FoodGuard AI API")


def _validation_detail(error: ValidationError) -> list[dict[str, object]]:
    return [
        {
            "loc": item["loc"],
            "msg": item["msg"],
            "type": item["type"],
        }
        for item in error.errors()
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "FoodGuard AI API",
    }


@app.get("/conversations/{conversation_id}")
async def fetch_conversation(conversation_id: str) -> dict[str, object]:
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError as err:
        raise HTTPException(
            status_code=400,
            detail="Invalid conversation ID format.",
        ) from err

    try:
        async with AsyncSessionFactory() as db_session:
            conv = await get_conversation(db_session, conv_uuid)

            if not conv:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

            msgs = await get_conversation_messages(
                db_session,
                conv_uuid,
            )

            return {
                "conversation_id": str(conv.id),
                "active_barcode": conv.active_barcode,
                "active_comparison_barcodes": (
                    conv.active_comparison_barcodes or []
                ),
                "ocr_combined_text": conv.ocr_combined_text,
                "ocr_nutrition_text": conv.ocr_nutrition_text,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "created_at": (
                            m.created_at.isoformat()
                            if m.created_at
                            else None
                        ),
                        "sequence_number": m.sequence_number,
                    }
                    for m in msgs
                ],
            }

    except HTTPException:
        raise

    except Exception as err:
        logger.error(
            "Failed to fetch conversation: %s",
            err,
        )

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve conversation.",
        ) from err


def _safe_error_message(error: Exception) -> str:
    message = str(error)

    message = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+",
        r"\1[REDACTED]",
        message,
    )

    message = re.sub(
        r"(?i)(api[\s_-]?key\s*[:=]\s*)[^\s,}]+",
        r"\1[REDACTED]",
        message,
    )

    return message.replace("\n", " ")


async def _parse_chat_request(
    request: Request,
) -> tuple[ChatRequest, list[tuple[bytes, str, str | None]]]:

    content_type = request.headers.get(
        "content-type",
        "",
    )

    # -----------------------------------------
    # Normal JSON request
    # -----------------------------------------
    if not content_type.startswith("multipart/form-data"):
        try:
            return ChatRequest.model_validate(
                await request.json()
            ), []

        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=400,
                detail="Invalid chat request.",
            ) from error

        except ValidationError as error:
            raise HTTPException(
                status_code=422,
                detail=_validation_detail(error),
            ) from error

    # -----------------------------------------
    # Multipart request with optional images
    # -----------------------------------------
    form = await request.form()

    raw_messages = form.get(
        "messages",
        "[]",
    )

    raw_nutrition = form.get(
        "nutrition_data",
    )

    raw_barcodes = form.get(
        "barcodes",
    )

    try:
        messages = (
            json.loads(raw_messages)
            if isinstance(raw_messages, str)
            else []
        )

        nutrition_data = (
            json.loads(raw_nutrition)
            if isinstance(raw_nutrition, str)
            and raw_nutrition
            else None
        )

        barcodes = (
            json.loads(raw_barcodes)
            if isinstance(raw_barcodes, str)
            and raw_barcodes
            else [
                item
                for item in form.getlist("barcodes")
                if isinstance(item, str)
            ]
        )

        payload = ChatRequest.model_validate(
            {
                "message": form.get("message"),
                "messages": messages,
                "ingredient_text": form.get("ingredient_text"),
                "allergen_statement": form.get(
                    "allergen_statement"
                ),
                "cross_contact_statement": form.get(
                    "cross_contact_statement"
                ),
                "nutrition_data": nutrition_data,

                # Existing OCR context from previous
                # conversation turns.
                "ocr_combined_text": form.get(
                    "ocr_combined_text"
                ),
                "ocr_nutrition_text": form.get(
                    "ocr_nutrition_text"
                ),

                "barcode": form.get("barcode"),
                "barcodes": barcodes,
                "conversation_id": form.get(
                    "conversation_id"
                ),
            }
        )

    except (
        TypeError,
        json.JSONDecodeError,
    ) as error:

        raise HTTPException(
            status_code=400,
            detail="Invalid multipart chat request.",
        ) from error

    except ValidationError as error:

        raise HTTPException(
            status_code=422,
            detail=_validation_detail(error),
        ) from error

    uploads = [
        item
        for item in form.getlist("images")
        if isinstance(
            item,
            (
                UploadFile,
                StarletteUploadFile,
            ),
        )
    ]

    if len(uploads) > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                "A maximum of 5 images can be "
                "uploaded at a time."
            ),
        )

    image_data: list[
        tuple[bytes, str, str | None]
    ] = []

    for upload in uploads:

        data = await upload.read(
            MAX_IMAGE_SIZE + 1
        )

        if len(data) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Each image must be 10 MB or smaller."
                ),
            )

        image_data.append(
            (
                data,
                upload.filename or "uploaded-image",
                upload.content_type,
            )
        )

    return payload, image_data


def _serialize_ocr_results(
    ocr_result,
) -> list[dict[str, object]]:
    """
    Preserve each uploaded image's OCR result separately.

    This is important for image-to-image comparison:

        image 1 -> Product A
        image 2 -> Product B

    The existing combined OCR text is still preserved
    separately for backward compatibility.
    """

    if not ocr_result:
        return []

    serialized_results: list[
        dict[str, object]
    ] = []

    for image_result in ocr_result.images:

        item: dict[str, object] = {
            "index": image_result.index,
            "filename": image_result.filename,
            "text": image_result.text,
            "confidence": image_result.confidence,
            "status": (
                image_result.status.value
                if hasattr(
                    image_result.status,
                    "value",
                )
                else str(image_result.status)
            ),
            "warning": image_result.warning,
            "sections": {
                "ingredients_text": (
                    image_result.sections.ingredients_text
                ),
                "nutrition_text": (
                    image_result.sections.nutrition_text
                ),
                "allergen_text": (
                    image_result.sections.allergen_text
                ),
                "serving_text": (
                    image_result.sections.serving_text
                ),
                "other_text": (
                    image_result.sections.other_text
                ),
                "nutrition_data": (
                    image_result.sections.nutrition_data.model_dump()
                    if image_result.sections.nutrition_data
                    else None
                ),
            },
        }

        serialized_results.append(item)

    return serialized_results


@app.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
async def chat(request: Request) -> ChatResponse:

    request_data, image_data = await _parse_chat_request(
        request
    )

    # --------------------------------------------------
    # Phase 17 Conversation UUID Resolution
    # --------------------------------------------------
    conv_uuid = None

    if request_data.conversation_id:
        try:
            conv_uuid = uuid.UUID(
                request_data.conversation_id
            )
        except ValueError:
            conv_uuid = uuid.uuid4()
    else:
        conv_uuid = uuid.uuid4()

    # --------------------------------------------------
    # OCR
    # --------------------------------------------------
    ocr_result = None

    if image_data:

        try:
            ocr_result = process_images(
                image_data
            )

        except ImageValidationError as error:

            raise HTTPException(
                status_code=error.status_code,
                detail=str(error),
            ) from error

        if (
            ocr_result.status
            == OCRStatus.OCR_UNAVAILABLE
        ):

            raise HTTPException(
                status_code=503,
                detail=(
                    "OCR is unavailable. Install "
                    "Tesseract or configure "
                    "TESSERACT_CMD."
                ),
            )

    # --------------------------------------------------
    # Preserve individual OCR results.
    #
    # This is the new information needed by the
    # image-to-image comparison workflow.
    # --------------------------------------------------
    ocr_results = _serialize_ocr_results(
        ocr_result
    )

    # --------------------------------------------------
    # Product/OCR context resolution
    #
    # New OCR data takes priority.
    # If no new image was uploaded, reuse the OCR
    # context supplied by the frontend from the
    # previous turn.
    # --------------------------------------------------

    ingredient_data = (
        (
            ocr_result.sections.ingredients_text
            or ocr_result.combined_text
        )
        if ocr_result
        else (
            request_data.ingredient_text
            or request_data.ocr_combined_text
        )
    )

    allergen_statement = (
        ocr_result.sections.allergen_text
        if ocr_result
        else request_data.allergen_statement
    )

    cross_contact_statement = (
        request_data.cross_contact_statement
    )

    if (
        ocr_result
        and ocr_result.sections.allergen_text
        and "may contain"
        in ocr_result.sections.allergen_text.lower()
    ):
        cross_contact_statement = (
            cross_contact_statement
            or ocr_result.sections.allergen_text
        )

    current_ocr_combined_text = (
        ocr_result.combined_text
        if ocr_result
        else request_data.ocr_combined_text
    )

    current_ocr_nutrition_text = (
        ocr_result.sections.nutrition_text
        if ocr_result
        else request_data.ocr_nutrition_text
    )

    # --------------------------------------------------
    # Phase 15 Active Context Hydration Precedence Rules
    # --------------------------------------------------

    message_barcodes = re.findall(
        r"\b\d{8,14}\b",
        request_data.message or "",
    )

    if ocr_result:

        active_single_barcode = None
        active_comparison_barcodes = []

    elif len(message_barcodes) >= 2:

        active_single_barcode = None
        active_comparison_barcodes = (
            message_barcodes[:2]
        )

        current_ocr_combined_text = None
        current_ocr_nutrition_text = None

    elif len(message_barcodes) == 1:

        active_single_barcode = (
            message_barcodes[0]
        )

        active_comparison_barcodes = []

        current_ocr_combined_text = None
        current_ocr_nutrition_text = None

    else:

        active_single_barcode = (
            request_data.barcode
        )

        active_comparison_barcodes = (
            request_data.barcodes or []
        )

    # --------------------------------------------------
    # Phase 17 PostgreSQL Conversation & User Message
    # Ingest
    # --------------------------------------------------

    try:

        async with AsyncSessionFactory() as db_session:

            db_conv = await get_conversation(
                db_session,
                conv_uuid,
            )

            if not db_conv:

                db_conv = await create_conversation(
                    db_session,
                    conversation_id=conv_uuid,
                    active_barcode=(
                        active_single_barcode
                    ),
                    active_comparison_barcodes=(
                        active_comparison_barcodes
                    ),
                    ocr_combined_text=(
                        current_ocr_combined_text
                    ),
                    ocr_nutrition_text=(
                        current_ocr_nutrition_text
                    ),
                )

            else:

                if (
                    not active_single_barcode
                    and db_conv.active_barcode
                ):
                    active_single_barcode = (
                        db_conv.active_barcode
                    )

                if (
                    not active_comparison_barcodes
                    and db_conv.active_comparison_barcodes
                ):
                    active_comparison_barcodes = (
                        db_conv.active_comparison_barcodes
                        or []
                    )

                if (
                    not current_ocr_combined_text
                    and db_conv.ocr_combined_text
                ):
                    current_ocr_combined_text = (
                        db_conv.ocr_combined_text
                    )

                if (
                    not current_ocr_nutrition_text
                    and db_conv.ocr_nutrition_text
                ):
                    current_ocr_nutrition_text = (
                        db_conv.ocr_nutrition_text
                    )

            if request_data.message:

                await add_message(
                    db_session,
                    conv_uuid,
                    role="user",
                    content=request_data.message,
                )

    except Exception as db_err:

        logger.warning(
            "PostgreSQL ingest persistence warning: %s",
            db_err,
        )

    # --------------------------------------------------
    # Build conversation messages
    # --------------------------------------------------

    messages = [
        SystemMessage(
            content=(
                "You are FoodGuard AI, a helpful "
                "conversational food-analysis assistant. "
                "For now, answer general food and nutrition "
                "questions clearly. Do not diagnose medical "
                "conditions or provide medical treatment. "
                "When product-specific data is not available, "
                "say so instead of inventing information."
            ),
        ),
    ]

    messages.extend(
        {
            "user": HumanMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
        }[message.role](
            content=message.content
        )
        for message in request_data.messages
    )

    if request_data.message:

        last_message = (
            request_data.messages[-1]
            if request_data.messages
            else None
        )

        if (
            not last_message
            or last_message.content
            != request_data.message
        ):

            messages.append(
                HumanMessage(
                    content=request_data.message
                )
            )

    current_nutrition_data = (
        request_data.nutrition_data
        or (
            ocr_result.nutrition_data
            if ocr_result
            else None
        )
    )

    # --------------------------------------------------
    # LangGraph Execution
    # --------------------------------------------------

    try:

        result = workflow.invoke(
            {
                "messages": messages,

                # Phase 9 & Phase 15 barcode/product
                # lookup request
                "product_lookup_request": (
                    active_single_barcode
                ),
                "product_data": None,
                "product_lookup_error": None,

                # OCR context
                "ocr_combined_text": (
                    current_ocr_combined_text
                ),

                # NEW:
                # Preserve individual OCR results so
                # comparison can distinguish Product A
                # and Product B.
                "ocr_results": ocr_results,

                "ocr_status": (
                    ocr_result.status.value
                    if ocr_result
                    else None
                ),

                "ocr_warnings": (
                    ocr_result.warnings
                    if ocr_result
                    else []
                ),

                "ocr_nutrition_text": (
                    current_ocr_nutrition_text
                ),

                # Ingredient context
                "ingredient_data": ingredient_data,
                "ingredient_analysis": None,

                # Nutrition context
                "nutrition_data": (
                    current_nutrition_data
                ),
                "nutrition_analysis": None,

                # Safety context
                "safety_data": {
                    "ingredient_text": (
                        ingredient_data
                    ),
                    "allergen_statement": (
                        allergen_statement
                    ),
                    "cross_contact_statement": (
                        cross_contact_statement
                    ),
                },
                "safety_analysis": None,

                # RAG context
                "rag_context": None,
                "rag_sources": None,

                # Phase 12 & Phase 15 Product Comparison
                "comparison_barcodes": (
                    active_comparison_barcodes
                    if len(
                        active_comparison_barcodes
                    ) >= 2
                    else None
                ),
                "comparison_products": None,
                "comparison_analysis": None,

                # Recommendation context
                "recommendation_analysis": None,

                # Verification context
                "verification_analysis": None,
            }
        )

        response = result["messages"][-1].content

        if not isinstance(response, str):

            raise RuntimeError(
                "LangGraph returned an invalid "
                "assistant response."
            )

    except LLMConfigurationError as error:

        logger.error(
            "NVIDIA configuration failed: type=%s message=%s",
            type(error).__name__,
            _safe_error_message(error),
        )

        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception as error:

        logger.error(
            "NVIDIA LLM request failed: type=%s message=%s",
            type(error).__name__,
            _safe_error_message(error),
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "NVIDIA LLM is currently unavailable."
            ),
        ) from error

    # --------------------------------------------------
    # Return response + reusable conversation context
    # --------------------------------------------------

    res_product_data = result.get(
        "product_data"
    )

    res_barcode = (
        res_product_data.barcode
        if (
            res_product_data
            and hasattr(
                res_product_data,
                "barcode",
            )
            and res_product_data.barcode
        )
        else active_single_barcode
    )

    res_comp_barcodes = (
        result.get(
            "comparison_barcodes"
        )
        if (
            result.get(
                "comparison_barcodes"
            )
            and len(
                result.get(
                    "comparison_barcodes"
                )
            ) >= 2
        )
        else (
            active_comparison_barcodes
            if len(
                active_comparison_barcodes
            ) >= 2
            else None
        )
    )

    # --------------------------------------------------
    # Phase 17 PostgreSQL Response & Active Context
    # Commit
    # --------------------------------------------------

    try:

        async with AsyncSessionFactory() as db_session:

            await add_message(
                db_session,
                conv_uuid,
                role="assistant",
                content=response,
            )

            await update_conversation_context(
                db_session,
                conversation_id=conv_uuid,
                active_barcode=res_barcode,
                active_comparison_barcodes=(
                    res_comp_barcodes
                ),
                ocr_combined_text=(
                    current_ocr_combined_text
                ),
                ocr_nutrition_text=(
                    current_ocr_nutrition_text
                ),
            )

    except Exception as db_err:

        logger.warning(
            "PostgreSQL commit persistence warning: %s",
            db_err,
        )

    return ChatResponse(
        response=response,

        ocr_status=(
            ocr_result.status.value
            if ocr_result
            else None
        ),

        ocr_warnings=(
            ocr_result.warnings
            if ocr_result
            else None
        ),

        ocr_combined_text=(
            current_ocr_combined_text
        ),

        ocr_nutrition_text=(
            current_ocr_nutrition_text
        ),

        ingredient_text=ingredient_data,

        allergen_statement=(
            allergen_statement
        ),

        cross_contact_statement=(
            cross_contact_statement
        ),

        nutrition_data=current_nutrition_data,

        barcode=res_barcode,

        comparison_barcodes=res_comp_barcodes,

        conversation_id=str(conv_uuid),
    )