from __future__ import annotations

import io
import re
from enum import Enum
from typing import Iterable

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel, Field

from app import config
from app.tools.nutrition import NutritionData


MAX_IMAGES = 5
MAX_IMAGE_SIZE = 10 * 1024 * 1024

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class OCRStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    NO_TEXT = "NO_TEXT"
    OCR_UNAVAILABLE = "OCR_UNAVAILABLE"
    INVALID_IMAGE = "INVALID_IMAGE"
    OCR_FAILED = "OCR_FAILED"


class LabelSections(BaseModel):
    ingredients_text: str | None = None
    nutrition_text: str | None = None
    allergen_text: str | None = None
    serving_text: str | None = None
    other_text: str | None = None
    nutrition_data: NutritionData | None = None


class OCRImageResult(BaseModel):
    index: int
    filename: str
    text: str = ""
    confidence: float | None = None
    status: OCRStatus
    warning: str | None = None
    sections: LabelSections = Field(default_factory=LabelSections)


class OCRBatchResult(BaseModel):
    images: list[OCRImageResult] = Field(default_factory=list)
    combined_text: str = ""
    status: OCRStatus
    confidence: float | None = None
    images_processed: int = 0
    warnings: list[str] = Field(default_factory=list)
    sections: LabelSections = Field(default_factory=LabelSections)
    nutrition_data: NutritionData | None = None


class ImageValidationError(ValueError):
    """Raised when an upload violates the image input contract."""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def normalize_ocr_text(text: str) -> str:
    """
    Normalize OCR text while preserving line boundaries.

    Nutrition labels depend heavily on row structure, so we do not
    collapse the entire OCR result into one line.
    """
    normalized_lines: list[str] = []

    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


def _section_block(
    lines: list[str],
    start: int,
    stop_words: tuple[str, ...],
) -> str | None:
    collected: list[str] = []

    for line in lines[start:]:
        lowered = line.lower()

        if collected and any(
            re.search(
                word,
                lowered,
                flags=re.IGNORECASE,
            )
            for word in stop_words
        ):
            break

        collected.append(line)

    result = " ".join(collected).strip()

    return result or None


def _extract_serving_size(text: str) -> float | None:
    patterns = (
        r"serving\s+size[^\n]{0,120}?\(\s*(\d+(?:\.\d+)?)\s*g\s*\)",
        r"serving\s+size[^\n]{0,120}?\b(\d+(?:\.\d+)?)\s*g\b",
        r"serv\.?\s*size[^\n]{0,120}?\(\s*(\d+(?:\.\d+)?)\s*g\s*\)",
        r"serv\.?\s*size[^\n]{0,120}?\b(\d+(?:\.\d+)?)\s*g\b",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def _extract_label_value(
    text: str,
    labels: tuple[str, ...],
    units: tuple[str, ...] = (),
) -> float | None:
    """
    Extract a value from the same OCR line as its label.

    This intentionally avoids searching across unrelated lines.
    """

    for line in text.splitlines():
        normalized = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        for label in labels:
            pattern = (
                rf"\b{label}\b"
                rf"[^\d\n]{{0,30}}"
                rf"(\d+(?:\.\d+)?)"
            )

            match = re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            try:
                value = float(match.group(1))
            except ValueError:
                continue

            if not units:
                return value

            after = normalized[match.end() :]

            unit_after = any(
                re.search(
                    rf"\b{re.escape(unit)}\b",
                    after,
                    flags=re.IGNORECASE,
                )
                for unit in units
            )

            before = normalized[: match.start()]

            unit_before = any(
                re.search(
                    rf"\b{re.escape(unit)}\b",
                    before,
                    flags=re.IGNORECASE,
                )
                for unit in units
            )

            if unit_after or unit_before:
                return value

    return None


def _extract_calories(text: str) -> float | None:
    """
    Calories usually have no unit.
    Only inspect lines containing a calories label.
    """

    for line in text.splitlines():
        if not re.search(
            r"\bcalories?\b",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        match = re.search(
            r"\bcalories?\b[^\d\n]{0,30}(\d+(?:\.\d+)?)",
            line,
            flags=re.IGNORECASE,
        )

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None

    return None


def extract_nutrition_data(
    text: str | None,
) -> NutritionData | None:
    """
    Extract nutrition information conservatively.

    No value is guessed when OCR is ambiguous.
    """

    if not text:
        return None

    serving_size = _extract_serving_size(text)

    calories = _extract_calories(text)

    fat_g = _extract_label_value(
        text,
        (
            "total fat",
            "total fats",
        ),
        ("g",),
    )

    saturated_fat_g = _extract_label_value(
        text,
        (
            "saturated fat",
            "saturated fats",
            "sat. fat",
            "sat fat",
        ),
        ("g",),
    )

    trans_fat_g = _extract_label_value(
        text,
        (
            "trans fat",
            "trans fats",
        ),
        ("g",),
    )

    carbohydrates_g = _extract_label_value(
        text,
        (
            "total carbohydrates",
            "total carbohydrate",
            "total carb",
            "total carbs",
            "carbohydrates",
            "carbohydrate",
        ),
        ("g",),
    )

    fiber_g = _extract_label_value(
        text,
        (
            "dietary fiber",
            "dietary fibre",
            "fiber",
            "fibre",
        ),
        ("g",),
    )

    added_sugar_g = _extract_label_value(
        text,
        (
            "added sugars",
            "added sugar",
        ),
        ("g",),
    )

    total_sugar_g = _extract_label_value(
        text,
        (
            "total sugars",
            "total sugar",
            "sugars",
            "sugar",
        ),
        ("g",),
    )

    protein_g = _extract_label_value(
        text,
        (
            "protein",
        ),
        ("g",),
    )

    sodium_mg = _extract_label_value(
        text,
        (
            "sodium",
        ),
        ("mg",),
    )

    cholesterol_mg = _extract_label_value(
        text,
        (
            "cholesterol",
            "cholest",
        ),
        ("mg",),
    )

    fields = {
        "calories": calories,
        "fat_g": fat_g,
        "saturated_fat_g": saturated_fat_g,
        "trans_fat_g": trans_fat_g,
        "carbohydrates_g": carbohydrates_g,
        "fiber_g": fiber_g,
        "total_sugar_g": total_sugar_g,
        "added_sugar_g": added_sugar_g,
        "protein_g": protein_g,
        "sodium_mg": sodium_mg,
        "cholesterol_mg": cholesterol_mg,
    }

    if serving_size is None and not any(
        value is not None
        for value in fields.values()
    ):
        return None

    return NutritionData(
        serving_size_g=serving_size,
        **fields,
    )


def _extract_ingredients(
    lines: list[str],
) -> str | None:
    """
    Extract ingredients even when OCR slightly corrupts
    the word 'INGREDIENTS'.
    """

    patterns = (
        r"\bingredients?\b",
        r"\bingredient\b",
        r"\bwgredients\b",
        r"\bingredients?\s*:",
    )

    for index, line in enumerate(lines):
        if not any(
            re.search(
                pattern,
                line,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        ):
            continue

        cleaned = re.sub(
            r"^\s*(?:w)?ingredients?\s*[:\-]?\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()

        if cleaned:
            return cleaned

        if index + 1 < len(lines):
            next_line = lines[index + 1].strip()

            if next_line:
                return next_line

    return None


def _extract_allergen_statement(
    lines: list[str],
) -> str | None:
    for line in lines:
        if re.search(
            r"\bcontains\b|\bmay contain\b|\ballergen\b|\ballergens\b|\btraces?\b",
            line,
            flags=re.IGNORECASE,
        ):
            return line.strip()

    return None


def _find_nutrition_lines(
    lines: list[str],
) -> list[str]:
    """
    Find nutrition rows even when OCR misses the Nutrition Facts heading.
    """

    keywords = (
        "calories",
        "calorie",
        "total fat",
        "fat",
        "sat fat",
        "saturated fat",
        "trans fat",
        "cholesterol",
        "cholest",
        "sodium",
        "total carb",
        "total carbohydrates",
        "carbohydrate",
        "carbohydrates",
        "dietary fiber",
        "fiber",
        "fibre",
        "total sugars",
        "total sugar",
        "added sugars",
        "added sugar",
        "sugars",
        "sugar",
        "protein",
        "serving size",
        "serv. size",
    )

    nutrition_lines: list[str] = []

    for line in lines:
        lowered = line.lower()

        if any(
            keyword in lowered
            for keyword in keywords
        ):
            nutrition_lines.append(line)

    return nutrition_lines


def extract_label_sections(
    text: str,
) -> LabelSections:
    """
    Extract ingredients, allergens, serving size, nutrition rows,
    and remaining OCR text.
    """

    normalized = normalize_ocr_text(text)

    if not normalized:
        return LabelSections()

    lines = [
        line
        for line in normalized.splitlines()
        if line
    ]

    ingredients = _extract_ingredients(lines)

    allergen = _extract_allergen_statement(lines)

    serving = None

    for line in lines:
        if re.search(
            r"\bserving\s+size\b|\bserv\.?\s*size\b",
            line,
            flags=re.IGNORECASE,
        ):
            serving = line
            break

    nutrition = None

    for index, line in enumerate(lines):
        if re.search(
            r"nutrition\s+(facts|information)|nutritional\s+information",
            line,
            flags=re.IGNORECASE,
        ):
            nutrition = _section_block(
                lines,
                index,
                (
                    r"\bingredients?\b",
                    r"\ballergens?\b",
                    r"\bcontains\b",
                    r"\bmay contain\b",
                ),
            )
            break

    if not nutrition:
        nutrition_lines = _find_nutrition_lines(lines)

        if nutrition_lines:
            nutrition = "\n".join(
                nutrition_lines
            )

    nutrition_data = extract_nutrition_data(
        nutrition or normalized
    )

    consumed_lines: set[str] = set()

    if ingredients:
        for line in lines:
            if (
                ingredients in line
                or line in ingredients
            ):
                consumed_lines.add(line)

    if allergen:
        consumed_lines.add(allergen)

    if serving:
        consumed_lines.add(serving)

    if nutrition:
        for line in nutrition.splitlines():
            consumed_lines.add(line.strip())

    other_lines = [
        line
        for line in lines
        if line not in consumed_lines
    ]

    other = "\n".join(
        other_lines
    ).strip() or None

    return LabelSections(
        ingredients_text=ingredients,
        nutrition_text=nutrition,
        allergen_text=allergen,
        serving_text=serving,
        other_text=other,
        nutrition_data=nutrition_data,
    )


def _preprocess_variants(
    image: Image.Image,
) -> list[Image.Image]:
    """
    Generate several OCR-friendly versions of the same image.

    Different label photographs need different preprocessing.
    """

    image = ImageOps.exif_transpose(
        image
    ).convert("L")

    minimum_width = 1800

    if image.width < minimum_width:
        scale = minimum_width / image.width

        image = image.resize(
            (
                minimum_width,
                round(
                    image.height * scale
                ),
            )
        )

    variants: list[Image.Image] = []

    # Variant 1: grayscale + contrast
    normal = ImageOps.autocontrast(image)

    normal = ImageEnhance.Contrast(
        normal
    ).enhance(1.35)

    variants.append(normal)

    # Variant 2: sharpened
    sharpened = normal.filter(
        ImageFilter.SHARPEN
    )

    variants.append(sharpened)

    # Variant 3: thresholded
    threshold = normal.point(
        lambda pixel: 255
        if pixel > 165
        else 0
    )

    variants.append(threshold)

    # Variant 4: stronger threshold
    strong_threshold = normal.point(
        lambda pixel: 255
        if pixel > 195
        else 0
    )

    variants.append(strong_threshold)

    return variants


def _configure_tesseract() -> None:
    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = (
            config.TESSERACT_CMD
        )

    pytesseract.get_tesseract_version()


def _confidence(
    data: dict,
) -> float | None:
    values: list[float] = []

    for value in data.get(
        "conf",
        [],
    ):
        try:
            number = float(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if number >= 0:
            values.append(number)

    if not values:
        return None

    return (
        sum(values)
        / len(values)
        / 100
    )


def _run_ocr_pass(
    image: Image.Image,
    psm: int,
) -> tuple[str, float | None]:
    text = pytesseract.image_to_string(
        image,
        config=f"--psm {psm}",
    )

    confidence_data = (
        pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config=f"--psm {psm}",
        )
    )

    confidence = _confidence(
        confidence_data
    )

    return (
        normalize_ocr_text(text),
        confidence,
    )


def _multi_pass_ocr(
    image: Image.Image,
) -> tuple[str, float | None]:
    """
    Run multiple OCR layouts and preprocessing variants.

    PSM 6:
        assumes a uniform block of text.

    PSM 11:
        sparse text; useful for packaging.

    PSM 12:
        sparse text with more layout flexibility.
    """

    variants = _preprocess_variants(
        image
    )

    all_text: list[str] = []
    confidences: list[float] = []

    for variant in variants:
        for psm in (6, 11, 12):
            try:
                text, confidence = (
                    _run_ocr_pass(
                        variant,
                        psm,
                    )
                )

            except Exception:
                continue

            if text:
                all_text.append(text)

            if confidence is not None:
                confidences.append(
                    confidence
                )

    # Preserve the first complete OCR result,
    # then append unique lines discovered by
    # other OCR passes.
    merged_lines: list[str] = []
    seen: set[str] = set()

    for text in all_text:
        for line in text.splitlines():
            cleaned = line.strip()

            if not cleaned:
                continue

            normalized_key = re.sub(
                r"[^a-z0-9]+",
                "",
                cleaned.lower(),
            )

            if not normalized_key:
                continue

            if normalized_key in seen:
                continue

            seen.add(normalized_key)
            merged_lines.append(cleaned)

    merged_text = "\n".join(
        merged_lines
    )

    average_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else None
    )

    return (
        normalize_ocr_text(
            merged_text
        ),
        average_confidence,
    )


def process_image_bytes(
    data: bytes,
    filename: str,
    content_type: str | None = None,
    index: int = 0,
) -> OCRImageResult:
    """
    Process one uploaded image.
    """

    if len(data) > MAX_IMAGE_SIZE:
        raise ImageValidationError(
            "IMAGE_TOO_LARGE",
            "Each image must be 10 MB or smaller.",
            413,
        )

    try:
        image = Image.open(
            io.BytesIO(data)
        )

        image_format = image.format

        image.verify()

        if image_format not in SUPPORTED_FORMATS:
            raise ImageValidationError(
                "UNSUPPORTED_IMAGE",
                "Only JPG, JPEG, PNG, and WEBP images are supported.",
                415,
            )

        if (
            content_type
            and content_type.startswith(
                "image/"
            )
            and content_type
            not in SUPPORTED_CONTENT_TYPES
        ):
            raise ImageValidationError(
                "UNSUPPORTED_IMAGE",
                "Only JPG, JPEG, PNG, and WEBP images are supported.",
                415,
            )

    except ImageValidationError:
        raise

    except Exception as error:
        raise ImageValidationError(
            "INVALID_IMAGE",
            "The uploaded file is not a valid supported image.",
            400,
        ) from error

    try:
        with Image.open(
            io.BytesIO(data)
        ) as decoded:
            _configure_tesseract()

            text, confidence = (
                _multi_pass_ocr(
                    decoded
                )
            )

    except (
        pytesseract.TesseractNotFoundError,
        FileNotFoundError,
        OSError,
    ):
        return OCRImageResult(
            index=index,
            filename=filename,
            status=OCRStatus.OCR_UNAVAILABLE,
            warning=(
                "Tesseract OCR is unavailable."
            ),
        )

    except Exception:
        return OCRImageResult(
            index=index,
            filename=filename,
            status=OCRStatus.OCR_FAILED,
            warning=(
                "OCR failed for this image."
            ),
        )

    if not text:
        return OCRImageResult(
            index=index,
            filename=filename,
            status=OCRStatus.NO_TEXT,
            confidence=confidence,
        )

    sections = extract_label_sections(
        text
    )

    return OCRImageResult(
        index=index,
        filename=filename,
        text=text,
        confidence=confidence,
        status=OCRStatus.SUCCESS,
        sections=sections,
    )


def process_images(
    images: Iterable[
        tuple[bytes, str, str | None]
    ],
) -> OCRBatchResult:
    """
    Process multiple uploaded images independently.

    Each uploaded image retains its own OCR
    result so image-to-image comparison can
    distinguish Product A from Product B.
    """

    image_list = list(images)

    if len(image_list) > MAX_IMAGES:
        raise ImageValidationError(
            "TOO_MANY_IMAGES",
            "A maximum of 5 images can be processed at a time.",
            400,
        )

    results: list[OCRImageResult] = []

    for index, (
        data,
        filename,
        content_type,
    ) in enumerate(image_list):
        results.append(
            process_image_bytes(
                data,
                filename,
                content_type,
                index,
            )
        )

    successful = [
        result
        for result in results
        if result.status
        == OCRStatus.SUCCESS
    ]

    combined = "\n\n".join(
        dict.fromkeys(
            result.text
            for result in successful
            if result.text
        )
    )

    warnings = [
        result.warning
        for result in results
        if result.warning
    ]

    confidences = [
        result.confidence
        for result in successful
        if result.confidence is not None
    ]

    if (
        successful
        and len(successful)
        == len(results)
    ):
        status = OCRStatus.SUCCESS

    elif successful:
        status = OCRStatus.PARTIAL

    else:
        status = OCRStatus.NO_TEXT

    if results and all(
        result.status
        == OCRStatus.OCR_UNAVAILABLE
        for result in results
    ):
        status = OCRStatus.OCR_UNAVAILABLE

    sections = (
        extract_label_sections(
            combined
        )
        if combined
        else LabelSections()
    )

    return OCRBatchResult(
        images=results,
        combined_text=combined,
        status=status,
        confidence=(
            sum(confidences)
            / len(confidences)
            if confidences
            else None
        ),
        images_processed=len(results),
        warnings=warnings,
        sections=sections,
        nutrition_data=(
            sections.nutrition_data
        ),
    )