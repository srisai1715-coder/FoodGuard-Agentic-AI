from __future__ import annotations

import re
from enum import Enum

import httpx
from pydantic import BaseModel, Field

from app import config
from app.tools.barcode import InvalidBarcodeError, normalize_barcode
from app.tools.nutrition import NutritionData


class ProductDataQuality(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING_KEY_FIELDS = "MISSING_KEY_FIELDS"
    UNKNOWN = "UNKNOWN"


class ProductData(BaseModel):
    barcode: str
    product_name: str | None = None
    brand: str | None = None
    ingredients_text: str | None = None
    allergen_statement: str | None = None
    allergens: list[str] = Field(default_factory=list)
    traces: list[str] = Field(default_factory=list)
    nutrition: NutritionData | None = None
    image_url: str | None = None
    categories: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    source: str = "open_food_facts"
    source_url: str
    data_quality: ProductDataQuality = ProductDataQuality.UNKNOWN


class ProductLookupError(str, Enum):
    INVALID_BARCODE = "INVALID_BARCODE"
    PRODUCT_NOT_FOUND = "PRODUCT_NOT_FOUND"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    INVALID_UPSTREAM_RESPONSE = "INVALID_UPSTREAM_RESPONSE"
    MISSING_PRODUCT_DATA = "MISSING_PRODUCT_DATA"


class ProductLookupResult(BaseModel):
    found: bool
    product: ProductData | None = None
    source: str = "open_food_facts"
    error: ProductLookupError | None = None


_FIELDS = ",".join(
    [
        "code",
        "product_name",
        "brands",
        "ingredients_text",
        "allergens",
        "allergens_tags",
        "traces",
        "traces_tags",
        "nutriments",
        "serving_size",
        "categories",
        "labels",
        "image_front_url",
    ]
)


def _split_tags(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).removeprefix("en:").replace("-", " ") for item in value if item]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r",", value) if item.strip()]
    return []


def _parse_serving_size(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*g\b", value.lower())
    return float(match.group(1)) if match else None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _nutrition(product: dict) -> NutritionData | None:
    raw = product.get("nutriments")
    if not isinstance(raw, dict):
        return None

    serving_size_g = _parse_serving_size(product.get("serving_size"))
    field_map = {
        "calories": "energy-kcal_serving",
        "protein_g": "proteins_serving",
        "carbohydrates_g": "carbohydrates_serving",
        "total_sugar_g": "sugars_serving",
        "added_sugar_g": "added-sugars_serving",
        "fat_g": "fat_serving",
        "saturated_fat_g": "saturated-fat_serving",
        "trans_fat_g": "trans-fat_serving",
        "fiber_g": "fiber_serving",
        "sodium_mg": "sodium_serving",
        "cholesterol_mg": "cholesterol_serving",
    }
    values = {field: _number(raw.get(source)) for field, source in field_map.items()}
    if values["sodium_mg"] is not None:
        values["sodium_mg"] *= 1000
    if values["cholesterol_mg"] is not None:
        values["cholesterol_mg"] *= 1000

    per_100g: dict[str, float] = {}
    per_100_map = {
        "calories_per_100g": "energy-kcal_100g",
        "protein_g_per_100g": "proteins_100g",
        "carbohydrates_g_per_100g": "carbohydrates_100g",
        "total_sugar_g_per_100g": "sugars_100g",
        "added_sugar_g_per_100g": "added-sugars_100g",
        "fat_g_per_100g": "fat_100g",
        "saturated_fat_g_per_100g": "saturated-fat_100g",
        "trans_fat_g_per_100g": "trans-fat_100g",
        "fiber_g_per_100g": "fiber_100g",
        "sodium_mg_per_100g": "sodium_100g",
        "cholesterol_mg_per_100g": "cholesterol_100g",
    }
    for field, source in per_100_map.items():
        number = _number(raw.get(source))
        if number is not None:
            per_100g[field] = number * 1000 if field in {"sodium_mg_per_100g", "cholesterol_mg_per_100g"} else number

    if not serving_size_g and not per_100g and not any(value is not None for value in values.values()):
        return None
    return NutritionData(serving_size_g=serving_size_g, values_per_100g=per_100g, **values)


def _normalize_product(barcode: str, product: dict) -> ProductData:
    ingredients = product.get("ingredients_text")
    nutrition = _nutrition(product)
    product_name = product.get("product_name") or None
    brand = product.get("brands") or None
    has_core = bool(product_name and (ingredients or nutrition))
    has_any = bool(product_name or brand or ingredients or nutrition or product.get("categories"))
    quality = (
        ProductDataQuality.COMPLETE
        if product_name and ingredients and nutrition
        else ProductDataQuality.PARTIAL
        if has_any
        else ProductDataQuality.MISSING_KEY_FIELDS
    )
    return ProductData(
        barcode=barcode,
        product_name=product_name,
        brand=brand,
        ingredients_text=ingredients or None,
        allergen_statement=product.get("allergens") or None,
        allergens=_split_tags(product.get("allergens_tags") or product.get("allergens")),
        traces=_split_tags(product.get("traces_tags") or product.get("traces")),
        nutrition=nutrition,
        image_url=product.get("image_front_url") or None,
        categories=_split_tags(product.get("categories")),
        labels=_split_tags(product.get("labels")),
        source_url=f"{config.OPEN_FOOD_FACTS_BASE_URL}/product/{barcode}.json",
        data_quality=quality if has_core or has_any else ProductDataQuality.MISSING_KEY_FIELDS,
    )


def lookup_product_by_barcode(barcode: str) -> ProductLookupResult:
    try:
        normalized = normalize_barcode(barcode)
    except (InvalidBarcodeError, AttributeError):
        return ProductLookupResult(found=False, error=ProductLookupError.INVALID_BARCODE)

    url = f"{config.OPEN_FOOD_FACTS_BASE_URL}/product/{normalized}.json"
    try:
        with httpx.Client(timeout=config.OPEN_FOOD_FACTS_TIMEOUT) as client:
            response = client.get(url, params={"fields": _FIELDS}, headers={"User-Agent": config.FOODGUARD_USER_AGENT})
    except httpx.TimeoutException:
        return ProductLookupResult(found=False, error=ProductLookupError.UPSTREAM_TIMEOUT)
    except httpx.HTTPError:
        return ProductLookupResult(found=False, error=ProductLookupError.UPSTREAM_UNAVAILABLE)

    if response.status_code == 404:
        return ProductLookupResult(found=False, error=ProductLookupError.PRODUCT_NOT_FOUND)
    if response.status_code >= 400:
        return ProductLookupResult(found=False, error=ProductLookupError.UPSTREAM_UNAVAILABLE)
    try:
        payload = response.json()
    except ValueError:
        return ProductLookupResult(found=False, error=ProductLookupError.INVALID_UPSTREAM_RESPONSE)
    if not isinstance(payload, dict):
        return ProductLookupResult(found=False, error=ProductLookupError.INVALID_UPSTREAM_RESPONSE)
    if payload.get("status") == 0 or not isinstance(payload.get("product"), dict):
        return ProductLookupResult(found=False, error=ProductLookupError.PRODUCT_NOT_FOUND)
    try:
        product = _normalize_product(normalized, payload["product"])
    except (TypeError, ValueError):
        return ProductLookupResult(found=False, error=ProductLookupError.INVALID_UPSTREAM_RESPONSE)
    return ProductLookupResult(found=True, product=product)