from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field


NutrientName = Literal[
    "calories",
    "protein_g",
    "carbohydrates_g",
    "total_sugar_g",
    "added_sugar_g",
    "fat_g",
    "saturated_fat_g",
    "trans_fat_g",
    "fiber_g",
    "sodium_mg",
    "cholesterol_mg",
]


class NutritionData(BaseModel):
    serving_size_g: float | None = Field(default=None)
    servings_per_container: float | None = Field(default=None)
    calories: float | None = None
    protein_g: float | None = None
    carbohydrates_g: float | None = None
    total_sugar_g: float | None = None
    added_sugar_g: float | None = None
    fat_g: float | None = None
    saturated_fat_g: float | None = None
    trans_fat_g: float | None = None
    fiber_g: float | None = None
    sodium_mg: float | None = None
    cholesterol_mg: float | None = None
    values_per_100g: dict[str, float] = Field(default_factory=dict)


def _valid_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def normalize_nutrient_per_100g(
    value_per_serving: float | None,
    serving_size_g: float | None,
) -> float | None:
    """Convert a non-negative per-serving value to a per-100g value."""
    if not _valid_number(value_per_serving) or not _valid_number(serving_size_g) or serving_size_g == 0:
        return None
    return value_per_serving / serving_size_g * 100


def normalize_nutrient_per_serving(
    value_per_100g: float | None,
    serving_size_g: float | None,
) -> float | None:
    """Convert a non-negative per-100g value to a per-serving value."""
    if not _valid_number(value_per_100g) or not _valid_number(serving_size_g):
        return None
    return value_per_100g * serving_size_g / 100


def compare_nutrients(
    first: NutritionData,
    second: NutritionData,
    nutrient: NutrientName,
) -> dict[str, float | None | str]:
    """Compare one supplied nutrient across two nutrition records."""
    first_value = getattr(first, nutrient, None)
    second_value = getattr(second, nutrient, None)
    if not _valid_number(first_value) or not _valid_number(second_value):
        return {"first": first_value, "second": second_value, "difference": None, "higher": "unknown"}
    higher = "equal" if first_value == second_value else ("first" if first_value > second_value else "second")
    return {
        "first": first_value,
        "second": second_value,
        "difference": second_value - first_value,
        "higher": higher,
    }


def calculate_percentage_difference(
    first_value: float | None,
    second_value: float | None,
) -> float | None:
    """Return the percentage change from the first value to the second value."""
    if not _valid_number(first_value) or not _valid_number(second_value) or first_value == 0:
        return None
    return (second_value - first_value) / first_value * 100


def normalized_values(data: NutritionData) -> dict[str, float]:
    """Calculate available per-100g values deterministically from per-serving data."""
    values: dict[str, float] = {}
    for field_name in (
        "calories",
        "protein_g",
        "carbohydrates_g",
        "total_sugar_g",
        "added_sugar_g",
        "fat_g",
        "saturated_fat_g",
        "trans_fat_g",
        "fiber_g",
        "sodium_mg",
        "cholesterol_mg",
    ):
        value = normalize_nutrient_per_100g(
            getattr(data, field_name),
            data.serving_size_g,
        )
        if value is not None:
            values[f"{field_name}_per_100g"] = value
    for name, value in data.values_per_100g.items():
        if _valid_number(value):
            values[name] = value
    return values