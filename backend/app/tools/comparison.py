from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.tools.nutrition import (
    NutrientName,
    NutritionData,
    calculate_percentage_difference,
    normalize_nutrient_per_100g,
)
from app.tools.product_lookup import ProductData


class NutrientComparisonItem(BaseModel):
    nutrient_name: str
    value_a_serving: float | None = None
    value_b_serving: float | None = None
    value_a_100g: float | None = None
    value_b_100g: float | None = None
    unit: str = "g"
    delta_100g: float | None = None
    percent_diff_100g: float | None = None
    lower_product: str | None = None  # "A", "B", or "equal"
    higher_product: str | None = None  # "A", "B", or "equal"
    status: str = "COMPARED"  # "COMPARED", "PARTIAL", or "MISSING"


class ComparisonResult(BaseModel):
    product_a_name: str
    product_b_name: str
    product_a_barcode: str | None = None
    product_b_barcode: str | None = None
    serving_size_a_g: float | None = None
    serving_size_b_g: float | None = None
    serving_size_match: bool = True
    nutrients: dict[str, NutrientComparisonItem] = Field(default_factory=dict)
    common_ingredients: list[str] = Field(default_factory=list)
    unique_ingredients_a: list[str] = Field(default_factory=list)
    unique_ingredients_b: list[str] = Field(default_factory=list)
    allergens_a: list[str] = Field(default_factory=list)
    allergens_b: list[str] = Field(default_factory=list)
    common_allergens: list[str] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)


def _split_ingredients(text: str | None) -> set[str]:
    if not text or not text.strip():
        return set()
    raw = re.split(r"[,;\n\(\)]", text.lower())
    clean = set()
    for item in raw:
        item = item.strip().strip(".*- ")
        if item and len(item) > 1 and not item.isdigit():
            clean.add(item)
    return clean


def compare_products(
    product_a: ProductData,
    product_b: ProductData,
) -> ComparisonResult:
    name_a = product_a.product_name or product_a.brand or f"Product ({product_a.barcode})"
    name_b = product_b.product_name or product_b.brand or f"Product ({product_b.barcode})"

    nutr_a = product_a.nutrition or NutritionData()
    nutr_b = product_b.nutrition or NutritionData()

    serv_a = nutr_a.serving_size_g
    serv_b = nutr_b.serving_size_g
    serv_match = serv_a is not None and serv_b is not None and abs(serv_a - serv_b) < 0.1

    nutrient_configs: list[tuple[NutrientName, str, str]] = [
        ("calories", "Calories", "kcal"),
        ("total_sugar_g", "Total Sugars", "g"),
        ("added_sugar_g", "Added Sugars", "g"),
        ("sodium_mg", "Sodium", "mg"),
        ("protein_g", "Protein", "g"),
        ("fat_g", "Total Fat", "g"),
        ("saturated_fat_g", "Saturated Fat", "g"),
        ("trans_fat_g", "Trans Fat", "g"),
        ("fiber_g", "Dietary Fiber", "g"),
        ("carbohydrates_g", "Carbohydrates", "g"),
        ("cholesterol_mg", "Cholesterol", "mg"),
    ]

    compared_nutrients: dict[str, NutrientComparisonItem] = {}
    data_notes: list[str] = []

    if not serv_match and serv_a and serv_b:
        data_notes.append(
            f"Serving sizes differ: {name_a} is {serv_a}g vs {name_b} is {serv_b}g. "
            "Comparison includes normalized per-100g values for fair comparison."
        )

    for field_key, display_name, unit in nutrient_configs:
        val_a_serv = getattr(nutr_a, field_key, None)
        val_b_serv = getattr(nutr_b, field_key, None)

        val_a_100 = nutr_a.values_per_100g.get(f"{field_key}_per_100g")
        if val_a_100 is None and val_a_serv is not None and serv_a:
            val_a_100 = normalize_nutrient_per_100g(val_a_serv, serv_a)

        val_b_100 = nutr_b.values_per_100g.get(f"{field_key}_per_100g")
        if val_b_100 is None and val_b_serv is not None and serv_b:
            val_b_100 = normalize_nutrient_per_100g(val_b_serv, serv_b)

        # Determine delta & percentage diff on per-100g if available, else per-serving
        ref_a = val_a_100 if val_a_100 is not None else val_a_serv
        ref_b = val_b_100 if val_b_100 is not None else val_b_serv

        delta = None
        p_diff = None
        lower = None
        higher = None
        status = "COMPARED"

        if ref_a is not None and ref_b is not None:
            delta = ref_b - ref_a
            p_diff = calculate_percentage_difference(ref_a, ref_b)
            if abs(ref_a - ref_b) < 1e-4:
                lower = "equal"
                higher = "equal"
            elif ref_a < ref_b:
                lower = "A"
                higher = "B"
            else:
                lower = "B"
                higher = "A"
        elif ref_a is not None or ref_b is not None:
            status = "PARTIAL"
            missing_prod = name_b if ref_b is None else name_a
            data_notes.append(f"{display_name} is missing for {missing_prod}.")
        else:
            status = "MISSING"

        compared_nutrients[field_key] = NutrientComparisonItem(
            nutrient_name=display_name,
            value_a_serving=val_a_serv,
            value_b_serving=val_b_serv,
            value_a_100g=val_a_100,
            value_b_100g=val_b_100,
            unit=unit,
            delta_100g=delta,
            percent_diff_100g=p_diff,
            lower_product=lower,
            higher_product=higher,
            status=status,
        )

    # Ingredient Comparison
    ing_a = _split_ingredients(product_a.ingredients_text)
    ing_b = _split_ingredients(product_b.ingredients_text)
    common_ing = sorted(list(ing_a.intersection(ing_b)))
    uniq_a = sorted(list(ing_a - ing_b))
    uniq_b = sorted(list(ing_b - ing_a))

    # Allergen Comparison
    all_a = set(product_a.allergens)
    all_b = set(product_b.allergens)
    common_all = sorted(list(all_a.intersection(all_b)))

    return ComparisonResult(
        product_a_name=name_a,
        product_b_name=name_b,
        product_a_barcode=product_a.barcode,
        product_b_barcode=product_b.barcode,
        serving_size_a_g=serv_a,
        serving_size_b_g=serv_b,
        serving_size_match=serv_match,
        nutrients=compared_nutrients,
        common_ingredients=common_ing,
        unique_ingredients_a=uniq_a,
        unique_ingredients_b=uniq_b,
        allergens_a=sorted(list(all_a)),
        allergens_b=sorted(list(all_b)),
        common_allergens=common_all,
        data_quality_notes=data_notes,
    )
