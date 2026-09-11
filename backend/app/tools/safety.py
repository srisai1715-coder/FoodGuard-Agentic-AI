from __future__ import annotations

import re
from enum import Enum


class AllergenStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN = "UNKNOWN"


COMMON_ALLERGENS = (
    "milk",
    "egg",
    "peanut",
    "tree_nuts",
    "wheat",
    "soy",
    "fish",
    "shellfish",
    "sesame",
)

_ALLERGEN_TERMS = {
    "milk": ("milk", "milk powder", "whey", "casein", "butter", "cream"),
    "egg": ("egg", "eggs", "albumin", "mayonnaise"),
    "peanut": ("peanut", "peanuts", "groundnut", "groundnuts"),
    "tree_nuts": ("tree nut", "tree nuts", "almond", "cashew", "walnut", "hazelnut", "pecan", "pistachio"),
    "wheat": ("wheat", "wheat flour", "semolina", "durum", "spelt"),
    "soy": ("soy", "soya", "soybean", "soy lecithin"),
    "fish": ("fish", "salmon", "tuna", "cod", "anchovy"),
    "shellfish": ("shellfish", "shrimp", "prawn", "crab", "lobster", "mussel", "oyster"),
    "sesame": ("sesame", "tahini"),
}


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text.lower()) is not None


def detect_explicit_allergens(
    ingredient_text: str | None,
    allergen_statement: str | None,
) -> dict[str, dict[str, str]]:
    """Detect allergen evidence explicitly present in supplied label text."""
    source_parts = []
    for part in (ingredient_text, allergen_statement):
        if not part:
            continue
        source_parts.extend(
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+|\n", part)
            if "may contain" not in sentence.lower()
            and "facility" not in sentence.lower()
            and "processes" not in sentence.lower()
        )
    combined = " ".join(source_parts).strip()
    if not combined:
        return {name: {"status": AllergenStatus.UNKNOWN.value, "evidence": "No allergen information was supplied."} for name in COMMON_ALLERGENS}

    findings = {}
    for name, terms in _ALLERGEN_TERMS.items():
        matched = next((term for term in terms if _contains_term(combined, term)), None)
        if matched:
            findings[name] = {
                "status": AllergenStatus.CONFIRMED.value,
                "evidence": f"The supplied information explicitly mentions {matched}.",
            }
        else:
            findings[name] = {
                "status": AllergenStatus.NOT_FOUND.value,
                "evidence": f"No {name.replace('_', ' ')} was identified in the supplied information.",
            }
    return findings


def detect_cross_contact_statement(*texts: str | None) -> list[str]:
    """Return supplied statements that explicitly describe possible cross-contact."""
    warnings = []
    for text in texts:
        if not text:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text.strip()):
            lowered = sentence.lower()
            if "may contain" in lowered or "facility" in lowered or "processes" in lowered:
                warnings.append(sentence.strip())
    return list(dict.fromkeys(warnings))


def classify_allergen_status(
    ingredient_text: str | None,
    allergen_statement: str | None,
) -> dict[str, dict[str, str]]:
    """Classify each supported allergen without treating missing data as absence."""
    return detect_explicit_allergens(ingredient_text, allergen_statement)