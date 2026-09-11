import re


class InvalidBarcodeError(ValueError):
    """Raised when a barcode is not a plausible numeric product code."""


def normalize_barcode(barcode: str) -> str:
    """Validate and normalize common EAN/UPC/GTIN barcode lengths."""
    value = barcode.strip()
    if not re.fullmatch(r"\d{7,14}", value) or set(value) == {"0"}:
        raise InvalidBarcodeError("Barcode must contain 7 to 14 numeric digits.")

    significant_digits = value.lstrip("0") or "0"
    if len(significant_digits) <= 7:
        return significant_digits.zfill(8)
    if len(significant_digits) <= 12:
        return significant_digits.zfill(13)
    return significant_digits