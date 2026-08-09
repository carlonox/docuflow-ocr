"""OCR text normalization.

OCR engines misread characters in predictable ways (``l`` for ``1``, ``O``
for ``0``, ``|`` for ``I``). Normalization cleans raw OCR output before
validation so field matchers compare semantically equal text.
"""

from __future__ import annotations

import re
from typing import Optional

# Common OCR confusions, ordered so more specific replacements win first.
OCR_CONFUSIONS: dict = {
    "|": "I",
    "l": "1",
    "O": "0",
    "o": "0",
    "S": "5",
    "B": "8",
    "Z": "2",
}


def normalize_ocr_text(text: str) -> str:
    """Apply common OCR character corrections to raw text.

    Args:
        text: Raw text as produced by an OCR engine.

    Returns:
        Normalized text.
    """
    normalized = text
    for wrong, right in OCR_CONFUSIONS.items():
        normalized = normalized.replace(wrong, right)
    return normalized


def clean_number(number: Optional[str]) -> Optional[str]:
    """Normalize a detected numeric value.

    Applies the following rules:

    - Strips spaces, dots and commas (thousands separators).
    - Drops a leading ``E`` (foreign-ID marker used in some documents).
    - Strips leading zeros.
    - Returns ``None`` unless a valid 6-10 digit number remains.

    Args:
        number: Raw numeric string detected by OCR.

    Returns:
        The cleaned number, or None when no valid number remains.
    """
    if not number:
        return None

    cleaned = number.replace(" ", "").replace(".", "").replace(",", "")
    if cleaned.upper().startswith("E"):
        cleaned = cleaned[1:]
    cleaned = cleaned.lstrip("0")

    if cleaned.isdigit() and 6 <= len(cleaned) <= 10:
        return cleaned
    return None


_NUMBER_PATTERNS = (
    r"\b[E0]*\d[\d\s\.,]{5,14}\d\b",  # with separators
    r"\b[E0]*\d{6,10}\b",             # plain digits
)


def extract_numbers(text: str) -> list:
    """Extract all plausible identification numbers from OCR text.

    Catches old-format (6-8 digit) and new-format (10 digit) numbers, with or
    without separators and leading zeros.

    Args:
        text: Normalized OCR text.

    Returns:
        A list of unique cleaned numbers (6-10 digits each).
    """
    found: list = []
    for pattern in _NUMBER_PATTERNS:
        for match in re.findall(pattern, text, re.IGNORECASE):
            cleaned = clean_number(match)
            if cleaned and cleaned not in found:
                found.append(cleaned)
    return found
