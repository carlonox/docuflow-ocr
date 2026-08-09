"""Fictional ID card extractor (example implementation).

The document format used here — an ID number like ``ID-12345678`` and a name
line — is invented for demonstration purposes. It is deliberately NOT any
real national ID format.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from docuflow.ocr.base_extractor import AbstractDocumentExtractor
from docuflow.validation.validators import RegexValidator


class IDCardExtractor(AbstractDocumentExtractor):
    """Extract fields from a fictional ID card.

    Fields:
        - ``id_number``: matches ``ID-########``.
        - ``full_name``: first non-empty line without the ID.
    """

    document_type = "id_card"

    def __init__(self, cascade, validator=None, learning=None) -> None:
        super().__init__(cascade, validator, learning, fields=["id_number", "full_name"])

    def extract_fields(self, raw_text: str) -> Dict[str, str]:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        id_number = ""
        name_lines: list = []
        for line in lines:
            match = re.search(r"\bID-\d{8}\b", line)
            if match:
                id_number = match.group(0)
            else:
                name_lines.append(line)

        return {
            "id_number": id_number,
            "full_name": " ".join(name_lines) if name_lines else "",
        }

    def validate_fields(self, fields: Dict[str, str]) -> Dict[str, object]:
        id_validator = RegexValidator(r"ID-\d{8}", description="ID card number")
        id_result = id_validator.validate("id_number", fields.get("id_number"))
        name_valid = bool(fields.get("full_name", "").strip())

        return {
            "id_number": id_result.to_dict(),
            "full_name": {"valid": name_valid, "value": fields.get("full_name")},
        }
