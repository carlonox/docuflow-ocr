"""Fictional form extractor (example implementation).

Extracts a form code (``FORM-XXXXXX``) and a signature presence flag from OCR
text. All formats are invented for demonstration.
"""

from __future__ import annotations

import re
from typing import Dict

from docuflow.ocr.base_extractor import AbstractDocumentExtractor
from docuflow.validation.validators import RegexValidator


class FormExtractor(AbstractDocumentExtractor):
    """Extract fields from a fictional form.

    Fields:
        - ``form_code``: matches ``FORM-[A-Z0-9]{6}``.
        - ``signature_present``: ``yes``/``no`` heuristic based on the word
          "signature" appearing near a filled-in marker.
    """

    document_type = "form"

    def __init__(self, cascade, validator=None, learning=None) -> None:
        super().__init__(cascade, validator, learning, fields=["form_code", "signature_present"])

    def extract_fields(self, raw_text: str) -> Dict[str, str]:
        form_code = ""
        match_code = re.search(r"\bFORM-[A-Z0-9]{6}\b", raw_text)
        if match_code:
            form_code = match_code.group(0)

        # Heuristic: a signature line is "filled" when text appears between
        # the word "signature" and the next blank line.
        signature_present = "no"
        lowered = raw_text.lower()
        sig_index = lowered.find("signature")
        if sig_index != -1:
            window = raw_text[sig_index : sig_index + 200]
            if re.search(r"[A-Za-z]{3,}", window.replace("signature", "", 1)):
                signature_present = "yes"

        return {
            "form_code": form_code,
            "signature_present": signature_present,
        }

    def validate_fields(self, fields: Dict[str, str]) -> Dict[str, object]:
        code_result = RegexValidator(
            r"FORM-[A-Z0-9]{6}", description="form code"
        ).validate("form_code", fields.get("form_code"))

        return {
            "form_code": code_result.to_dict(),
            "signature_present": {
                "valid": fields.get("signature_present") in ("yes", "no"),
                "value": fields.get("signature_present"),
            },
        }
