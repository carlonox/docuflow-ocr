"""Fictional invoice extractor (example implementation).

Extracts an invoice number (``INV-YYYY-NNNN``), a total amount and a vendor
line from OCR text. All formats are invented.
"""

from __future__ import annotations

import re
from typing import Dict

from docuflow.ocr.base_extractor import AbstractDocumentExtractor
from docuflow.validation.validators import RegexValidator


class InvoiceExtractor(AbstractDocumentExtractor):
    """Extract fields from a fictional invoice.

    Fields:
        - ``invoice_number``: matches ``INV-\\d{4}-\\d{4}``.
        - ``total_amount``: decimal amount found near a TOTAL marker.
        - ``vendor``: line containing the vendor name.
    """

    document_type = "invoice"

    def __init__(self, cascade, validator=None, learning=None) -> None:
        super().__init__(
            cascade, validator, learning, fields=["invoice_number", "total_amount", "vendor"]
        )

    def extract_fields(self, raw_text: str) -> Dict[str, str]:
        invoice_number = ""
        match_invoice = re.search(r"\bINV-\d{4}-\d{4}\b", raw_text)
        if match_invoice:
            invoice_number = match_invoice.group(0)

        total_amount = ""
        match_total = re.search(
            r"(?:TOTAL|TOTal|Total)[\s:]*\$?\s*(\d{1,7}(?:\.\d{2})?)", raw_text
        )
        if match_total:
            total_amount = match_total.group(1)

        vendor = ""
        for line in raw_text.splitlines():
            lowered = line.strip().lower()
            if "vendor" in lowered or "supplier" in lowered:
                vendor = line.strip()
                break

        return {
            "invoice_number": invoice_number,
            "total_amount": total_amount,
            "vendor": vendor,
        }

    def validate_fields(self, fields: Dict[str, str]) -> Dict[str, object]:
        number_result = RegexValidator(
            r"INV-\d{4}-\d{4}", description="invoice number"
        ).validate("invoice_number", fields.get("invoice_number"))
        amount_result = RegexValidator(
            r"\d{1,7}(\.\d{2})?", description="amount"
        ).validate("total_amount", fields.get("total_amount"))

        return {
            "invoice_number": number_result.to_dict(),
            "total_amount": amount_result.to_dict(),
            "vendor": {
                "valid": bool(fields.get("vendor", "").strip()),
                "value": fields.get("vendor"),
            },
        }
