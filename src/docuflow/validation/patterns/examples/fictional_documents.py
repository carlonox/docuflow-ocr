"""Example validation patterns.

These patterns are fictional and exist only to show how to declare field
validators for a document type. Replace them with your own rules; never
commit patterns derived from real personal data.
"""

from __future__ import annotations

from docuflow.validation.validators import (
    CompositeValidator,
    LengthValidator,
    RegexValidator,
)

# ---------------------------------------------------------------------- #
# Example: fictional ID card (IDs of the form "ID-12345678")
# ---------------------------------------------------------------------- #

ID_CARD_NUMBER = CompositeValidator([
    RegexValidator(r"ID-\d{8}", description="ID card number format 'ID-########'"),
    LengthValidator(min_length=10, max_length=11),
])

# ---------------------------------------------------------------------- #
# Example: fictional invoice (numbers of the form "INV-2026-0042")
# ---------------------------------------------------------------------- #

INVOICE_NUMBER = CompositeValidator([
    RegexValidator(r"INV-\d{4}-\d{4}", description="invoice number format 'INV-YYYY-NNNN'"),
])

INVOICE_AMOUNT = RegexValidator(
    r"\d{1,7}(\.\d{2})?", description="amount with two decimal places"
)

# ---------------------------------------------------------------------- #
# Example: fictional form code (alphanumeric, e.g. "FORM-A7B3C9")
# ---------------------------------------------------------------------- #

FORM_CODE = CompositeValidator([
    RegexValidator(r"FORM-[A-Z0-9]{6}", description="form code format 'FORM-XXXXXX'"),
])
