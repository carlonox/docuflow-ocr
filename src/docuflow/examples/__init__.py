"""Example extractors (fictional formats only)."""

from docuflow.ocr.base_extractor import AbstractDocumentExtractor  # noqa: F401
from docuflow.examples.id_card_extractor import IDCardExtractor  # noqa: F401
from docuflow.examples.invoice_extractor import InvoiceExtractor  # noqa: F401

__all__ = ["IDCardExtractor", "InvoiceExtractor"]
