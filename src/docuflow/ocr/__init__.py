"""OCR subsystem: providers, preprocessing, cascade orchestration."""

from docuflow.ocr.base_extractor import AbstractDocumentExtractor  # noqa: F401
from docuflow.ocr.cascade import CascadeOCR, CascadeConfig  # noqa: F401

__all__ = ["AbstractDocumentExtractor", "CascadeOCR", "CascadeConfig"]
