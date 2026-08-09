"""Abstract document extractor.

Defines the lifecycle every document extractor follows:

    extract -> validate -> learn

1. ``extract``: run the OCR cascade over the document image.
2. ``validate``: apply deterministic rules to the extracted fields.
3. ``learn``: feed successes/failures into the learning subsystem so the
   next extraction is better.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from docuflow.ocr.providers.base import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of a full extraction lifecycle.

    Attributes:
        document_id: Identifier of the processed document (e.g. row number).
        fields: Extracted field name -> value mapping.
        validation: Per-field validation outcomes.
        success: Whether the extraction passed validation.
        confidence: Overall extraction confidence in [0, 1].
        raw_text: Raw OCR text (for debugging and learning).
        provider_chain: Names of providers that ran.
        metadata: Arbitrary extra data (timings, versions, ...).
    """

    document_id: Optional[str] = None
    fields: Dict[str, str] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    confidence: float = 0.0
    raw_text: str = ""
    provider_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AbstractDocumentExtractor(ABC):
    """Base class for document-specific extractors.

    Subclasses implement :meth:`extract_fields` (turning OCR text into
    structured fields) and optionally :meth:`validate_fields`. The base class
    orchestrates the full lifecycle including the learning hooks.
    """

    #: Human-readable document type, used as learning context.
    document_type: str = "generic"

    def __init__(
        self,
        cascade: Any,
        validator: Optional[Any] = None,
        learning: Optional[Any] = None,
        fields: Optional[List[str]] = None,
    ) -> None:
        self.cascade = cascade
        self.validator = validator
        self.learning = learning
        self.fields = fields or ["field_a", "field_b"]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def process(self, image_path: str, document_id: Optional[str] = None) -> ExtractionResult:
        """Run the full extract -> validate -> learn lifecycle.

        Args:
            image_path: Path to the document image.
            document_id: Optional identifier recorded in the result.

        Returns:
            An :class:`ExtractionResult` with fields and validation outcomes.
        """
        ocr_result = self.cascade.extract(image_path)
        raw_text = ocr_result.text

        fields = self.extract_fields(raw_text)
        # Subclasses implement validate_fields() as a method; the optional
        # validator object is an additional external gate, not a replacement.
        validation = self.validate_fields(fields)
        if self.validator is not None:
            external = self.validator.validate(fields)
            validation.update(external)

        success = self._decide_success(validation)
        result = ExtractionResult(
            document_id=document_id,
            fields=fields,
            validation=validation,
            success=success,
            confidence=ocr_result.confidence,
            raw_text=raw_text,
            provider_chain=[ocr_result.provider],
        )

        if self.learning is not None:
            self._learn(result)
        return result

    def _decide_success(self, validation: Dict[str, Any]) -> bool:
        """A result is successful when every field passed validation."""
        outcomes = [
            v.get("valid", False)
            for v in validation.values()
            if isinstance(v, dict)
        ]
        return bool(outcomes) and all(outcomes)

    def _learn(self, result: ExtractionResult) -> None:
        """Feed the extraction outcome into the learning subsystem."""
        if self.learning is None:
            return
        if not hasattr(self.learning, "record"):
            logger.debug("Learning object has no record() hook; skipping")
            return
        try:
            matches = [
                result.validation.get(f, {}).get("valid", False)
                for f in self.fields
            ]
            self.learning.record(result.document_id or "", matches, result.confidence)
        except Exception as exc:  # pragma: no cover - learning must never crash extraction
            logger.warning("Learning hook failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    def extract_fields(self, raw_text: str) -> Dict[str, str]:
        """Convert raw OCR text into structured fields.

        Args:
            raw_text: Text produced by the OCR cascade.

        Returns:
            A mapping of field name -> extracted value.
        """
        raise NotImplementedError

    def validate_fields(self, fields: Dict[str, str]) -> Dict[str, Any]:
        """Validate extracted fields with deterministic rules.

        Default implementation marks every field valid; subclasses with
        expectations (regex, checksums, dictionaries) override this.

        Args:
            fields: Extracted fields.

        Returns:
            A mapping of field name -> validation detail dict.
        """
        return {field: {"valid": True, "value": fields.get(field)} for field in self.fields}
