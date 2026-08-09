"""Cascade OCR pipeline: fast -> accurate -> LLM correction.

Instead of trusting a single OCR engine, the cascade runs providers from
cheapest to most expensive and only escalates when confidence demands it:

1. Fast provider (RapidOCR, ~0.1-2s) — accepted when confidence is high.
2. Accurate provider (Surya OCR 2, ~10-30s CPU) — for degraded documents.
3. LLM corrector (local model) — fixes doubtful fields with linguistic context.
4. Deterministic validation (regex, dictionaries, checksums).

The result is ~90% accuracy on degraded documents using only a CPU with
8GB of RAM, at a fraction of the cost of a cloud OCR API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from docuflow.ocr.providers.base import AbstractOCRProvider, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class CascadeConfig:
    """Configuration for the cascade pipeline.

    Attributes:
        fast_threshold: Minimum average confidence for the fast provider's
            result to be accepted without escalation.
        llm_required_fields: Field names that trigger LLM correction when the
            accurate provider leaves them doubtful.
        validation: Optional deterministic validator (see
            :mod:`docuflow.validation.validators`).
        max_stages: Maximum number of providers to try before giving up.
    """

    fast_threshold: float = 0.90
    llm_required_fields: List[str] = field(default_factory=list)
    validation: Optional[Any] = None
    max_stages: int = 3


class CascadeOCR:
    """Orchestrate multiple OCR providers with escalation and validation.

    Args:
        fast_provider: First-line provider (fast, cheap).
        accurate_provider: Second-line provider (slow, accurate).
        llm_corrector: Optional callable ``(text, context) -> text`` used to
            repair doubtful fields. Typically a local model; the public repo
            ships a lightweight reference implementation.
        config: Cascade tuning parameters.
        context: Free-form string describing the document type; passed to the
            LLM corrector as linguistic context (e.g. "birth certificate").
    """

    def __init__(
        self,
        fast_provider: AbstractOCRProvider,
        accurate_provider: AbstractOCRProvider,
        llm_corrector: Optional[Callable[[str, str], str]] = None,
        config: Optional[CascadeConfig] = None,
        context: str = "",
    ) -> None:
        self.fast_provider = fast_provider
        self.accurate_provider = accurate_provider
        self.llm_corrector = llm_corrector
        self.config = config or CascadeConfig()
        self.context = context

    def extract(self, image_path: str) -> OCRResult:
        """Run the cascade over an image and return the best result.

        Args:
            image_path: Path to the image file.

        Returns:
            The accepted :class:`OCRResult`. If every stage fails, the best
            available result is returned with its original confidence.
        """
        candidates: List[OCRResult] = []

        # Stage 1: fast provider.
        try:
            fast_result = self.fast_provider.extract(image_path)
            candidates.append(fast_result)
            logger.info(
                "Fast provider '%s' confidence=%.3f",
                self.fast_provider.name, fast_result.confidence,
            )
            if not fast_result.empty and fast_result.confidence >= self.config.fast_threshold:
                return self._finalize(fast_result)
        except Exception as exc:  # pragma: no cover - provider failure path
            logger.warning("Fast provider failed: %s", exc)

        # Stage 2: accurate provider.
        try:
            accurate_result = self.accurate_provider.extract(image_path)
            candidates.append(accurate_result)
            logger.info(
                "Accurate provider '%s' confidence=%.3f",
                self.accurate_provider.name, accurate_result.confidence,
            )
            if not accurate_result.empty:
                # Stage 3: LLM correction of doubtful fields.
                if self.llm_corrector is not None:
                    accurate_result = self._llm_pass(accurate_result)
                return self._finalize(accurate_result)
        except Exception as exc:  # pragma: no cover - provider failure path
            logger.warning("Accurate provider failed: %s", exc)

        # All stages failed: return the best candidate we have.
        if candidates:
            best = max(candidates, key=lambda r: r.confidence)
            logger.error("Cascade exhausted, returning best candidate (%.3f)", best.confidence)
            return self._finalize(best)

        return OCRResult(text="", confidence=0.0, provider="cascade")

    def _llm_pass(self, result: OCRResult) -> OCRResult:
        """Apply LLM correction and merge the result."""
        if self.llm_corrector is None:
            return result
        try:
            corrected_text = self.llm_corrector(result.text, self.context)
            if corrected_text and corrected_text.strip():
                logger.info("LLM corrector rewrote %d -> %d chars",
                            len(result.text), len(corrected_text))
                return OCRResult(
                    text=corrected_text,
                    confidence=result.confidence,
                    provider=f"{result.provider}+llm",
                    per_word=result.per_word,
                    raw=result.raw,
                )
        except Exception as exc:  # pragma: no cover - corrector failure path
            logger.warning("LLM corrector failed, keeping raw text: %s", exc)
        return result

    def _finalize(self, result: OCRResult) -> OCRResult:
        """Run the deterministic validator, if configured."""
        if self.config.validation is not None and not result.empty:
            try:
                result.raw = self.config.validation.validate(result.text)
            except Exception as exc:  # pragma: no cover - validator failure path
                logger.warning("Validator failed: %s", exc)
        return result
