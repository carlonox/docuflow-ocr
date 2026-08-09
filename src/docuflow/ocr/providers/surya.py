"""Surya OCR 2 provider.

Surya OCR 2 (by VikParuchuri) is a modern open-source OCR model: 650M
parameters, 91 language scripts, runs on CPU, ~4GB RAM, and tops olmOCR-bench
with 83.3% accuracy. It is the recommended primary provider for degraded
documents because it handles stamps, stains and old typography far better
than legacy engines.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from docuflow.ocr.providers.base import AbstractOCRProvider, OCRResult

logger = logging.getLogger(__name__)

try:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    HAS_SURYA = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_SURYA = False


class SuryaProvider(AbstractOCRProvider):
    """OCR provider wrapping Surya OCR 2.

    Args:
        config: Optional dict with ``langs`` (default ``["en"]``),
            ``device`` (default CPU), ``batch_size`` (default 1) and
            ``max_lines`` (default None).
    """

    name = "surya"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._detection_predictor = None
        self._recognition_predictor = None

    def available(self) -> bool:
        return HAS_SURYA

    def _lazy_load(self) -> None:
        if self._recognition_predictor is not None:
            return
        logger.info("Loading Surya OCR 2 models (first call only)...")
        self._detection_predictor = DetectionPredictor()
        self._recognition_predictor = RecognitionPredictor()

    def extract(self, image_path: str) -> OCRResult:
        if not HAS_SURYA:
            return OCRResult(text="", confidence=0.0, provider=self.name)

        self._lazy_load()
        langs = self.config.get("langs", ["en"])
        max_lines = self.config.get("max_lines")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Surya's detection/recognition API works on image paths.
            predictions = self._recognition_predictor(
                [image_path], langs, self._detection_predictor, max_lines=max_lines
            )
            lines = []
            confidences = []
            for prediction in predictions:
                for line in prediction.text_lines:
                    lines.append(line.text)
                    confidences.append(line.confidence)

            text = "\n".join(lines).strip()
            confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
            logger.info("Surya extracted %d chars, confidence=%.3f", len(text), confidence)
            return OCRResult(
                text=text,
                confidence=confidence,
                provider=self.name,
                per_word=[{"confidence": c} for c in confidences],
            )
