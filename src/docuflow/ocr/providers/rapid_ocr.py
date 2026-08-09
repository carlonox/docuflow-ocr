"""RapidOCR provider.

RapidOCR (ONNX Runtime based, PaddleOCR models converted to ONNX) is the
fast first-line provider of the cascade: 0.1-2s per image on CPU with ~1.2GB
RAM, and 98.3% accuracy on ID documents. Unlike the original PaddleOCR
bindings it has no Paddle dependency conflicts, which makes it ideal for
projects that also use OpenCV or other native libraries.
"""

from __future__ import annotations

import logging
from typing import Optional

from docuflow.ocr.providers.base import AbstractOCRProvider, OCRResult

logger = logging.getLogger(__name__)

try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR

    HAS_RAPID_OCR = True
except ImportError:  # pragma: no cover - environment dependent
    _RapidOCR = None  # type: ignore
    HAS_RAPID_OCR = False


class RapidOCRProvider(AbstractOCRProvider):
    """OCR provider wrapping RapidOCR (ONNX).

    Args:
        config: Optional dict with ``use_cuda`` (default False) and
            ``det_use_cuda`` / ``cls_use_cuda`` / ``rec_use_cuda`` passthroughs.
    """

    name = "rapidocr"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._engine = None

    def available(self) -> bool:
        return HAS_RAPID_OCR

    def _lazy_load(self):
        if self._engine is not None:
            return
        use_cuda = self.config.get("use_cuda", False)
        logger.info("Loading RapidOCR engine (first call only)...")
        self._engine = _RapidOCR(
            det_use_cuda=use_cuda,
            cls_use_cuda=use_cuda,
            rec_use_cuda=use_cuda,
        )

    def extract(self, image_path: str) -> OCRResult:
        if not HAS_RAPID_OCR:
            return OCRResult(text="", confidence=0.0, provider=self.name)

        self._lazy_load()
        result, _elapsed = self._engine(image_path)
        if not result:
            return OCRResult(text="", confidence=0.0, provider=self.name)

        lines = [item[1] for item in result]
        confidences = [float(item[2]) for item in result if len(item) > 2]
        text = "\n".join(lines).strip()
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info("RapidOCR extracted %d chars, confidence=%.3f", len(text), confidence)
        return OCRResult(
            text=text,
            confidence=confidence,
            provider=self.name,
            per_word=[{"confidence": c} for c in confidences],
            raw=result,
        )
