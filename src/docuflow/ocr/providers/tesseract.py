"""Tesseract provider.

Tesseract is the legacy fallback of the cascade: it needs no model downloads
and no API keys, and is fine for clean, well-scanned documents. For degraded
documents it is outperformed by Surya/RapidOCR, which is why it sits at the
end of the provider chain.
"""

from __future__ import annotations

import logging
import shutil
from typing import Optional

from PIL import Image

from docuflow.ocr.multi_psm import extract_text_multi_psm
from docuflow.ocr.providers.base import AbstractOCRProvider, OCRResult

logger = logging.getLogger(__name__)

try:
    import pytesseract

    HAS_PYTESSERACT = True
except ImportError:  # pragma: no cover - environment dependent
    pytesseract = None  # type: ignore
    HAS_PYTESSERACT = False


class TesseractProvider(AbstractOCRProvider):
    """OCR provider wrapping Tesseract via pytesseract.

    Args:
        config: Optional dict with ``lang`` (default ``"eng"``),
            ``tesseract_cmd`` (auto-detected from PATH when omitted) and
            ``psm_modes`` (default ``(6, 4, 3, 11)``).
    """

    name = "tesseract"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._tesseract_cmd: Optional[str] = self.config.get("tesseract_cmd")  # type: ignore[assignment]
        if not self._tesseract_cmd:
            self._tesseract_cmd = shutil.which("tesseract")

    def available(self) -> bool:
        return HAS_PYTESSERACT and self._tesseract_cmd is not None

    def extract(self, image_path: str) -> OCRResult:
        if not self.available():
            return OCRResult(text="", confidence=0.0, provider=self.name)

        try:
            with Image.open(image_path) as img:
                gray = img.convert("L")
        except (OSError, ValueError) as exc:
            logger.warning("Could not open image %s: %s", image_path, exc)
            return OCRResult(text="", confidence=0.0, provider=self.name)

        text, confidence = extract_text_multi_psm(
            gray,
            tesseract_cmd=self._tesseract_cmd,
            psm_modes=tuple(self.config.get("psm_modes", (6, 4, 3, 11))),  # type: ignore[arg-type]
            lang=str(self.config.get("lang", "eng")),
        )
        logger.info("Tesseract extracted %d chars, confidence=%.3f", len(text), confidence)
        return OCRResult(text=text, confidence=confidence, provider=self.name)
