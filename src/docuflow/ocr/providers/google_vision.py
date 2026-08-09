"""Google Cloud Vision provider.

Cloud OCR is the highest-accuracy option (~95% on degraded documents) but
costs per page and creates vendor lock-in. This provider is included for
projects that already have a Google Cloud account; credentials are read from
the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable, never from the
repository.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from docuflow.ocr.providers.base import AbstractOCRProvider, OCRResult

logger = logging.getLogger(__name__)

try:
    from google.cloud import vision

    HAS_GOOGLE_VISION = True
except ImportError:  # pragma: no cover - environment dependent
    vision = None  # type: ignore
    HAS_GOOGLE_VISION = False


class GoogleVisionProvider(AbstractOCRProvider):
    """OCR provider wrapping Google Cloud Vision ``document_text_detection``.

    Args:
        config: Optional dict with ``credentials_path``. When omitted the
            standard ``GOOGLE_APPLICATION_CREDENTIALS`` env var is used.
    """

    name = "google_vision"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._client = None

    def available(self) -> bool:
        if not HAS_GOOGLE_VISION:
            return False
        return bool(
            self.config.get("credentials_path")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )

    def _lazy_load(self):
        if self._client is not None:
            return
        credentials_path = self.config.get("credentials_path")
        if credentials_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self._client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            self._client = vision.ImageAnnotatorClient()
        logger.info("Google Vision client initialized")

    def extract(self, image_path: str) -> OCRResult:
        if not self.available():
            return OCRResult(text="", confidence=0.0, provider=self.name)

        self._lazy_load()
        try:
            with open(image_path, "rb") as fh:
                content = fh.read()
            image = vision.Image(content=content)
            response = self._client.document_text_detection(image=image)
            if response.error.message:
                logger.error("Vision API error: %s", response.error.message)
                return OCRResult(text="", confidence=0.0, provider=self.name)

            text = response.full_text_annotation.text if response.full_text_annotation else ""
            confidence = _average_word_confidence(response)

            logger.info("Google Vision extracted %d chars, confidence=%.3f", len(text), confidence)
            return OCRResult(
                text=text or "",
                confidence=confidence,
                provider=self.name,
                raw=response,
            )
        except OSError as exc:
            logger.error("Failed reading image for Vision: %s", exc)
            return OCRResult(text="", confidence=0.0, provider=self.name)


def _average_word_confidence(response) -> float:
    """Average word confidence over all pages/blocks/paragraphs/words."""
    if not response.full_text_annotation or not response.full_text_annotation.pages:
        return 0.0
    confidences = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    confidences.append(word.confidence)
    return sum(confidences) / len(confidences) if confidences else 0.0
