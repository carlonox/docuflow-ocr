"""Ollama GLM-OCR provider.

Runs OCR fully local through Ollama using the ``glm-ocr`` vision model. This
is the zero-API-cost option: no cloud bills, no document data leaving the
machine. The endpoint is configurable via ``OLLAMA_HOST`` (default
``http://localhost:11434``) or the ``host`` config key.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import requests

from docuflow.ocr.providers.base import AbstractOCRProvider, OCRResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "glm-ocr:latest"
DEFAULT_HOST = "http://localhost:11434"


class OllamaOCRProvider(AbstractOCRProvider):
    """OCR provider calling an Ollama vision model (e.g. ``glm-ocr``).

    Args:
        config: Optional dict with ``model`` (default ``glm-ocr:latest``),
            ``host`` (default from ``OLLAMA_HOST`` env var or
            ``http://localhost:11434``) and ``timeout`` (default 120s).
    """

    name = "ollama_glm"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.model = str(self.config.get("model", DEFAULT_MODEL))
        self.host = str(
            self.config.get("host")
            or os.environ.get("OLLAMA_HOST")
            or DEFAULT_HOST
        )
        self.timeout: float = float(self.config.get("timeout", 120))  # type: ignore[arg-type]

    def available(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def extract(self, image_path: str) -> OCRResult:
        try:
            with open(image_path, "rb") as fh:
                image_b64 = base64.b64encode(fh.read()).decode("utf-8")
        except OSError as exc:
            logger.error("Failed reading image for Ollama: %s", exc)
            return OCRResult(text="", confidence=0.0, provider=self.name)

        payload = {
            "model": self.model,
            "prompt": "Extract all text from this document image. "
                      "Return only the text, without commentary.",
            "images": [image_b64],
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            text = (data.get("response") or "").strip()
            logger.info("Ollama '%s' extracted %d chars", self.model, len(text))
            return OCRResult(text=text, confidence=0.8, provider=self.name, raw=data)
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            return OCRResult(text="", confidence=0.0, provider=self.name)
