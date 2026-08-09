"""LLM-based post-processing of OCR text.

The final stage of the cascade: when the accurate OCR provider leaves a field
doubtful, a local LLM repairs it using linguistic context (the document type,
expected field formats, domain vocabulary). The reference implementation
talks to any OpenAI-compatible endpoint — Ollama by default — so it runs
fully offline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class LLMCorrector:
    """Repair OCR text with a local LLM via an OpenAI-compatible endpoint.

    Args:
        model: Model name served by the endpoint.
        base_url: Endpoint base URL. Defaults to ``OLLAMA_HOST`` env var or
            ``http://localhost:11434``.
        system_prompt: Optional system prompt override.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "qwen2.5-3b-instruct-q4_k_m.gguf",
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt or (
            "You correct OCR errors in extracted document text. "
            "Fix character confusions, spacing and obvious digit mistakes "
            "using the provided context. Reply with the corrected text only."
        )

    def correct(self, text: str, context: str = "") -> str:
        """Correct OCR text.

        Args:
            text: Raw text from the OCR provider.
            context: Document type / domain hints (e.g. "invoice", "id card").

        Returns:
            The corrected text (unchanged when the endpoint is unreachable
            or the model returns nothing usable).
        """
        if not text.strip():
            return text

        user_prompt = f"Context: {context}\n\nOCR text:\n{text}" if context else text
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": user_prompt,
            "system": self.system_prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            corrected = (response.json().get("response") or "").strip()
            if corrected:
                logger.info("LLM correction applied (%d -> %d chars)", len(text), len(corrected))
                return corrected
        except requests.RequestException as exc:
            logger.warning("LLM corrector unavailable (%s); keeping raw text", exc)
        except json.JSONDecodeError as exc:
            logger.warning("LLM corrector returned invalid JSON (%s); keeping raw text", exc)
        return text
