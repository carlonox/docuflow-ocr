"""Base abstraction for OCR providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OCRResult:
    """Result of a single OCR extraction pass.

    Attributes:
        text: The full text extracted from the image.
        confidence: Average word confidence in the [0, 1] range.
        provider: Name of the provider that produced this result.
        per_word: Optional per-word confidence breakdown for debugging.
        raw: Provider-specific raw payload (kept out of the common API).
    """

    text: str
    confidence: float = 0.0
    provider: str = "unknown"
    per_word: List[Dict[str, float]] = field(default_factory=list)
    raw: Optional[object] = None

    @property
    def empty(self) -> bool:
        """True when no text was extracted."""
        return not self.text.strip()


class AbstractOCRProvider(ABC):
    """Base class every OCR provider must implement.

    Providers wrap a concrete OCR engine (cloud API, local model, legacy
    binary) behind a common interface so the cascade pipeline can treat them
    interchangeably.
    """

    name: str = "abstract"

    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def extract(self, image_path: str) -> OCRResult:
        """Extract text from an image file.

        Args:
            image_path: Path to a PNG/JPEG image.

        Returns:
            An :class:`OCRResult` with the extracted text and confidence.
        """
        raise NotImplementedError

    def available(self) -> bool:
        """Whether this provider can run in the current environment.

        Defaults to True; providers with optional dependencies override this
        to report their real availability.
        """
        return True
