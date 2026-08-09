"""Unit tests for the cascade OCR orchestration."""

from __future__ import annotations

from docuflow.ocr.cascade import CascadeConfig, CascadeOCR
from docuflow.ocr.providers.base import AbstractOCRProvider, OCRResult


class _StubProvider(AbstractOCRProvider):
    """Provider returning a canned result."""

    def __init__(self, name: str, text: str = "", confidence: float = 0.0) -> None:
        super().__init__()
        self.name = name
        self._text = text
        self._confidence = confidence
        self.calls = 0

    def extract(self, image_path: str) -> OCRResult:
        self.calls += 1
        return OCRResult(text=self._text, confidence=self._confidence, provider=self.name)


class TestCascade:
    def test_fast_provider_accepted_when_confident(self, tmp_path):
        fast = _StubProvider("fast", text="RESULT", confidence=0.95)
        accurate = _StubProvider("accurate", text="SHOULD NOT RUN", confidence=0.99)
        cascade = CascadeOCR(fast, accurate, config=CascadeConfig(fast_threshold=0.90))
        result = cascade.extract(str(tmp_path / "doc.png"))
        assert result.text == "RESULT"
        assert accurate.calls == 0

    def test_escalates_when_fast_is_uncertain(self, tmp_path):
        fast = _StubProvider("fast", text="PARTIAL", confidence=0.60)
        accurate = _StubProvider("accurate", text="FULL TEXT", confidence=0.95)
        cascade = CascadeOCR(fast, accurate, config=CascadeConfig(fast_threshold=0.90))
        result = cascade.extract(str(tmp_path / "doc.png"))
        assert result.text == "FULL TEXT"
        assert accurate.calls == 1

    def test_llm_corrector_applied(self, tmp_path):
        fast = _StubProvider("fast", text="bad txt", confidence=0.50)
        accurate = _StubProvider("accurate", text="g00d txt", confidence=0.80)
        cascade = CascadeOCR(
            fast,
            accurate,
            llm_corrector=lambda text, ctx: text.replace("g00d", "good"),
        )
        result = cascade.extract(str(tmp_path / "doc.png"))
        assert result.text == "good txt"
        assert result.provider == "accurate+llm"

    def test_empty_result_when_all_fail(self, tmp_path):
        fast = _StubProvider("fast", text="", confidence=0.0)
        accurate = _StubProvider("accurate", text="", confidence=0.0)
        cascade = CascadeOCR(fast, accurate)
        result = cascade.extract(str(tmp_path / "doc.png"))
        assert result.empty
