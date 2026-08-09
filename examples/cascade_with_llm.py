"""Full pipeline example: cascade + LLM corrector + learning loop.

Shows the complete DocuFlow stack on synthetic fixtures:

- cascade with RapidOCR (fast) and Surya (accurate),
- local LLM corrector via Ollama,
- learning engine that tunes parameters from outcomes,
- precision monitor tracking drift.

Usage:
    python examples/cascade_with_llm.py tests/fixtures/sample_invoice.png
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from docuflow.examples.invoice_extractor import InvoiceExtractor
from docuflow.learning.learning_engine import LearningEngine
from docuflow.learning.precision_monitor import PrecisionMonitor
from docuflow.ocr.cascade import CascadeConfig, CascadeOCR
from docuflow.ocr.providers.ollama_glm import OllamaOCRProvider
from docuflow.ocr.providers.rapid_ocr import RapidOCRProvider
from docuflow.ocr.providers.surya import SuryaProvider
from docuflow.validation.llm_corrector import LLMCorrector


def main(image_path: str) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="docuflow_example_"))

    # Local LLM corrector (Ollama; falls back to identity when unreachable).
    llm_corrector = LLMCorrector(model="qwen2.5-3b-instruct-q4_k_m.gguf")

    cascade = CascadeOCR(
        fast_provider=RapidOCRProvider(),
        accurate_provider=SuryaProvider(),
        llm_corrector=llm_corrector.correct,
        config=CascadeConfig(fast_threshold=0.90, llm_required_fields=["invoice_number"]),
        context="invoice",
    )

    # Learning + monitoring wired in.
    learning = LearningEngine(
        fields=["invoice_number", "total_amount", "vendor"],
        config_path=str(workdir / "ocr_config_learned.json"),
        stats_path=str(workdir / "learning_stats.json"),
        error_patterns_path=str(workdir / "error_patterns.json"),
    )
    monitor = PrecisionMonitor(
        stats_path=str(workdir / "precision_stats.json"),
        fields=["invoice_number", "total_amount", "vendor"],
    )

    extractor = InvoiceExtractor(cascade)
    result = extractor.process(image_path, document_id=image_path)

    # Feed the outcome into monitoring (learning hook runs inside extractor).
    monitor.record_result(
        [
            result.validation.get(f, {}).get("valid", False)
            for f in extractor.fields
        ]
    )

    print("=" * 60)
    print("DOCUFLOW CASCADE + LLM EXAMPLE")
    print("=" * 60)
    print(f"success:      {result.success}")
    print(f"fields:       {result.fields}")
    print(f"validation:   {result.validation}")
    print(f"provider:     {result.provider_chain}")
    print(f"confidence:   {result.confidence:.3f}")
    print(f"\nlearning state: {learning.summary()}")
    print(f"monitor state:  {monitor.summary()}")
    print(f"\nlearned files (temporary workspace): {workdir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
