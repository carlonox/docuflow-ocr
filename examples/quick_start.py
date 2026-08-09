"""Quick start: five lines to extract fields from a document.

Usage:
    python examples/quick_start.py tests/fixtures/sample_id_card.png
"""

from __future__ import annotations

import sys

from docuflow.examples.id_card_extractor import IDCardExtractor
from docuflow.ocr.cascade import CascadeConfig, CascadeOCR
from docuflow.ocr.providers.rapid_ocr import RapidOCRProvider
from docuflow.ocr.providers.surya import SuryaProvider


def main(image_path: str) -> None:
    # 1. Build the cascade: fast -> accurate.
    cascade = CascadeOCR(
        fast_provider=RapidOCRProvider(),
        accurate_provider=SuryaProvider(),
        config=CascadeConfig(fast_threshold=0.90),
        context="national id card",
    )

    # 2. Wrap it in an extractor with validation.
    extractor = IDCardExtractor(cascade)

    # 3. Process a document.
    result = extractor.process(image_path, document_id=image_path)

    print(f"success: {result.success}")
    print(f"fields:  {result.fields}")
    print(f"validation: {result.validation}")
    print(f"provider: {result.provider_chain}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
