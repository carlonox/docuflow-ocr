"""Benchmark the cascade providers against synthetic fixtures.

Runs every available provider over the generated fixtures and prints
per-provider timing and text length, so you can see which provider chain
makes sense for your hardware before wiring a pipeline.

Usage:
    python scripts/benchmark_cascade.py tests/fixtures
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from docuflow.ocr.providers.base import AbstractOCRProvider
from docuflow.ocr.providers.google_vision import GoogleVisionProvider
from docuflow.ocr.providers.ollama_glm import OllamaOCRProvider
from docuflow.ocr.providers.rapid_ocr import RapidOCRProvider
from docuflow.ocr.providers.surya import SuryaProvider
from docuflow.ocr.providers.tesseract import TesseractProvider

PROVIDERS = [
    RapidOCRProvider,
    SuryaProvider,
    TesseractProvider,
    GoogleVisionProvider,
    OllamaOCRProvider,
]


def main(fixtures_dir: str) -> None:
    fixtures = sorted(Path(fixtures_dir).glob("*.png"))
    if not fixtures:
        print(f"No fixtures found in {fixtures_dir}; run scripts/generate_sample_docs.py first")
        return

    print(f"{'provider':<16} {'available':<10} {'docs':<6} {'total_s':<9} {'chars':<8}")
    print("-" * 55)

    for provider_class in PROVIDERS:
        provider = provider_class()
        if not provider.available():
            print(f"{provider.name:<16} {'NO':<10}")
            continue

        start = time.monotonic()
        total_chars = 0
        for fixture in fixtures:
            result = provider.extract(str(fixture))
            total_chars += len(result.text)
        elapsed = time.monotonic() - start

        print(
            f"{provider.name:<16} {'yes':<10} {len(fixtures):<6} "
            f"{elapsed:>7.2f}s  {total_chars:<8}"
        )

    print("\nTip: the cascade uses the fast provider first and escalates only "
          "when confidence is low — benchmark each stage separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures_dir", nargs="?", default="tests/fixtures")
    args = parser.parse_args()
    main(args.fixtures_dir)
