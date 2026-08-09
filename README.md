# DocuFlow

A self-learning document processing framework. OCR pipelines that get better
at reading your documents the more they run — tuned automatically from their
own successes and failures.

> Status: early public release. The framework was extracted from a private,
> battle-tested system that processed tens of thousands of real documents in
> high-volume production. All example code and fixtures in this repository are
> synthetic.

## Why it exists

Classic OCR pipelines are fragile. They hardcode one engine (Tesseract, a
cloud API), one preprocessing recipe, and a pile of handwritten correction
rules. When the documents change — older scans, stamps, stains, new layouts —
accuracy collapses and somebody has to hand-tune the pipeline again.

We started there: Tesseract + PaddleOCR reached ~50% accuracy on degraded
documents from the 1980s-2000s. Switching to Google Cloud Vision pushed us to
95%, but at API cost and vendor lock-in. DocuFlow is the answer we built:
**the same ~90%+ accuracy, back on your own hardware**, with a cascade of
local OCR engines plus a correction model, and a learning loop that tunes
extraction parameters automatically from real outcomes.

## What it does

- **Cascade OCR pipeline**: fast provider (RapidOCR, ~0.1-2s) → accurate
  provider (Surya OCR 2, ~10-30s CPU) → local LLM correction → deterministic
  validation. ~90% accuracy on degraded documents with 8GB RAM, CPU-only.
- **Self-learning extraction**: the learning engine analyzes successes and
  failures, mines error patterns (added prefixes, digit confusions, low
  contrast, rotation), and adjusts ROI, preprocessing and correction rules
  per document field — automatically.
- **Manual feedback loop**: every human correction becomes training data;
  repeated corrections become automatic rules.
- **Precision monitoring**: a sliding window watches for accuracy drift and
  alerts before a bad batch propagates.
- **Resumable pipelines**: state machine + checkpoints, so long batch runs
  survive crashes without redoing work.
- **Pluggable providers**: Surya, RapidOCR, Google Cloud Vision, Ollama
  (glm-ocr) and Tesseract behind one interface.

## Features at a glance

| Area | What you get |
|---|---|
| OCR engines | Surya OCR 2, RapidOCR, Google Vision, Ollama GLM, Tesseract |
| Cascade orchestration | fast → accurate → LLM corrector → validation |
| Learning | per-field parameter tuning from successes/failures |
| Feedback | manual corrections converted into automatic rules |
| Monitoring | sliding-window precision with drift alerts |
| Resumability | state machine + checkpoints |
| Storage | Google Sheets / CSV / JSON / local folders |
| Validation | regex, length, choices, checksums — composable |

## Quick start

```bash
pip install -r requirements.txt

# Generate synthetic sample documents (ID cards, invoices, forms)
python scripts/generate_sample_docs.py --output tests/fixtures

# Run the cascade on a sample document
python examples/quick_start.py tests/fixtures/sample_id_card.png
```

Minimum working pipeline:

```python
from docuflow.ocr.cascade import CascadeOCR, CascadeConfig
from docuflow.ocr.providers.rapid_ocr import RapidOCRProvider
from docuflow.ocr.providers.surya import SuryaProvider
from docuflow.examples.id_card_extractor import IDCardExtractor

cascade = CascadeOCR(
    fast_provider=RapidOCRProvider(),
    accurate_provider=SuryaProvider(),
    config=CascadeConfig(),
)
extractor = IDCardExtractor(cascade)
result = extractor.process("tests/fixtures/sample_id_card.png")
print(result.fields, result.validation)
```

## 100% local OCR with Ollama

DocuFlow ships an Ollama provider for fully offline OCR with a local vision
model (e.g. `glm-ocr`). No cloud calls, no per-page cost, no document data
leaving your machine:

```python
from docuflow.ocr.providers.ollama_glm import OllamaOCRProvider

provider = OllamaOCRProvider({"model": "glm-ocr:latest", "host": "http://localhost:11434"})
result = provider.extract("tests/fixtures/sample_invoice.png")
```

## When to use it

- You process batches of similar documents (forms, IDs, invoices) with a
  known field schema.
- Your documents are degraded (stamps, stains, old scans) and single-engine
  OCR underperforms.
- You want to run OCR locally without paying per page.
- You can collect feedback — either a reference corpus or occasional manual
  corrections — to feed the learning loop.

## When not to use it

- **Single documents**: if you OCR one page a week, the learning loop and
  cascade overhead buy you nothing. Use a plain provider directly.
- **Unstructured text extraction**: this framework is about fields and
  validation, not open-ended document understanding.
- **Tiny hardware**: the accurate cascade stage (Surya) wants ~4GB RAM.
  RapidOCR alone works on less.
- **Real-time OCR at high throughput**: the accurate stage takes seconds;
  use only the fast provider or a cloud API.

## Things DocuFlow doesn't ship with

| Feature | Status |
|---|---|
| Cloud API keys | No — bring your own, via env vars |
| Pre-trained document models | No — you define fields and rules |
| A web UI | No — library + CLI examples |
| PaddleOCR bindings | No — RapidOCR (ONNX) instead, zero Paddle conflicts |
| GPU requirements | No — everything runs on CPU (GPU optional for Surya) |

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline layout and data flow
- [LEARNING_SYSTEM.md](docs/LEARNING_SYSTEM.md) — the self-learning subsystem
- [OCR_PROVIDERS.md](docs/OCR_PROVIDERS.md) — provider comparison and tuning
- [CUSTOM_EXTRACTORS.md](docs/CUSTOM_EXTRACTORS.md) — writing your own extractor
- [SECURITY.md](docs/SECURITY.md) — credential and data handling

## Project layout

```
src/docuflow/
├── core/          # pipeline, state machine, checkpoints
├── learning/      # learning engine, advanced learning, precision monitor, manual feedback
├── ocr/           # cascade, preprocessing, normalization, providers/
├── integrations/  # browser automation, spreadsheets, local storage
├── validation/    # validators, LLM corrector, pattern examples
└── examples/      # fictional ID card, invoice and form extractors
tests/             # unit + integration tests with synthetic fixtures
scripts/           # fixture generation, provider benchmarking
```

## License

MIT — Carlos Javier Cuervo Baracaldo. See [LICENSE](LICENSE).
