# Architecture

DocuFlow organizes document processing into four layers: **providers** (OCR
engines), **cascade** (orchestration), **learning** (self-improvement) and
**integrations** (data sources and sinks).

```
                        ┌─────────────────────────────────────────────┐
                        │                 Document batch              │
                        └───────────────────┬─────────────────────────┘
                                            │
                                            v
┌──────────────────────────────────────────────────────────────────────┐
│                         Pipeline (core/)                            │
│   state machine ──► extractor.process() ──► steps ──► monitor        │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                                            v
┌──────────────────────────────────────────────────────────────────────┐
│                        CascadeOCR (ocr/)                            │
│                                                                      │
│   ┌──────────┐    low confidence?    ┌──────────┐   doubtful fields? │
│   │  Rapid   │ ────────────────────► │  Surya   │ ──────────────────►│
│   │  (fast)  │                       │ (accurate)│                   │
│   └──────────┘                       └──────────┘                   │
│        │                                                                │
│        └─────────────► LLM corrector ──► validators ──► result         │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                    success / failure       v
┌──────────────────────────────────────────────────────────────────────┐
│                      Learning subsystem (learning/)                  │
│                                                                      │
│   LearningEngine ──► parameter tuning (ROI, preprocessing, rules)    │
│   AdvancedLearning ─► autocorrection rules from manual corrections   │
│   PrecisionMonitor ─► sliding-window drift alerts                    │
│   ManualFeedback ───► correction capture                             │
└──────────────────────────────────────────────────────────────────────┘
```

## Layers in detail

### 1. Providers (`src/docuflow/ocr/providers/`)

Every engine implements `AbstractOCRProvider.extract(image_path) -> OCRResult`.
The cascade only sees `text` and `confidence`, so engines are swappable and
testable in isolation.

| Provider | Speed | Accuracy | Cost | Hardware |
|---|---|---|---|---|
| RapidOCR (ONNX) | 0.1-2s | good on clean docs | free | CPU, ~1.2GB |
| Surya OCR 2 | 10-30s | best on degraded | free | CPU, ~4GB |
| Google Vision | 1-3s | ~95% | per page | cloud |
| Ollama (glm-ocr) | 5-20s | good | free | local GPU/CPU |
| Tesseract | <1s | ok on clean | free | CPU |

### 2. Cascade (`src/docuflow/ocr/cascade.py`)

Escalation policy:

1. Run the fast provider. If `confidence >= fast_threshold` (default 0.90),
   accept and validate.
2. Otherwise run the accurate provider.
3. If configured fields are still doubtful, run the LLM corrector with
   document context.
4. Always finish with deterministic validation.

The cascade is the answer to a practical problem: paying 30 seconds of Surya
per document is wasteful when 80% of the batch is readable by RapidOCR in a
second.

### 3. Learning (`src/docuflow/learning/`)

See [LEARNING_SYSTEM.md](LEARNING_SYSTEM.md) for the full design. Short
version: every extraction outcome is recorded; periodic analysis converts
outcomes into parameter changes (ROI expansion, preprocessing switches,
correction rules) that make the next batch better.

### 4. Integrations (`src/docuflow/integrations/`)

- `browser.py` — Selenium helpers for web-based document sources.
- `spreadsheet.py` — Google Sheets / CSV / JSON with the same interface.
- `local_storage.py` — offline queue + results (JSONL).

## Data flow of one document

```
1. Pipeline picks item from queue (or spreadsheet rows).
2. extractor.process(image_path):
   a. cascade.extract() -> OCRResult
   b. extract_fields(raw_text) -> structured fields
   c. validate_fields(fields) -> per-field validity
   d. learning hook: record outcome
3. Pipeline steps post-process (enrichment, sinks).
4. PrecisionMonitor.record_result() updates the sliding window.
5. Checkpoint saved every batch_size items.
```

## Failure semantics

- A provider raising is logged and the cascade moves to the next stage.
- An extractor failing marks the document as failed in the summary; the
  pipeline continues unless a step sets `on_error="stop"`.
- The monitor raises alerts (drift, streaks) but never stops the pipeline:
  operators decide whether to pause.

## Design decisions

- **No PaddleOCR**: the original system used it; ONNX-based RapidOCR delivers
  the same accuracy without Paddle's dependency conflicts.
- **Pillow + numpy in core**: the learning loop runs anywhere; OpenCV-style
  heavy lifting stays inside providers that actually need it.
- **JSON state files**: learned config, stats and patterns are plain JSON,
  versioned by the user, easy to inspect and easy to reset.
- **Synthetic fixtures only**: tests never touch real personal data.
