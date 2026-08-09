# OCR Providers

DocuFlow's provider layer wraps five OCR engines behind one interface.
This document compares them and explains when each one earns its place in
the cascade.

## Comparison

| Provider | Speed (CPU) | Accuracy | RAM | Cost | Best for |
|---|---|---|---|---|---|
| RapidOCR (ONNX) | 0.1-2s | good on clean | ~1.2GB | free | first pass, high volume |
| Surya OCR 2 | 10-30s | 83.3% olmOCR-bench | ~4GB | free | degraded documents, main stage |
| Google Cloud Vision | 1-3s | ~95% | n/a (cloud) | per page | maximum accuracy, budget allows |
| Ollama glm-ocr | 5-20s | good | local | free | 100% local, no data leaves machine |
| Tesseract | <1s | ok on clean | tiny | free | legacy fallback, clean scans |

## Cascade strategy

```
RapidOCR ──confidence ≥ 0.90──► accept
    │
    └──confidence < 0.90──► Surya ──► LLM corrector (doubtful fields) ──► validation
```

- **~90% of a batch** is handled by RapidOCR in under 2 seconds.
- The remaining ~10% (degraded, stained, skewed) escalates to Surya.
- Doubtful key fields get LLM correction with document context.
- Deterministic validation runs last, always.

This combination reaches ~90% accuracy on degraded documents with an
8GB-RAM CPU-only machine — the accuracy we previously only got from a cloud
API, without the per-page cost or vendor lock-in.

## Provider notes

### Surya OCR 2 (recommended primary)

- Open-source, 650M params, 91 language scripts.
- Strong on stamps, stains, old typography — the failure mode of legacy OCR.
- Install: `pip install surya-ocr`.
- Tuning: `langs`, `batch_size`, `max_lines` via provider config.

### RapidOCR (ONNX)

- PaddleOCR models converted to ONNX Runtime: same accuracy, no Paddle
  dependency conflicts (the reason the original system moved off Paddle).
- 98.3% on ID documents in published benchmarks.
- Install: `pip install rapidocr-onnxruntime`.

### Google Cloud Vision

- The accuracy ceiling (~95%), but per-page cost and data leaves your
  machine. Use when documents are legally allowed to go to the cloud and the
  budget exists.
- Credentials: `GOOGLE_APPLICATION_CREDENTIALS` env var, never a committed
  JSON file. See [SECURITY.md](SECURITY.md).

### Ollama (glm-ocr)

- Fully local OCR via Ollama's vision models. Zero cost, zero egress.
- `ollama pull glm-ocr` then point `OllamaOCRProvider` at your endpoint.
- Great for sensitive documents that must not leave the premises.

### Tesseract

- Legacy fallback. No downloads, no keys, works everywhere — but on degraded
  documents it lags far behind Surya. Keep it as the last stage for clean
  scans or as a smoke-test provider.

## Hardware guidance

| Setup | Recommended chain |
|---|---|
| 8GB RAM, CPU | RapidOCR → Surya (works, ~4GB peak) |
| 16GB+ RAM, CPU | RapidOCR → Surya → LLM corrector (Qwen2.5-3B Q4) |
| GPU (RTX 3080+) | add Ollama glm-ocr; Surya GPU mode optional |
| No OCR installed | Tesseract only (smoke tests) |

## Benchmarks

Run your own on synthetic fixtures:

```bash
python scripts/generate_sample_docs.py --output tests/fixtures
python scripts/benchmark_cascade.py tests/fixtures
```

The benchmark prints per-provider timing and text volume so you can pick the
chain for your hardware.
