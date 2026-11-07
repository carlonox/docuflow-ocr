# Running a large corpus through DocuFlow

How to process thousands of documents without paying OCR for pages that already
have text, and without restarting from zero when a run dies. Written from a real
corpus of tens of thousands of PDFs where only about a fifth needed OCR.

## 1. Measure before you run

Split the corpus in two classes first:

| Class | How to tell | Cost |
|---|---|---|
| **Has a text layer** | `page.get_text()` returns text | ~10-30 ms per PDF |
| **Image only (scan)** | the layer is empty | real OCR |

In the reference corpus fewer than one in five PDFs had no text layer. Running
OCR on everything would have multiplied the runtime by five for no gain.

```python
import fitz
doc = fitz.open(path)
has_layer = any(p.get_text().strip() for p in doc)
```

## 2. Let the cascade decide

For the pages that do need OCR, `CascadeOCR` runs the fast provider first
(RapidOCR, ~0.1-2 s) and escalates to the accurate one only when confidence is
below the threshold. In practice around 90% of pages stay on the first step.

```python
from docuflow.ocr.cascade import CascadeConfig, CascadeOCR
from docuflow.ocr.providers.rapid_ocr import RapidOCRProvider

cascade = CascadeOCR(
    fast_provider=RapidOCRProvider(),
    accurate_provider=None,          # add SuryaProvider when a GPU is around
    config=CascadeConfig(fast_threshold=0.90),
)
result = cascade.extract("page.png")   # .text, .confidence
```

If the fast provider returns nothing at all (heavy blur, bad rotation), that is
still a confidence of 0 and should escalate. Configure an accurate provider for
production runs; a single-provider cascade gives up on those pages.

## 3. Resumable batches

`examples/batch_corpus.py` implements the whole pattern:

```bash
# 1) reproducible synthetic corpus (no real documents)
python examples/make_sample_corpus.py --out samples --n 120

# 2) batch
python examples/batch_corpus.py --root samples --out demo.jsonl --sleep 0.3

# 3) report
python scripts/dashboard.py demo.jsonl -o dashboard.html --title "samples"
```

- One JSONL line per document, with the text of every page.
- `done.txt` makes the command resumable: finished documents are skipped.
- `--sleep` leaves room for the desktop; `--workers > 1` is blocked unless you
  pass `--allow-workers` (with RapidOCR on CPU, extra processes are slower and
  saturate the machine).
- The process runs at **BELOW_NORMAL** priority with `OMP_NUM_THREADS=2`.

## 4. Learning the repeated formats

The cascade pays the accurate model page by page. The next step is to learn per
**format**: if one kind of document (bank receipt, minutes, ID card) is always
resolved by the fast provider with high confidence, it does not need the accurate
stage again.

`LearningEngine` already does the hard part: mines character confusions (1 vs 7,
0 vs O), compares image features of failing documents and proposes parameter
adjustments. It expects fields with reference values, so for free text the
equivalent is:

1. Group pages by format signature (first line plus field layout).
2. Store mean confidence per format.
3. Mark formats that the fast provider resolves as settled.
4. Feed doubtful pages into `ManualFeedback` and then `LearningEngine.analyze()`.

Until that exists, the dashboard already shows which provider resolved each page
and which documents fell below 0.80.

## 5. Reference numbers

Synthetic corpus from `make_sample_corpus.py` (120 PDFs: 60 with a text layer,
60 image-only, some heavily degraded):

| Metric | Value |
|---|---|
| Documents | 120 |
| Resolved from the text layer | 60 (50%) |
| Real OCR | 60 pages |
| Speed (1 process, CPU) | ~2.2 PDF/s on clean scans |
| Mean confidence | see dashboard |
| Documents below 0.80 | 9 |

On the reference corpus (tens of thousands of PDFs, ~18% scans) a single process
needs several hours; with a GPU, one to three.
