# The Learning System

The learning subsystem is what makes DocuFlow a *self-learning* document
processing framework rather than a thin OCR wrapper. It was extracted from a
production system that processed high volumes of degraded documents and
needed its OCR parameters to improve without an engineer hand-tuning them.

## The problem

OCR parameters interact. The region of interest (ROI) where a field lives,
the preprocessing pipeline (binarization, denoising), the confidence
threshold, the correction rules — each one affects accuracy, and their
optimal values drift as the document population changes (new scanner, older
batch, different stamps).

Hand-tuning is slow and fragile. The learning system automates it from real
outcomes: successes and failures ARE the training signal.

## The loop

```
                ┌──────────────────────────────┐
                │        Extraction runs       │
                └──────────────┬───────────────┘
                               │ success / failure
                               v
                ┌──────────────────────────────┐
                │  ManualFeedback captures     │
                │  human corrections (if any)  │
                └──────────────┬───────────────┘
                               v
                ┌──────────────────────────────┐
                │  LearningEngine.analyze()    │
                │  - error patterns            │
                │  - image characteristics     │
                │  - success rates per field   │
                └──────────────┬───────────────┘
                               v
                ┌──────────────────────────────┐
                │  Parameter adjustments       │
                │  - ROI expansion             │
                │  - preprocessing switches    │
                │  - correction rules          │
                └──────────────┬───────────────┘
                               │ persisted as JSON
                               v
                ┌──────────────────────────────┐
                │   Next batch uses new params │
                └──────────────────────────────┘
```

## The four components

### 1. LearningEngine — parameter tuning from a reference corpus

Input: a folder of reference images per field, split into `correct/` and
`error/`. For each field it:

1. **Counts outcomes** — success rate per field.
2. **Mines error patterns** at digit level:
   - *Added prefixes*: the OCR read `012345` when the expected value is
     `12345` — a prefix `0` was glued on.
   - *Added suffixes*: `123450` vs `12345`.
   - *Missing digits*: `1234` vs `12345` (leading or trailing).
   - *Digit confusions*: `12345` vs `12355` — `4` misread as `5`.
3. **Measures image characteristics** of failing vs successful documents:
   brightness, contrast, sharpness, text density (Pillow + numpy only).
4. **Generates adjustments**:
   - ROI expansion when success rate is low (5% or 10% by priority).
   - Prefix/suffix strip rules learned from the mined patterns.
   - Digit-confusion rules for the top confusions.
   - CLAHE preprocessing when failing images are darker.
   - Reduced blur when failing images are blurrier.
5. **Applies and persists** adjustments to `ocr_config_learned.json`,
   patterns to `error_patterns.json`, stats to `learning_stats.json`.

### 2. AdvancedLearning — rules from manual corrections

When an operator corrects a misread value, `ManualFeedback.save_correction()`
stores the full record: expected, OCR-detected, corrected, per-field error
flags. `AdvancedLearning.train_from_corrections()` replays those records and:

- Counts substitution patterns (`detected -> corrected`).
- Promotes repeated patterns (frequency >= 2) into autocorrection rules,
  each carrying an `expected_precision` score.
- At runtime, rules with `expected_precision > 0.7` are applied automatically
  to new extractions.
- Visual validation against stored reference images (template matching +
  histogram correlation) with per-field thresholds; similar references (by
  edit distance) are used when exact ones don't exist, with a relaxed
  threshold.

### 3. PrecisionMonitor — drift detection

A sliding window (default 20 results) tracks FULL / PARTIAL / NONE / MANUAL
outcomes. Alerts fire when:

- Window full-precision drops 15+ points below the global average.
- More than 60% of the window failed completely.
- 4+ of the last 5 results failed completely.
- Full precision is below 25% with 15+ samples.
- Positive trend: window beats global by 10+ points (improvement signal).

State persists to `precision_stats.json`, so long-running pipelines compare
across restarts.

### 4. ManualFeedback — capturing human knowledge

Stores each correction as `<id>_metadata.json` under
`references/manual_corrections/` with expected/detected/corrected values and
error flags. `analyze_corrections()` produces suggestions ("field X fails
often — adjust its ROI") that operators can act on.

## JSON formats

### ocr_config_learned.json

```json
{
  "version": "1.0",
  "last_updated": "2026-08-09T10:00:00",
  "fields": {
    "invoice_number": {
      "roi": {"top": 0.0, "bottom": 0.40, "left": 0.0, "right": 0.60},
      "preprocessing": {"threshold": "adaptive", "blur_kernel": 3, "clahe": false},
      "min_confidence": 0.6,
      "correction_rules": [
        {"type": "strip_prefix", "pattern": "0"}
      ]
    }
  },
  "adjustment_history": [
    {"timestamp": "2026-08-09T10:00:00", "adjustments": 3}
  ]
}
```

### learning_stats.json

```json
{
  "total_analyses": 12,
  "last_analysis": "2026-08-09T10:00:00",
  "improvements_applied": 14,
  "initial_success_rate": 70.0,
  "current_success_rate": 92.0
}
```

> Example trajectory: an initial pipeline at ~70% accuracy reached ~92% after
> analyzing 500 documents' worth of outcomes. Exact numbers depend on your
> document population; the mechanism is the point.

### error_patterns.json

```json
{
  "fields": {
    "invoice_number": {
      "added_prefixes": {"0": 5},
      "added_suffixes": {},
      "missing_digits": {"end": 2},
      "digit_confusions": [
        {"expected": "1", "detected": "7", "count": 4}
      ],
      "errors_analyzed": 12
    }
  }
}
```

## Why JSON, not a model file?

The learned state is human-inspectable and diffable. You can see exactly why
the pipeline changed its behavior, revert a rule by editing the file, and
version it alongside your pipeline config. When a document population
changes drastically, deleting the learned files is a clean reset.

## Honest expectations

- The learning loop needs **signal**: a reference corpus or manual
  corrections. With zero feedback it learns nothing.
- Learned rules are **population-specific**. What your OCR learned for
  invoices from scanner A may not transfer to scanner B; that is why reset
  is cheap.
- The engine tunes **parameters, not models**. It does not train a neural
  network; it decides that this ROI, this preprocessing, these correction
  rules work best for the current population.
