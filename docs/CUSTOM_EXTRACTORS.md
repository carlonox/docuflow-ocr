# Custom Extractors

An extractor turns raw OCR text into structured, validated fields. DocuFlow
ships three fictional examples (`src/docuflow/examples/`); this guide shows
how to write your own.

## The lifecycle

```
process(image_path)
 ├─ cascade.extract()          # OCR -> raw text
 ├─ extract_fields(raw_text)   # text -> structured fields   (YOU implement)
 ├─ validate_fields(fields)    # fields -> validity          (YOU implement)
 ├─ _decide_success()          # all fields valid?
 └─ learning hook              # outcome recorded
```

## Minimal extractor

```python
import re
from typing import Dict

from docuflow.ocr.base_extractor import AbstractDocumentExtractor
from docuflow.validation.validators import RegexValidator


class LicensePlateExtractor(AbstractDocumentExtractor):
    """Extracts a fictional license plate (format PLT-1234)."""

    document_type = "license_plate"

    def __init__(self, cascade, validator=None, learning=None):
        super().__init__(cascade, validator, learning, fields=["plate", "owner"])

    def extract_fields(self, raw_text: str) -> Dict[str, str]:
        plate = ""
        match = re.search(r"\bPLT-\d{4}\b", raw_text)
        if match:
            plate = match.group(0)

        owner = ""
        for line in raw_text.splitlines():
            if line.strip().lower().startswith("owner"):
                owner = line.split(":", 1)[1].strip()

        return {"plate": plate, "owner": owner}

    def validate_fields(self, fields: Dict[str, str]) -> Dict[str, object]:
        plate_result = RegexValidator(
            r"PLT-\d{4}", description="license plate"
        ).validate("plate", fields.get("plate"))
        return {
            "plate": plate_result.to_dict(),
            "owner": {"valid": bool(fields.get("owner")), "value": fields.get("owner")},
        }
```

## Validation rules

`src/docuflow/validation/validators.py` provides composable rules:

- `RegexValidator` — full-match against a pattern.
- `LengthValidator` — length bounds, optionally digits-only.
- `ChoiceValidator` — allowed values.
- `ChecksumValidator` — custom checksum function.
- `CompositeValidator` — AND/OR of rules.

```python
from docuflow.validation.validators import CompositeValidator, LengthValidator, RegexValidator

invoice_number = CompositeValidator([
    RegexValidator(r"INV-\d{4}-\d{4}"),
    LengthValidator(min_length=12, max_length=12),
])
```

## Naming fields

Field names are the contract between extractors, learning and monitoring.
Keep them stable: the learning engine stores per-field parameters and
patterns keyed by name, and the precision monitor reports per-field accuracy
with the same names.

## Tips

- **Extract defensively**: OCR text has stray characters; use
  `docuflow.ocr.normalization.normalize_ocr_text()` before matching.
- **Prefer regexes with anchors**: `fullmatch` avoids partial matches.
- **Return dicts with the same keys always**: missing values as `""`, so
  validation can distinguish "empty" from "invalid".
- **Never hardcode real document specifics**: if your document type carries
  personally identifying formats, define them in your private config, not in
  the public framework. See [SECURITY.md](SECURITY.md).
