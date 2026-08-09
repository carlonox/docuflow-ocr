"""Manual feedback loop: capture human corrections as training data.

Even the best OCR pipeline gets values wrong. The practical answer is not
more models — it is capturing the corrections operators already make. This
module stores every manual fix (with what the OCR detected, what the operator
entered, and which fields were wrong) so the advanced learning engine can
convert repeated corrections into automatic rules.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class ManualFeedback:
    """Store and analyze human corrections of OCR extractions.

    Args:
        corrections_path: Directory where correction images and metadata are
            stored. Metadata files are named ``<correction_id>_metadata.json``.
        stats_path: JSON file with aggregate correction statistics.
    """

    def __init__(
        self,
        corrections_path: str = "references/manual_corrections",
        stats_path: str = "manual_corrections_stats.json",
    ) -> None:
        self.corrections_path = Path(corrections_path)
        self.stats_path = Path(stats_path)

    # ------------------------------------------------------------------ #
    # Saving corrections
    # ------------------------------------------------------------------ #

    def save_correction(
        self,
        expected: Dict[str, str],
        detected: Dict[str, Optional[str]],
        corrected: Dict[str, str],
        document_path: Optional[str] = None,
        rois: Optional[Dict[str, Sequence[int]]] = None,
    ) -> Optional[str]:
        """Store one manual correction.

        Args:
            expected: Ground-truth values per field (from the source of truth,
                e.g. a spreadsheet row).
            detected: Values the OCR pipeline produced per field.
            corrected: Values the operator actually entered per field.
            document_path: Optional path to the source document image.
            rois: Optional per-field ROI crop regions (x, y, w, h).

        Returns:
            The correction ID, or None on failure.
        """
        try:
            self.corrections_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            correction_id = f"correction_{timestamp}"

            errors = {
                f"{field}_was_wrong": corrected.get(field) != expected.get(field)
                for field in expected
            }
            errors["both_wrong"] = all(
                corrected.get(field) != expected.get(field) for field in expected
            )

            metadata: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "correction_id": correction_id,
                "expected": expected,
                "detected_by_ocr": {
                    field: detected.get(field) or "N/A" for field in expected
                },
                "corrected_manually": corrected,
                "errors": errors,
                "source_document": document_path,
                "rois": rois or {},
            }

            metadata_path = self.corrections_path / f"{correction_id}_metadata.json"
            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, ensure_ascii=False)

            self._update_stats(errors)
            logger.info("Manual correction saved: %s", correction_id)
            return correction_id

        except OSError as exc:
            logger.error("Failed to save manual correction: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def _update_stats(self, errors: Dict[str, bool]) -> None:
        stats = self._load_stats()
        stats["total_corrections"] += 1
        for field, was_wrong in errors.items():
            if field == "both_wrong":
                continue
            stats.setdefault(f"{field}_corrected", 0)
            if not was_wrong:
                stats[f"{field}_corrected"] += 1
        if not errors.get("both_wrong", False):
            stats["fully_corrected"] += 1
        stats["last_updated"] = datetime.now().isoformat()

        with self.stats_path.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)

        total = stats["total_corrections"]
        success = stats["fully_corrected"]
        logger.info(
            "Corrections accumulated: %d/%d fully successful (%.1f%%)",
            success, total, (success / total * 100.0) if total else 0.0,
        )

    def _load_stats(self) -> Dict[str, Any]:
        if self.stats_path.exists():
            try:
                with self.stats_path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "total_corrections": 0,
            "fully_corrected": 0,
            "started_at": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #

    def analyze_corrections(self) -> Dict[str, Any]:
        """Analyze stored corrections and produce improvement suggestions."""
        if not self.corrections_path.exists():
            return {"error": "No corrections directory"}

        corrections = self._load_corrections()
        if not corrections:
            return {"error": "No correction metadata found"}

        field_errors: Dict[str, int] = {}
        both_errors = 0
        for correction in corrections:
            errors = correction.get("errors", {})
            for key, was_wrong in errors.items():
                if key == "both_wrong":
                    both_errors += int(was_wrong)
                elif was_wrong:
                    field_errors[key] = field_errors.get(key, 0) + 1

        suggestions: List[str] = []
        threshold = len(corrections) * 0.5
        for field, count in field_errors.items():
            if count > threshold:
                suggestions.append(
                    f"Field '{field.replace('_was_wrong', '')}' fails often "
                    f"({count}/{len(corrections)}) — adjust its ROI or preprocessing"
                )
        if both_errors > len(corrections) * 0.3:
            suggestions.append(
                "Systemic problem: multiple fields wrong together — "
                "review the document preprocessing stage"
            )

        return {
            "total_corrections": len(corrections),
            "field_errors": field_errors,
            "both_errors": both_errors,
            "suggestions": suggestions,
        }

    def _load_corrections(self) -> List[Dict[str, Any]]:
        corrections: List[Dict[str, Any]] = []
        for metadata_file in self.corrections_path.glob("*_metadata.json"):
            try:
                with metadata_file.open("r", encoding="utf-8") as fh:
                    corrections.append(json.load(fh))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping unreadable correction %s: %s", metadata_file, exc)
        return corrections
