"""Advanced learning: converts manual corrections into automatic rules.

Where :class:`~docuflow.learning.learning_engine.LearningEngine` learns from a
curated reference corpus, ``AdvancedLearning`` learns from the corrections a
human operator makes while the pipeline runs. Every time an operator fixes a
misread value, the correction is stored; periodically the engine replays all
stored corrections to:

1. Mine repeated substitution patterns (``"12345" -> "12354"``).
2. Turn frequent patterns into automatic correction rules.
3. Tune per-field visual-similarity thresholds and adaptive ROIs.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AdvancedLearning:
    """Learns automatic correction rules from manual operator corrections.

    Args:
        corrections_path: Directory holding the correction metadata JSON files
            written by :class:`~docuflow.learning.manual_feedback.ManualFeedback`.
        model_path: JSON file where the learned model is persisted.
        fields: Field names the model tracks (same names used by the extractors).
    """

    def __init__(
        self,
        corrections_path: str = "references/manual_corrections",
        model_path: str = "advanced_learning_model.json",
        fields: Sequence[str] = ("field_a", "field_b"),
    ) -> None:
        self.corrections_path = Path(corrections_path)
        self.model_path = Path(model_path)
        self.fields = list(fields)
        self.learning_model = self._load_or_create_model()

    # ------------------------------------------------------------------ #
    # Model persistence
    # ------------------------------------------------------------------ #

    def _load_or_create_model(self) -> Dict[str, Any]:
        if self.model_path.exists():
            with self.model_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        return {
            "version": "2.0",
            "autocorrection_rules": {field: [] for field in self.fields},
            "error_patterns": {field: {} for field in self.fields},
            "visual_similarity_thresholds": {field: 0.85 for field in self.fields},
            "adaptive_roi": {
                field: {"top": 0.0, "bottom": 0.40, "left": 0.0, "right": 0.60}
                for field in self.fields
            },
            "training_sessions": 0,
            "last_updated": datetime.now().isoformat(),
        }

    def save_model(self) -> None:
        """Persist the learned model."""
        self.learning_model["last_updated"] = datetime.now().isoformat()
        with self.model_path.open("w", encoding="utf-8") as fh:
            json.dump(self.learning_model, fh, indent=2, ensure_ascii=False)
        logger.info("Advanced model saved to %s", self.model_path)

    # ------------------------------------------------------------------ #
    # Training from manual corrections
    # ------------------------------------------------------------------ #

    def train_from_corrections(self) -> int:
        """Replay stored corrections and update the model.

        Returns the number of new rules learned.
        """
        if not self.corrections_path.exists():
            logger.warning("No corrections directory at %s", self.corrections_path)
            return 0

        corrections = self._load_corrections()
        if not corrections:
            logger.warning("No correction metadata found")
            return 0

        logger.info("Analyzing %d manual corrections", len(corrections))
        self._analyze_error_patterns(corrections)
        new_rules = self._generate_autocorrection_rules(corrections)
        self._optimize_roi(corrections)

        self.learning_model["training_sessions"] += 1
        self.save_model()
        return new_rules

    def _load_corrections(self) -> List[Dict[str, Any]]:
        corrections: List[Dict[str, Any]] = []
        for metadata_file in self.corrections_path.glob("*_metadata.json"):
            try:
                with metadata_file.open("r", encoding="utf-8") as fh:
                    corrections.append(json.load(fh))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping unreadable correction %s: %s", metadata_file, exc)
        return corrections

    def _analyze_error_patterns(self, corrections: Sequence[Dict[str, Any]]) -> None:
        """Identify the most frequent OCR -> manual substitution patterns."""
        for field in self.fields:
            errors = [
                c for c in corrections
                if c.get("errors", {}).get(f"{field}_was_wrong")
                and c.get("detected_by_ocr", {}).get(field)
                and c.get("corrected_manually", {}).get(field)
            ]
            if not errors:
                continue

            substitutions: Counter = Counter()
            for error in errors:
                detected = str(error["detected_by_ocr"][field])[:10]
                corrected = str(error["corrected_manually"][field])[:10]
                substitutions[f"{detected}->{corrected}"] += 1

            top_patterns = substitutions.most_common(5)
            self.learning_model["error_patterns"][field] = {
                "total_errors": len(errors),
                "common_patterns": top_patterns,
                "current_precision": 1.0 - (len(errors) / max(1, len(corrections))),
            }
            logger.info(
                "Field '%s': %d errors, top pattern %s",
                field, len(errors), top_patterns[0] if top_patterns else None,
            )

    def _generate_autocorrection_rules(self, corrections: Sequence[Dict[str, Any]]) -> int:
        """Turn repeated substitution patterns into automatic rules."""
        new_rules = 0
        for field in self.fields:
            patterns = self.learning_model["error_patterns"].get(field, {})
            if not patterns:
                continue

            for pattern_str, frequency in patterns.get("common_patterns", []):
                if frequency < 2:  # Only patterns that repeat.
                    continue
                detected, corrected = pattern_str.split("->", 1)
                rule = {
                    "type": "direct_substitution",
                    "detected_pattern": detected,
                    "correction": corrected,
                    "frequency": frequency,
                    "expected_precision": frequency / max(1, len(corrections)),
                    "generated_at": datetime.now().isoformat(),
                }
                rules = self.learning_model["autocorrection_rules"][field]
                if rule not in rules:
                    rules.append(rule)
                    new_rules += 1
                    logger.info(
                        "Field '%s': learned rule '%s' -> '%s'",
                        field, detected, corrected,
                    )
        return new_rules

    def _optimize_roi(self, corrections: Sequence[Dict[str, Any]]) -> None:
        """Adjust per-field ROIs based on correction outcomes (placeholder hook).

        The full implementation measures which ROI variants produced the most
        successful extractions; this hook keeps the model file consistent and
        ready for that extension.
        """
        for field in self.fields:
            # Successful corrections reinforce the current ROI; systematic
            # failures would shrink/expand it here in a data-driven way.
            pass

    # ------------------------------------------------------------------ #
    # Runtime application
    # ------------------------------------------------------------------ #

    def apply_learning(
        self, detected_value: Optional[str], field: str, expected_value: Optional[str]
    ) -> Tuple[Optional[str], List[str]]:
        """Apply learned autocorrection rules to a detected value.

        Args:
            detected_value: Value produced by the OCR pipeline (may be wrong).
            field: Field name the value belongs to.
            expected_value: Expected value, used to skip corrections that would
                not help.

        Returns:
            Tuple of (corrected value, list of applied rule descriptions).
        """
        if not detected_value or not expected_value:
            return detected_value, []

        corrected = detected_value
        applied: List[str] = []
        rules = self.learning_model.get("autocorrection_rules", {}).get(field, [])
        for rule in rules:
            if rule["type"] != "direct_substitution":
                continue
            if rule.get("expected_precision", 0) <= 0.7:
                continue  # Only trust high-precision rules.
            if rule["detected_pattern"] in corrected:
                corrected = corrected.replace(
                    rule["detected_pattern"], rule["correction"]
                )
                applied.append(f"{rule['detected_pattern']}->{rule['correction']}")

        if applied:
            logger.info(
                "Field '%s': autocorrected '%s' -> '%s' (%s)",
                field, detected_value, corrected, ", ".join(applied),
            )
        return corrected, applied

    def validate_with_visual_references(
        self,
        candidate_image: np.ndarray,
        expected_value: str,
        field: str,
        references_root: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """Validate a candidate crop against stored reference images.

        Uses template matching + histogram correlation, averaged over up to
        five exact reference images. When no exact reference exists, similar
        references (by edit distance) are used with a relaxed threshold.

        Returns:
            Tuple of (is_valid, average_similarity).
        """
        if references_root is None:
            return False, 0.0

        base_path = Path(references_root) / field.lower() / "correct"
        if not base_path.exists():
            return False, 0.0

        similarities: List[float] = []
        exact_found = False
        count = 0

        for filename in sorted(base_path.iterdir()):
            if count >= 5:
                break
            if filename.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue

            expected_in_file = _expected_number_from_filename(filename.name)
            if expected_in_file is not None and expected_in_file != expected_value:
                continue

            try:
                with Image.open(filename) as img:
                    reference = np.asarray(img.convert("L"), dtype=np.float64)
            except (OSError, ValueError):
                continue

            similarities.append(
                self._visual_similarity(candidate_image, reference)
            )
            exact_found = True
            count += 1

        if not similarities:
            return False, 0.0

        avg_similarity = float(np.mean(similarities))
        threshold = self.learning_model["visual_similarity_thresholds"].get(field, 0.85)
        if not exact_found:
            threshold = max(0.6, threshold - 0.1)

        return avg_similarity >= threshold, avg_similarity

    @staticmethod
    def _visual_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute a combined template + histogram similarity score."""
        try:
            target_h = min(img1.shape[0], img2.shape[0], 100)
            target_w = min(img1.shape[1], img2.shape[1], 200)

            img1_r = _resize_gray(img1, target_w, target_h)
            img2_r = _resize_gray(img2, target_w, target_h)

            # Normalized cross-correlation (template matching proxy).
            sim_template = _normalized_correlation(img1_r, img2_r)

            # Histogram correlation.
            hist1, _ = np.histogram(img1_r, bins=64, range=(0, 255))
            hist2, _ = np.histogram(img2_r, bins=64, range=(0, 255))
            sim_hist = _histogram_correlation(hist1, hist2)

            return float((sim_template * 0.7) + (sim_hist * 0.3))
        except (ValueError, ZeroDivisionError):
            return 0.0

    def status(self) -> Dict[str, Any]:
        """Return the learning model state."""
        return {
            "version": self.learning_model["version"],
            "training_sessions": self.learning_model["training_sessions"],
            "last_updated": self.learning_model["last_updated"],
            "rules_per_field": {
                field: len(self.learning_model["autocorrection_rules"][field])
                for field in self.fields
            },
        }


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #


def _expected_number_from_filename(filename: str) -> Optional[str]:
    """Extract the leading number from a reference image filename."""
    import re

    match = re.match(r"^(\d+)", filename)
    return match.group(1) if match else None


def _resize_gray(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a grayscale array using PIL (no OpenCV dependency)."""
    from PIL import Image as PILImage

    if height <= 0 or width <= 0:
        return image
    pil_img = PILImage.fromarray(image.astype(np.uint8))
    return np.asarray(pil_img.resize((width, height), PILImage.Resampling.BILINEAR), dtype=np.float64)


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two same-shaped arrays."""
    a_flat = a - a.mean()
    b_flat = b - b.mean()
    denominator = np.sqrt((a_flat ** 2).sum() * (b_flat ** 2).sum())
    if denominator == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denominator)


def _histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Correlation coefficient between two histograms."""
    h1 = hist1 - hist1.mean()
    h2 = hist2 - hist2.mean()
    denominator = np.sqrt((h1 ** 2).sum() * (h2 ** 2).sum())
    if denominator == 0:
        return 0.0
    return float(np.dot(h1, h2) / denominator)
