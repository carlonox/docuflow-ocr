"""Learning engine: analyzes extraction successes and failures and tunes OCR parameters.

The engine is the core of the self-learning loop. For each configured document
field it:

1. Counts correct and failed extractions from a reference corpus.
2. Mines concrete error patterns: added prefixes/suffixes, missing digits and
   repeated digit confusions (e.g. ``1`` misread as ``7``).
3. Measures image characteristics of the failing documents (brightness,
   contrast, sharpness, text density) against the successful ones.
4. Generates and applies parameter adjustments: ROI expansion, preprocessing
   switches (CLAHE), and correction rules learned from the patterns.

All learned state is persisted as JSON so a long-running pipeline keeps
improving across runs without retraining.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

DEFAULT_FIELDS: Tuple[str, ...] = ("field_a", "field_b")

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "1.0",
    "last_updated": None,
    "fields": {
        # One section per field; fields are declared in the pipeline config.
        # "field_a": {
        #     "roi": {"top": 0.0, "bottom": 0.40, "left": 0.0, "right": 0.60},
        #     "preprocessing": {"threshold": "adaptive", "blur_kernel": 3, "clahe": False},
        #     "min_confidence": 0.6,
        #     "correction_rules": [],
        # }
    },
    "adjustment_history": [],
}

DEFAULT_STATS: Dict[str, Any] = {
    "total_analyses": 0,
    "last_analysis": None,
    "improvements_applied": 0,
    "initial_success_rate": 0.0,
    "current_success_rate": 0.0,
}

DEFAULT_ERROR_PATTERNS: Dict[str, Any] = {
    "fields": {
        # "field_a": {
        #     "added_prefixes": {"0": 5, "01": 2},
        #     "added_suffixes": {},
        #     "missing_digits": {"start": 1, "end": 3},
        #     "digit_confusions": [{"expected": "1", "detected": "7", "count": 4}],
        # }
    },
}


def extract_numbers_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract expected and detected numbers from a reference image filename.

    Expected formats:

    - Correct: ``7173638_1764821192.png`` -> ``("7173638", None)``
    - Error:   ``7173638_detected_010494800542_1764821192.png``
               -> ``("7173638", "010494800542")``
    """
    match_error = re.match(r"^(\d+)_detected_(\d+)_\d+\.png$", filename)
    if match_error:
        return match_error.group(1), match_error.group(2)

    match_correct = re.match(r"^(\d+)_\d+\.png$", filename)
    if match_correct:
        return match_correct.group(1), None

    return None, None


class LearningEngine:
    """Tune OCR parameters from extraction successes and failures.

    Args:
        fields: Field names to learn about. Each field gets its own parameter
            section and error patterns.
        config_path: JSON file with the learned configuration.
        stats_path: JSON file with learning statistics.
        error_patterns_path: JSON file with mined error patterns.
    """

    def __init__(
        self,
        fields: Sequence[str] = DEFAULT_FIELDS,
        config_path: str = "ocr_config_learned.json",
        stats_path: str = "learning_stats.json",
        error_patterns_path: str = "error_patterns.json",
    ) -> None:
        self.fields = list(fields)
        self.config_path = Path(config_path)
        self.stats_path = Path(stats_path)
        self.error_patterns_path = Path(error_patterns_path)

        self.config = self._load_config()
        self.stats = self._load_stats()
        self.error_patterns = self._load_error_patterns()

        # Make sure every declared field exists in the learned state.
        for field in self.fields:
            self.config["fields"].setdefault(field, self._default_field_config())
            self.error_patterns["fields"].setdefault(field, self._default_field_patterns())

        # Adjustments staged during reference analysis, applied in batch.
        self._pending_adjustments: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------ #
    # Loading / saving
    # ------------------------------------------------------------------ #

    def _default_field_config(self) -> Dict[str, Any]:
        return {
            "roi": {"top": 0.0, "bottom": 0.40, "left": 0.0, "right": 0.60},
            "preprocessing": {
                "threshold": "adaptive",
                "blur_kernel": 3,
                "clahe": False,
            },
            "min_confidence": 0.6,
            "correction_rules": [],
        }

    def _default_field_patterns(self) -> Dict[str, Any]:
        return {
            "added_prefixes": {},
            "added_suffixes": {},
            "missing_digits": {},
            "digit_confusions": [],
            "errors_analyzed": 0,
        }

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                loaded.setdefault("fields", {})
                loaded.setdefault("adjustment_history", [])
                return loaded
        return json.loads(json.dumps(DEFAULT_CONFIG))

    def _load_stats(self) -> Dict[str, Any]:
        if self.stats_path.exists():
            with self.stats_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                for key, value in DEFAULT_STATS.items():
                    loaded.setdefault(key, value)
                return loaded
        return json.loads(json.dumps(DEFAULT_STATS))

    def _load_error_patterns(self) -> Dict[str, Any]:
        if self.error_patterns_path.exists():
            with self.error_patterns_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                loaded.setdefault("fields", {})
                return loaded
        return json.loads(json.dumps(DEFAULT_ERROR_PATTERNS))

    def save_config(self) -> None:
        """Persist the learned configuration."""
        self.config["version"] = "1.0"
        with self.config_path.open("w", encoding="utf-8") as fh:
            json.dump(self.config, fh, indent=2, ensure_ascii=False)
        logger.info("Configuration saved to %s", self.config_path)

    def save_error_patterns(self) -> None:
        """Persist the mined error patterns."""
        with self.error_patterns_path.open("w", encoding="utf-8") as fh:
            json.dump(self.error_patterns, fh, indent=2, ensure_ascii=False)
        logger.info("Error patterns saved to %s", self.error_patterns_path)

    def save_stats(self) -> None:
        """Persist learning statistics."""
        with self.stats_path.open("w", encoding="utf-8") as fh:
            json.dump(self.stats, fh, indent=2, ensure_ascii=False)
        logger.info("Statistics saved to %s", self.stats_path)

    # ------------------------------------------------------------------ #
    # Reference analysis
    # ------------------------------------------------------------------ #

    def analyze_references(self, references_root: str) -> int:
        """Analyze a reference corpus and apply parameter adjustments.

        Expected layout under ``references_root``::

            references_root/
                <field>/
                    correct/        # images that extracted correctly
                    error/          # images that extracted incorrectly

        Returns the number of adjustments applied.
        """
        logger.info("Starting reference analysis under %s", references_root)
        root = Path(references_root)

        for field in self.fields:
            correct_path = root / field / "correct"
            error_path = root / field / "error"
            if not correct_path.exists() and not error_path.exists():
                logger.warning("No references found for field '%s'", field)
                continue
            self._analyze_field(field, correct_path, error_path)

        applied = self._apply_all_adjustments()
        self.save_config()
        self.save_error_patterns()
        self._update_stats(root)
        self.save_stats()
        logger.info("Reference analysis finished: %d adjustments applied", applied)
        return applied

    def _analyze_field(
        self, field: str, correct_path: Path, error_path: Path
    ) -> None:
        """Analyze one field's reference corpus and mine error patterns."""
        correct_count = _count_images(correct_path)
        error_count = _count_images(error_path)
        total = correct_count + error_count
        if total == 0:
            return

        success_rate = correct_count / total * 100.0
        logger.info(
            "Field '%s': %d/%d correct (%.1f%%), %d errors",
            field, correct_count, total, success_rate, error_count,
        )

        patterns: Dict[str, Any] = self._default_field_patterns()
        if error_count > 0:
            patterns = self._mine_error_patterns(error_path, patterns)
            self.error_patterns["fields"][field] = patterns

        if correct_count > 0 and error_count > 0:
            correct_features = _analyze_image_features(correct_path)
            error_features = _analyze_image_features(error_path)
            adjustments = self._generate_adjustments(
                field, correct_features, error_features, success_rate, patterns
            )
            self._pending_adjustments[field] = adjustments

    def _mine_error_patterns(self, error_path: Path, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Mine concrete digit-level error patterns from failing images."""
        added_prefixes: Counter = Counter()
        added_suffixes: Counter = Counter()
        missing_digits: Dict[str, int] = defaultdict(int)
        digit_confusions: Counter = Counter()

        for filename in sorted(error_path.iterdir()):
            if filename.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            expected, detected = extract_numbers_from_filename(filename.name)
            if not expected or not detected:
                continue

            patterns["errors_analyzed"] += 1

            # Pattern 1: prefix glued to an otherwise correct number.
            if detected.endswith(expected):
                prefix = detected[: -len(expected)]
                if prefix:
                    added_prefixes[prefix] += 1

            # Pattern 2: suffix glued to an otherwise correct number.
            if detected.startswith(expected):
                suffix = detected[len(expected):]
                if suffix:
                    added_suffixes[suffix] += 1

            # Pattern 3: missing leading/trailing digits.
            if len(detected) < len(expected):
                if expected.startswith(detected):
                    missing_digits["end"] += 1
                elif expected.endswith(detected):
                    missing_digits["start"] += 1

            # Pattern 4: single-digit confusions.
            if len(detected) == len(expected):
                for exp_digit, det_digit in zip(expected, detected):
                    if exp_digit != det_digit:
                        digit_confusions[(exp_digit, det_digit)] += 1

        patterns["added_prefixes"] = dict(added_prefixes)
        patterns["added_suffixes"] = dict(added_suffixes)
        patterns["missing_digits"] = dict(missing_digits)
        patterns["digit_confusions"] = [
            {"expected": exp, "detected": det, "count": count}
            for (exp, det), count in digit_confusions.most_common(10)
        ]
        return patterns

    def _generate_adjustments(
        self,
        field: str,
        correct_features: Dict[str, float],
        error_features: Dict[str, float],
        success_rate: float,
        patterns: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate concrete parameter adjustments from patterns and features."""
        adjustments: List[Dict[str, Any]] = []

        # Adjustment 1: ROI expansion when success rate is low.
        if success_rate < 40:
            adjustments.append({
                "type": "roi_expansion",
                "reason": f"Very low success rate ({success_rate:.1f}%)",
                "action": "Expand ROI by 10%",
                "priority": "high",
            })
        elif success_rate < 60:
            adjustments.append({
                "type": "roi_expansion",
                "reason": f"Low success rate ({success_rate:.1f}%)",
                "action": "Expand ROI by 5%",
                "priority": "medium",
            })

        # Adjustment 2: prefix correction rules.
        top_prefixes = [
            prefix for prefix, _ in Counter(patterns["added_prefixes"]).most_common(3)
        ]
        if top_prefixes:
            adjustments.append({
                "type": "prefix_rule",
                "reason": f"OCR adds prefixes: {', '.join(top_prefixes)}",
                "action": "Strip common prefixes before validation",
                "prefixes": top_prefixes,
                "priority": "high",
            })

        # Adjustment 3: suffix correction rules.
        top_suffixes = [
            suffix for suffix, _ in Counter(patterns["added_suffixes"]).most_common(3)
        ]
        if top_suffixes:
            adjustments.append({
                "type": "suffix_rule",
                "reason": f"OCR adds suffixes: {', '.join(top_suffixes)}",
                "action": "Strip common suffixes before validation",
                "suffixes": top_suffixes,
                "priority": "high",
            })

        # Adjustment 4: digit confusion rules.
        confusions = [
            {"expected": c["expected"], "detected": c["detected"], "count": c["count"]}
            for c in patterns["digit_confusions"][:5]
        ]
        if confusions:
            adjustments.append({
                "type": "digit_confusion_rules",
                "reason": "Specific digits are frequently confused",
                "action": "Apply digit confusion corrections",
                "confusions": confusions,
                "priority": "medium",
            })

        # Adjustment 5: image characteristic shifts.
        if error_features.get("brightness", 0) < correct_features.get("brightness", 0) * 0.8:
            adjustments.append({
                "type": "preprocessing_change",
                "reason": "Failing images are darker than successful ones",
                "action": "Enable CLAHE preprocessing",
                "preprocessing": {"threshold": "clahe"},
                "priority": "medium",
            })

        if error_features.get("sharpness", 0) < correct_features.get("sharpness", 0) * 0.7:
            adjustments.append({
                "type": "preprocessing_change",
                "reason": "Failing images are blurrier than successful ones",
                "action": "Reduce blur before OCR",
                "preprocessing": {"blur_kernel": 1},
                "priority": "low",
            })

        return adjustments

    # ------------------------------------------------------------------ #
    # Applying adjustments
    # ------------------------------------------------------------------ #

    def _apply_all_adjustments(self) -> int:
        """Apply all pending per-field adjustments to the config."""
        total = 0
        timestamp = datetime.now().isoformat()

        for field, adjustments in self._pending_adjustments.items():
            field_config = self.config["fields"][field]
            for adjustment in adjustments:
                adj_type = adjustment["type"]
                if adj_type == "roi_expansion":
                    increment = 0.10 if adjustment["priority"] == "high" else 0.05
                    roi = field_config["roi"]
                    roi["top"] = max(0.0, roi["top"] - increment)
                    roi["bottom"] = min(1.0, roi["bottom"] + increment)
                    roi["left"] = max(0.0, roi["left"] - increment)
                    roi["right"] = min(1.0, roi["right"] + increment)
                    total += 1
                    logger.info("Field '%s': ROI expanded to %s", field, roi)
                elif adj_type == "prefix_rule":
                    rules = field_config["correction_rules"]
                    for prefix in adjustment["prefixes"]:
                        rule = {"type": "strip_prefix", "pattern": prefix}
                        if rule not in rules:
                            rules.append(rule)
                            total += 1
                elif adj_type == "suffix_rule":
                    rules = field_config["correction_rules"]
                    for suffix in adjustment["suffixes"]:
                        rule = {"type": "strip_suffix", "pattern": suffix}
                        if rule not in rules:
                            rules.append(rule)
                            total += 1
                elif adj_type == "preprocessing_change":
                    pre = field_config["preprocessing"]
                    for key, value in adjustment.get("preprocessing", {}).items():
                        if pre.get(key) != value:
                            pre[key] = value
                            total += 1
                            logger.info(
                                "Field '%s': preprocessing %s -> %s", field, key, value
                            )

        if total > 0:
            self.config["adjustment_history"].append({
                "timestamp": timestamp,
                "adjustments": total,
            })
            self.config["last_updated"] = timestamp
        return total

    def _update_stats(self, references_root: Path) -> None:
        """Update learning statistics from the reference corpus."""
        self.stats["total_analyses"] += 1
        self.stats["last_analysis"] = datetime.now().isoformat()
        self.stats["improvements_applied"] = len(self.config["adjustment_history"])

        total_correct = 0
        total_all = 0
        for field in self.fields:
            correct = _count_images(references_root / field / "correct")
            error = _count_images(references_root / field / "error")
            total_correct += correct
            total_all += correct + error

        if total_all > 0:
            current = total_correct / total_all * 100.0
            if self.stats["initial_success_rate"] == 0.0:
                self.stats["initial_success_rate"] = current
            self.stats["current_success_rate"] = current

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #

    def get_config(self) -> Dict[str, Any]:
        """Return the current learned configuration."""
        return self.config

    def get_field_config(self, field: str) -> Dict[str, Any]:
        """Return the learned configuration for a single field."""
        return self.config["fields"].get(field, self._default_field_config())

    def get_error_patterns(self) -> Dict[str, Any]:
        """Return the mined error patterns."""
        return self.error_patterns

    def correction_rules_for(self, field: str) -> List[Dict[str, Any]]:
        """Return the learned correction rules for a field."""
        return self.config["fields"].get(field, {}).get("correction_rules", [])

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary of the learning state."""
        return {
            "total_analyses": self.stats["total_analyses"],
            "last_analysis": self.stats["last_analysis"],
            "improvements_applied": self.stats["improvements_applied"],
            "initial_success_rate": round(self.stats["initial_success_rate"], 1),
            "current_success_rate": round(self.stats["current_success_rate"], 1),
            "fields": {
                field: {
                    "correction_rules": len(self.config["fields"][field]["correction_rules"]),
                    "patterns": len(self.error_patterns["fields"][field].get("digit_confusions", [])),
                }
                for field in self.fields
            },
        }


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #


def _count_images(path: Path) -> int:
    """Count image files in a directory."""
    if not path.exists():
        return 0
    return sum(1 for f in path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg"))


def _analyze_image_features(path: Path, max_images: int = 50) -> Dict[str, float]:
    """Compute average image characteristics for a set of images.

    Features: brightness, contrast, sharpness and text density. Uses only
    Pillow + numpy so the learning loop runs on any machine.
    """
    brightness: List[float] = []
    contrast: List[float] = []
    sharpness: List[float] = []
    text_density: List[float] = []

    for image_path in sorted(path.iterdir()):
        if len(brightness) >= max_images:
            break
        if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        try:
            with Image.open(image_path) as img:
                gray = np.asarray(img.convert("L"), dtype=np.float64)
        except (OSError, ValueError):
            continue

        brightness.append(float(np.mean(gray)))
        contrast.append(float(np.std(gray)))

        # Sharpness proxy: variance of the Laplacian (edge energy).
        edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float64)
        sharpness.append(float(np.var(edges)))

        # Text density: fraction of dark pixels.
        binary = gray < 127
        text_density.append(float(np.mean(binary)))

    return {
        "brightness": float(np.mean(brightness)) if brightness else 0.0,
        "contrast": float(np.mean(contrast)) if contrast else 0.0,
        "sharpness": float(np.mean(sharpness)) if sharpness else 0.0,
        "text_density": float(np.mean(text_density)) if text_density else 0.0,
    }


def levenshtein_distance(str1: str, str2: str) -> int:
    """Return the Levenshtein edit distance between two strings."""
    if len(str1) < len(str2):
        return levenshtein_distance(str2, str1)
    if not str2:
        return len(str1)

    previous_row = list(range(len(str2) + 1))
    for i, char1 in enumerate(str1):
        current_row = [i + 1]
        for j, char2 in enumerate(str2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (char1 != char2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def find_similar_references(
    target_number: str,
    reference_index: Dict[str, List[str]],
    max_similar: int = 5,
    max_distance: int = 2,
) -> List[Dict[str, Any]]:
    """Find reference numbers similar to a target by Levenshtein distance.

    Args:
        target_number: Number to look up.
        reference_index: Mapping of reference number -> list of image paths.
        max_similar: Maximum results to return.
        max_distance: Maximum allowed edit distance.

    Returns:
        Sorted list of dicts with ``number``, ``distance``, ``similarity``,
        ``path`` and ``priority`` keys.
    """
    results: List[Dict[str, Any]] = []
    target = str(target_number)

    for reference, paths in reference_index.items():
        if not reference:
            continue
        distance = levenshtein_distance(target, str(reference))
        if distance <= max_distance:
            similarity = difflib.SequenceMatcher(None, target, str(reference)).ratio()
            # Prefer manually curated references over automatic ones.
            manual_paths = [p for p in paths if "manual" in p.lower()]
            best_path = manual_paths[0] if manual_paths else paths[0]
            results.append({
                "number": reference,
                "distance": distance,
                "similarity": similarity,
                "path": best_path,
                "priority": "manual" if manual_paths else "automatic",
            })

    results.sort(key=lambda item: (-item["similarity"], item["priority"] != "manual"))
    return results[:max_similar]


def apply_similar_reference_knowledge(
    target_number: str, similar: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Derive correction strategies from similar reference numbers.

    Used when no exact reference exists for a number: the error patterns of
    numerically-similar references are transferred to the current extraction.
    """
    strategies: Dict[str, Any] = {
        "prefix_corrections": [],
        "suffix_corrections": [],
        "digit_confusion_corrections": [],
        "patterns_identified": {},
        "confidence": 0.0,
    }
    if not similar:
        return strategies

    target = str(target_number)
    for item in similar:
        similar_number = str(item["number"])

        if similar_number.endswith(target):
            prefix = similar_number[: -len(target)]
            if prefix:
                strategies["prefix_corrections"].append(prefix)

        if similar_number.startswith(target):
            suffix = similar_number[len(target):]
            if suffix:
                strategies["suffix_corrections"].append(suffix)

        if len(similar_number) == len(target):
            for i, (expected, detected) in enumerate(zip(target, similar_number)):
                if expected != detected:
                    strategies["digit_confusion_corrections"].append({
                        "position": i,
                        "expected": expected,
                        "detected": detected,
                    })

    strategies["confidence"] = min(1.0, len(similar) / 3.0)
    strategies["patterns_identified"] = {
        "prefixes": sorted(set(strategies["prefix_corrections"])),
        "suffixes": sorted(set(strategies["suffix_corrections"])),
        "confusions": _group_confusions(strategies["digit_confusion_corrections"]),
    }
    return strategies


def _group_confusions(confusions: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Group per-position digit confusions by (expected -> detected) pair."""
    counts: Counter = Counter()
    for conf in confusions:
        counts[(conf["expected"], conf["detected"])] += 1
    return [
        {"expected": expected, "detected": detected, "frequency": count}
        for (expected, detected), count in counts.items()
    ]
