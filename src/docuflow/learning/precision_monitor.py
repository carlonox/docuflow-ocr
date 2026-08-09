"""Precision monitor: sliding-window accuracy tracking with drift alerts.

Tracks every extraction result (per-field match flags) and watches for
accuracy degradation in near-real time. When the recent window of results
drops significantly below the global average — or shows a streak of total
failures — the monitor raises alerts so the pipeline can pause, retrain or
flag documents for human review before a bad batch propagates.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

RESULT_FULL = "FULL"
RESULT_PARTIAL = "PARTIAL"
RESULT_NONE = "NONE"
RESULT_MANUAL = "MANUAL"


class PrecisionMonitor:
    """Monitor extraction precision over a sliding window of results.

    Args:
        window_size: Number of recent results considered for drift detection.
        stats_path: JSON file for persistence across runs.
        fields: Field names tracked by the pipeline.
        min_samples_for_alerts: Minimum results before alerts can fire.
    """

    def __init__(
        self,
        window_size: int = 20,
        stats_path: str = "precision_stats.json",
        fields: Sequence[str] = ("field_a", "field_b"),
        min_samples_for_alerts: int = 10,
    ) -> None:
        self.window_size = window_size
        self.stats_path = Path(stats_path)
        self.fields = list(fields)
        self.min_samples_for_alerts = min_samples_for_alerts

        self.history: Deque[str] = deque(maxlen=window_size)
        self.global_stats: Dict[str, Any] = {
            "total_processed": 0,
            "full": 0,
            "partial": 0,
            "none": 0,
            "manual": 0,
            "started_at": time.time(),
            "field_matches": {field: {"ok": 0, "fail": 0} for field in self.fields},
        }
        self._load_stats()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load_stats(self) -> None:
        if not self.stats_path.exists():
            return
        try:
            with self.stats_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key, value in data.get("global", {}).items():
                self.global_stats[key] = value
            self.history = deque(
                data.get("recent_history", [])[-self.window_size:],
                maxlen=self.window_size,
            )
            logger.info(
                "Loaded %d previous results from %s",
                self.global_stats["total_processed"], self.stats_path,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load stats from %s: %s", self.stats_path, exc)

    def _save_stats(self) -> None:
        data = {
            "global": self.global_stats,
            "recent_history": list(self.history),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            with self.stats_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as exc:
            logger.warning("Could not save stats: %s", exc)

    # ------------------------------------------------------------------ #
    # Recording results
    # ------------------------------------------------------------------ #

    def record_result(
        self,
        field_matches: Sequence[bool],
        manual: bool = False,
        field_names: Optional[Sequence[str]] = None,
    ) -> str:
        """Record one extraction outcome.

        Args:
            field_matches: One boolean per tracked field (True if the field
                was extracted and validated correctly).
            manual: True when the result came from a human correction.
            field_names: Optional override for the field names (useful when
                the pipeline extracts a subset of fields).

        Returns:
            The result category: FULL, PARTIAL, NONE or MANUAL.
        """
        names = list(field_names) if field_names else self.fields
        matches = list(field_matches)

        if manual:
            result = RESULT_MANUAL
            self.global_stats["manual"] += 1
        elif all(matches):
            result = RESULT_FULL
            self.global_stats["full"] += 1
        elif any(matches):
            result = RESULT_PARTIAL
            self.global_stats["partial"] += 1
        else:
            result = RESULT_NONE
            self.global_stats["none"] += 1

        for name, matched in zip(names, matches):
            if name in self.global_stats["field_matches"]:
                bucket = self.global_stats["field_matches"][name]
                bucket["ok" if matched else "fail"] += 1

        self.global_stats["total_processed"] += 1
        self.history.append(result)
        self._save_stats()
        self.check_alerts()
        return result

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def window_precision(self) -> Dict[str, float]:
        """Precision percentages over the current window (excludes MANUAL)."""
        processed = [r for r in self.history if r != RESULT_MANUAL]
        if not processed:
            return {"full": 0.0, "partial": 0.0, "none": 0.0}

        total = len(processed)
        return {
            "full": processed.count(RESULT_FULL) / total * 100.0,
            "partial": processed.count(RESULT_PARTIAL) / total * 100.0,
            "none": processed.count(RESULT_NONE) / total * 100.0,
        }

    def global_precision(self) -> Dict[str, float]:
        """Precision percentages over all recorded results."""
        total = (
            self.global_stats["full"]
            + self.global_stats["partial"]
            + self.global_stats["none"]
        )
        if total == 0:
            return {"full": 0.0, "partial": 0.0, "none": 0.0}
        return {
            "full": self.global_stats["full"] / total * 100.0,
            "partial": self.global_stats["partial"] / total * 100.0,
            "none": self.global_stats["none"] / total * 100.0,
        }

    def per_field_precision(self) -> Dict[str, float]:
        """Per-field accuracy percentages."""
        result: Dict[str, float] = {}
        for name, counts in self.global_stats["field_matches"].items():
            total = counts["ok"] + counts["fail"]
            result[name] = (counts["ok"] / total * 100.0) if total else 0.0
        return result

    # ------------------------------------------------------------------ #
    # Alerts
    # ------------------------------------------------------------------ #

    def check_alerts(self) -> List[str]:
        """Check for precision drift and return active alerts."""
        if len(self.history) < self.min_samples_for_alerts:
            return []

        window = self.window_precision()
        global_p = self.global_precision()
        alerts: List[str] = []
        recent = list(self.history)

        # Alert 1: window dropped significantly vs global average.
        if window["full"] < global_p["full"] - 15:
            alerts.append(
                f"Full-match precision dropped {global_p['full'] - window['full']:.1f} "
                "points vs the global average in the recent window"
            )

        # Alert 2: total failures dominate the window.
        if window["none"] > 60:
            alerts.append(
                f"Critical: {window['none']:.1f}% of recent extractions failed completely"
            )

        # Alert 3: streak of consecutive total failures.
        if recent[-5:].count(RESULT_NONE) >= 4:
            alerts.append(
                f"{recent[-5:].count(RESULT_NONE)}/5 most recent extractions failed completely"
            )

        # Alert 4: full precision is very low.
        if window["full"] < 25 and len(self.history) >= 15:
            alerts.append(
                f"Full precision is very low ({window['full']:.1f}%) — "
                "consider retraining or manual review"
            )

        # Alert 5: positive trend (improvement over global).
        if window["full"] > global_p["full"] + 10 and window["full"] > 45:
            alerts.append(
                f"Improvement: window precision is {window['full'] - global_p['full']:.1f} "
                "points above the global average"
            )

        for alert in alerts:
            logger.warning(alert)
        return alerts

    def summary(self) -> Dict[str, Any]:
        """Return a complete summary of monitored precision."""
        global_p = self.global_precision()
        total = self.global_stats["total_processed"]
        return {
            "total_processed": total,
            "full": self.global_stats["full"],
            "partial": self.global_stats["partial"],
            "none": self.global_stats["none"],
            "manual": self.global_stats["manual"],
            "full_precision": round(global_p["full"], 1),
            "partial_precision": round(global_p["partial"], 1),
            "none_precision": round(global_p["none"], 1),
            "per_field": {
                name: round(precision, 1)
                for name, precision in self.per_field_precision().items()
            },
            "active_alerts": self.check_alerts(),
        }
