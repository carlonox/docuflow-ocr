"""Resumable pipeline checkpoints.

Long document-processing runs are interrupted all the time: network drops,
driver crashes, manual review pauses. The checkpoint store persists the
pipeline position (current item + already-processed set) so a restart
continues exactly where it stopped instead of redoing work.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class CheckpointStore:
    """Persist and restore pipeline progress.

    Args:
        path: JSON file used for persistence.
    """

    def __init__(self, path: str = "checkpoint.json") -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        """Load the saved checkpoint.

        Returns:
            A dict with ``current_item``, ``processed_items`` and any
            extra metadata saved by the pipeline.
        """
        if not self.path.exists():
            return {"current_item": None, "processed_items": []}

        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            data.setdefault("current_item", None)
            data.setdefault("processed_items", [])
            logger.info(
                "Checkpoint loaded: %d processed items",
                len(data["processed_items"]),
            )
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load checkpoint %s: %s", self.path, exc)
            return {"current_item": None, "processed_items": []}

    def save(
        self,
        current_item: Any,
        processed_items: Set[Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the current pipeline position.

        Args:
            current_item: Identifier of the item being processed.
            processed_items: Set of already-completed item identifiers.
            extra: Optional additional metadata (timings, counters, ...).
        """
        data: Dict[str, Any] = {
            "current_item": current_item,
            "processed_items": sorted(processed_items, key=str),
        }
        if extra:
            data.update(extra)

        try:
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            logger.info("Checkpoint saved: %d processed items", len(processed_items))
        except OSError as exc:
            logger.error("Could not save checkpoint %s: %s", self.path, exc)

    def clear(self) -> None:
        """Remove the checkpoint file (fresh start)."""
        if self.path.exists():
            self.path.unlink()
            logger.info("Checkpoint cleared")
