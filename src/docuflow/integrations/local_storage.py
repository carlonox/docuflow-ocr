"""Offline document storage.

When a pipeline must run without any spreadsheet service, documents and their
extracted fields are stored locally. The layout mirrors the original
production system's folder-based queue but with no external dependencies:

- ``queue/`` — documents waiting to be processed (named ``<id>.<ext>``).
- ``processed/`` — documents moved after successful extraction.
- ``failed/`` — documents that failed validation (with the error recorded).
- ``results.jsonl`` — one JSON object per processed document.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from docuflow.ocr.base_extractor import ExtractionResult

logger = logging.getLogger(__name__)


class LocalStorage:
    """Local folder-based queue and result store.

    Args:
        root: Base directory for queue/processed/failed and results.jsonl.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.queue_dir = self.root / "queue"
        self.processed_dir = self.root / "processed"
        self.failed_dir = self.root / "failed"
        self.results_path = self.root / "results.jsonl"
        for directory in (self.queue_dir, self.processed_dir, self.failed_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Queue management
    # ------------------------------------------------------------------ #

    def list_queue(self) -> List[Path]:
        """List documents waiting in the queue."""
        return sorted(
            p for p in self.queue_dir.iterdir() if p.is_file() and not p.name.startswith(".")
        )

    def add_to_queue(self, source: Path, document_id: Optional[str] = None) -> Path:
        """Copy a document into the queue."""
        target = self.queue_dir / (document_id or source.name)
        shutil.copy2(source, target)
        logger.info("Queued %s", target)
        return target

    def move_to_processed(self, document_id: str, source_path: Optional[Path] = None) -> None:
        """Move a document to the processed folder."""
        self._move(document_id, source_path, self.processed_dir)

    def move_to_failed(self, document_id: str, source_path: Optional[Path] = None) -> None:
        """Move a document to the failed folder."""
        self._move(document_id, source_path, self.failed_dir)

    def _move(self, document_id: str, source_path: Optional[Path], target_dir: Path) -> None:
        source = source_path or self.queue_dir / document_id
        if source.exists():
            shutil.move(str(source), str(target_dir / source.name))
            logger.info("Moved %s to %s", source.name, target_dir.name)

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #

    def append_result(self, result: ExtractionResult) -> None:
        """Append one extraction result to the JSONL results file."""
        record = {
            "document_id": result.document_id,
            "success": result.success,
            "confidence": result.confidence,
            "fields": result.fields,
            "validation": {
                key: value.to_dict() if hasattr(value, "to_dict") else value
                for key, value in result.validation.items()
            },
            "provider_chain": result.provider_chain,
            "metadata": result.metadata,
        }
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Result appended for %s", result.document_id)

    def load_results(self) -> List[Dict[str, Any]]:
        """Load all recorded results."""
        if not self.results_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.results_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        return records
