"""Pipeline orchestrator: batch document processing with checkpoints.

The pipeline iterates over a queue of documents, runs an extractor per
document, records the outcome in a monitor, and persists checkpoints so
interrupted batches resume instead of restarting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from docuflow.core.checkpoint import CheckpointStore
from docuflow.ocr.base_extractor import AbstractDocumentExtractor, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A processing stage applied to each document.

    Attributes:
        name: Step identifier.
        fn: Callable ``(document_id, result, context) -> context``. The
            function may mutate the result or raise to signal failure.
        on_error: When ``"skip"`` the document is skipped on failure; when
            ``"stop"`` the whole pipeline stops.
    """

    name: str
    fn: Callable[[str, ExtractionResult, Dict[str, Any]], Dict[str, Any]]
    on_error: str = "skip"


class Pipeline:
    """Run a document batch through extractors and post-processing steps.

    Args:
        extractor: Document extractor used for every item.
        steps: Optional post-extraction steps (validation, enrichment, sink).
        checkpoint: Optional store for resume support.
        monitor: Optional precision monitor that records each outcome.
        batch_size: Number of items between checkpoint saves.
    """

    def __init__(
        self,
        extractor: AbstractDocumentExtractor,
        steps: Optional[List[PipelineStep]] = None,
        checkpoint: Optional[CheckpointStore] = None,
        monitor: Optional[Any] = None,
        batch_size: int = 5,
    ) -> None:
        self.extractor = extractor
        self.steps = steps or []
        self.checkpoint = checkpoint
        self.monitor = monitor
        self.batch_size = max(1, batch_size)
        self.context: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Batch execution
    # ------------------------------------------------------------------ #

    def run(self, items: Iterable[Any], document_id_fn: Callable[[Any], str]) -> Dict[str, Any]:
        """Process a batch of documents.

        Args:
            items: Iterable of document entries (paths, dicts, row objects).
            document_id_fn: Extracts a stable string ID from each item.

        Returns:
            Summary dict: total, succeeded, failed, skipped, results.
        """
        processed: Set[str] = set()
        if self.checkpoint is not None:
            saved = self.checkpoint.load()
            processed = set(saved.get("processed_items", []))

        summary = {"total": 0, "succeeded": 0, "failed": 0, "skipped": 0, "results": []}

        for item in items:
            document_id = document_id_fn(item)
            summary["total"] += 1

            if document_id in processed:
                summary["skipped"] += 1
                continue

            logger.info("Processing document %s", document_id)
            try:
                result = self.extractor.process(str(item), document_id=document_id)
                for step in self.steps:
                    self.context = step.fn(document_id, result, self.context)
                summary["results"].append(result)
                summary["succeeded" if result.success else "failed"] += 1

                if self.monitor is not None:
                    self.monitor.record_result(
                        [
                            result.validation.get(f, {}).get("valid", False)
                            for f in getattr(self.extractor, "fields", [])
                        ]
                    )
            except Exception as exc:  # pragma: no cover - failure path
                logger.error("Document %s failed: %s", document_id, exc)
                summary["failed"] += 1
                for step in self.steps:
                    if step.on_error == "stop":
                        raise

            processed.add(document_id)
            if self.checkpoint is not None and summary["total"] % self.batch_size == 0:
                self.checkpoint.save(document_id, processed)

        if self.checkpoint is not None:
            self.checkpoint.save(None, processed)
        logger.info(
            "Batch finished: %d total, %d ok, %d failed, %d skipped",
            summary["total"], summary["succeeded"], summary["failed"], summary["skipped"],
        )
        return summary
