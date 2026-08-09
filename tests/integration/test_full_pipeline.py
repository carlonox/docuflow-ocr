"""Integration test: full pipeline with synthetic fixtures and stub providers.

Runs the whole extract -> validate -> learn -> monitor flow without any real
OCR engine: a stub provider returns text, the extractor parses it, the
learning hook records the outcome, and the precision monitor tracks it.
"""

from __future__ import annotations

from docuflow.core.checkpoint import CheckpointStore
from docuflow.core.pipeline import Pipeline, PipelineStep
from docuflow.examples.id_card_extractor import IDCardExtractor
from docuflow.learning.precision_monitor import PrecisionMonitor
from docuflow.ocr.cascade import CascadeConfig, CascadeOCR
from docuflow.ocr.providers.base import AbstractOCRProvider, OCRResult


class _SyntheticIDProvider(AbstractOCRProvider):
    """Simulates a perfect OCR engine for the fictional ID card format."""

    name = "synthetic"

    def extract(self, image_path: str) -> OCRResult:
        return OCRResult(
            text=(
                "REPUBLIC OF TESTLAND\n"
                "NATIONAL IDENTIFICATION CARD\n"
                "ID Number: ID-12345678\n"
                "Full Name: TEST USER 42\n"
            ),
            confidence=0.98,
            provider=self.name,
        )


class _BadIDProvider(AbstractOCRProvider):
    """Simulates an OCR engine that misreads the ID (broken format)."""

    name = "synthetic_bad"

    def extract(self, image_path: str) -> OCRResult:
        return OCRResult(
            # The OCR read "ID-1234567A" — a digit confused with a letter,
            # which fails the ID-######## format validation.
            text="ID Number: ID-1234567A\nFull Name: TEST USER 42\n",
            confidence=0.70,
            provider=self.name,
        )


def _make_pipeline(provider, tmp_path):
    cascade = CascadeOCR(
        fast_provider=provider,
        accurate_provider=provider,
        config=CascadeConfig(fast_threshold=0.95),
    )
    extractor = IDCardExtractor(cascade)
    checkpoint = CheckpointStore(str(tmp_path / "checkpoint.json"))
    monitor = PrecisionMonitor(
        stats_path=str(tmp_path / "precision.json"),
        fields=["id_number", "full_name"],
        min_samples_for_alerts=10,
    )
    pipeline = Pipeline(extractor, checkpoint=checkpoint, monitor=monitor)
    return pipeline


class TestFullPipeline:
    def test_successful_batch(self, tmp_path):
        pipeline = _make_pipeline(_SyntheticIDProvider(), tmp_path)
        items = [str(tmp_path / f"doc_{i}.png") for i in range(3)]
        summary = pipeline.run(items, document_id_fn=lambda item: item.split("/")[-1])

        assert summary["total"] == 3
        assert summary["succeeded"] == 3
        assert summary["failed"] == 0

        # Monitor recorded all results.
        assert pipeline.monitor.global_stats["total_processed"] == 3

    def test_failed_batch_recorded(self, tmp_path):
        pipeline = _make_pipeline(_BadIDProvider(), tmp_path)
        items = [str(tmp_path / "doc.png")]
        summary = pipeline.run(items, document_id_fn=lambda item: "doc.png")

        assert summary["total"] == 1
        assert summary["failed"] == 1

    def test_checkpoint_resume_skips_processed(self, tmp_path):
        pipeline = _make_pipeline(_SyntheticIDProvider(), tmp_path)
        items = [str(tmp_path / f"doc_{i}.png") for i in range(2)]
        first = pipeline.run(items, document_id_fn=lambda item: item.split("/")[-1])
        assert first["succeeded"] == 2

        # Second run: same items, checkpoint present -> all skipped.
        second = pipeline.run(items, document_id_fn=lambda item: item.split("/")[-1])
        assert second["skipped"] == 2
        assert second["succeeded"] == 0

    def test_pipeline_steps_run(self, tmp_path):
        pipeline = _make_pipeline(_SyntheticIDProvider(), tmp_path)
        touched = []

        def step_fn(document_id, result, context):
            touched.append(document_id)
            return context

        pipeline.steps.append(PipelineStep(name="record", fn=step_fn))
        pipeline.run([str(tmp_path / "doc.png")], document_id_fn=lambda item: "doc.png")
        assert touched == ["doc.png"]
