"""Unit tests for image preprocessing."""

from __future__ import annotations

from PIL import Image

from docuflow.ocr.preprocessing import ImagePreprocessor


class TestImagePreprocessor:
    def test_process_returns_variants(self, tmp_path):
        # Create a small synthetic image.
        image = Image.new("L", (600, 400), 200)
        path = tmp_path / "doc.png"
        image.save(path)

        preprocessor = ImagePreprocessor(denoise=True, binarize=False, deskew=False)
        variants = preprocessor.process(str(path))
        assert len(variants) >= 1
        assert all(v.mode == "L" for v in variants)

    def test_clean_upscales_low_resolution(self, tmp_path):
        image = Image.new("L", (600, 400), 200)
        path = tmp_path / "small.png"
        image.save(path)

        preprocessor = ImagePreprocessor(binarize=False, deskew=False)
        variants = preprocessor.process(str(path))
        assert variants[0].width >= 1200  # upscaled 2x

    def test_invalid_image_returns_empty(self, tmp_path):
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")
        preprocessor = ImagePreprocessor()
        assert preprocessor.process(str(path)) == []
