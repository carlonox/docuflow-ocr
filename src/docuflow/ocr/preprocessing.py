"""Image preprocessing for degraded documents.

Documents scanned from paper (stamps, stains, low contrast, skewed pages)
defeat naive OCR. This module generates multiple cleaned variants of an input
image so the cascade can pick the variant that yields the best extraction:

- conservative denoising (median filter),
- Sauvola binarization (outperforms Otsu on stained documents),
- 2x upscaling when the source resolution is low,
- deskewing (rotation correction).

Only Pillow + numpy are required; the optional scikit-image dependency enables
the full Sauvola and skew-estimation paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

try:  # Optional dependency for advanced paths.
    from skimage import color, filters, transform  # type: ignore

    HAS_SKIMAGE = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_SKIMAGE = False


class ImagePreprocessor:
    """Generate OCR-ready variants of a document image.

    Args:
        min_dpi: Below this DPI the image is upscaled 2x.
        denoise: Whether to apply the conservative median denoise variant.
        binarize: Whether to include the Sauvola binarized variant.
        deskew: Whether to include the deskewed variant.
        max_variants: Cap on the number of variants generated.
    """

    def __init__(
        self,
        min_dpi: int = 300,
        denoise: bool = True,
        binarize: bool = True,
        deskew: bool = True,
        max_variants: int = 4,
    ) -> None:
        self.min_dpi = min_dpi
        self.denoise = denoise
        self.binarize = binarize
        self.deskew = deskew
        self.max_variants = max_variants

    def process(self, image_path: str) -> List[Image.Image]:
        """Generate preprocessing variants of an image.

        Args:
            image_path: Path to the source image.

        Returns:
            A list of PIL images: always the clean base variant first, then
            the configured specializations. An empty list signals the image
            could not be opened.
        """
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except (OSError, ValueError) as exc:
            logger.warning("Could not open image %s: %s", image_path, exc)
            return []

        variants: List[Image.Image] = [self._clean(image.copy())]

        if self.denoise and len(variants) < self.max_variants:
            variants.append(self._denoise(image.copy()))

        if self.binarize and HAS_SKIMAGE and len(variants) < self.max_variants:
            binarized = self._binarize_sauvola(image)
            if binarized is not None:
                variants.append(binarized)

        if self.deskew and HAS_SKIMAGE and len(variants) < self.max_variants:
            deskewed = self._deskew(image)
            if deskewed is not None:
                variants.append(deskewed)

        return variants[: self.max_variants]

    # ------------------------------------------------------------------ #
    # Variants
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean(image: Image.Image) -> Image.Image:
        """Upscale low-resolution scans and normalize contrast."""
        if image.width < 1200 and image.height < 1600:  # rough <300 DPI proxy
            image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        return image.convert("L")

    @staticmethod
    def _denoise(image: Image.Image) -> Image.Image:
        """Conservative median denoising (keeps text edges)."""
        return image.convert("L").filter(ImageFilter.MedianFilter(size=3))

    @staticmethod
    def _binarize_sauvola(image: Image.Image) -> Optional[Image.Image]:
        """Sauvola binarization, robust against stains and stamps."""
        try:
            gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
            binary = filters.threshold_sauvola(gray, window_size=25, k=0.2)
            output = (gray > binary) * 255
            return Image.fromarray(output.astype(np.uint8))
        except Exception as exc:  # pragma: no cover - skimage version variance
            logger.warning("Sauvola binarization failed: %s", exc)
            return None

    @staticmethod
    def _deskew(image: Image.Image) -> Optional[Image.Image]:
        """Estimate and correct page skew."""
        try:
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
            angle = _estimate_skew(gray)
            if angle is None or abs(angle) < 0.5:
                return None
            rotated = transform.rotate(gray, angle, resize=False, preserve_range=True)
            return Image.fromarray(rotated.astype(np.uint8))
        except Exception as exc:  # pragma: no cover - skimage version variance
            logger.warning("Deskew failed: %s", exc)
            return None


def _estimate_skew(gray: np.ndarray) -> Optional[float]:
    """Estimate skew angle by maximizing variance of horizontal projections."""
    if not HAS_SKIMAGE:
        return None

    best_angle: Optional[float] = None
    best_score = -1.0
    for angle in range(-10, 11):
        rotated = transform.rotate(gray, angle, resize=False, preserve_range=True)
        projection = rotated.mean(axis=1)
        score = float(np.var(projection))
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle if best_angle is not None else None
