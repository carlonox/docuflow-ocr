"""Multi-PSM Tesseract extraction.

Tesseract's page segmentation modes (PSM) trade speed against structure
awareness. A single PSM often fails on documents with unusual layouts; this
module runs several PSMs over a region of interest and returns the most
confident result. Pattern borrowed from production OCR pipelines that face
stamps, tables and multi-column scans.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:  # pragma: no cover - environment dependent
    pytesseract = None  # type: ignore
    HAS_TESSERACT = False


#: PSMs tried in order. 6 = uniform block, 4 = single column, 3 = fully auto.
PSM_MODES: Tuple[int, ...] = (6, 4, 3, 11)


def extract_text_multi_psm(
    image: Image.Image,
    tesseract_cmd: Optional[str] = None,
    psm_modes: Tuple[int, ...] = PSM_MODES,
    lang: str = "eng",
    min_text_length: int = 4,
) -> Tuple[str, float]:
    """Run Tesseract with several PSMs and return the best result.

    Args:
        image: Grayscale or RGB PIL image of the region to read.
        tesseract_cmd: Optional path to the ``tesseract`` binary.
        psm_modes: PSM modes to try, in order.
        lang: Tesseract language pack.
        min_text_length: Results shorter than this are discarded.

    Returns:
        Tuple of (best text, best confidence). Confidence is 0.0 when
        Tesseract is unavailable or nothing usable was read.
    """
    if not HAS_TESSERACT:
        logger.warning("pytesseract not installed; multi-PSM extraction unavailable")
        return "", 0.0

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    best_text = ""
    best_confidence = 0.0
    for psm in psm_modes:
        try:
            data = pytesseract.image_to_data(
                image, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
            )
            text = " ".join(
                word for word, conf in zip(data["text"], data["conf"])
                if word.strip() and conf != "-1"
            )
            confidences = [
                float(conf) / 100.0
                for conf, word in zip(data["conf"], data["text"])
                if word.strip() and conf != "-1"
            ]
            confidence = float(np.mean(confidences)) if confidences else 0.0

            if len(text.strip()) >= min_text_length and confidence > best_confidence:
                best_text = text.strip()
                best_confidence = confidence
        except Exception as exc:  # pragma: no cover - tesseract binary issues
            logger.warning("PSM %d failed: %s", psm, exc)

    return best_text, best_confidence
