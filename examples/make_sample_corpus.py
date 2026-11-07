#!/usr/bin/env python3
"""
make_sample_corpus.py - synthetic document corpus for demos and benchmarks.

Creates PDFs with invented data (invoices, contracts, minutes, receipts) in two
variants: with a text layer, and image-only (like a scan). About one in six
scanned pages gets heavy degradation (rotation, blur, noise) so the review queue
in the dashboard is not empty.

Nothing here comes from real documents.

Usage:
    python examples/make_sample_corpus.py --out samples --n 40
"""
from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

TIPOS = [
    ("INVOICE", "Sales invoice", ["No. {n}", "Date: 2026-{m:02d}-{d:02d}", "Subtotal: ${v:,}", "VAT (19%): ${iva:,}", "Total: ${tot:,}"]),
    ("CONTRACT", "Service agreement", ["Party A: Example Corp.", "Party B: Demo Vendor Ltd.", "Term: {m} months", "Value: ${v:,}"]),
    ("MINUTES", "Meeting minutes", ["Attendees: 4", "Item 1: progress review", "Item 2: open actions", "Next session: 2026-{m:02d}-{d:02d}"]),
    ("RECEIPT", "Payment receipt", ["Received from: Demo Client", "Concept: monthly service", "Amount: ${v:,}", "Method: transfer"]),
]


def _render_page(titulo: str, lines: list[str]) -> fitz.Pixmap:
    doc = fitz.open()
    page = doc.new_page()
    y = 90
    page.insert_text((72, y), titulo, fontsize=16, fontname="helv")
    for line in lines:
        y += 26
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
    pix = page.get_pixmap(dpi=150)
    doc.close()
    return pix


def make_text_pdf(path: Path, titulo: str, lines: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 90
    page.insert_text((72, y), titulo, fontsize=16, fontname="helv")
    for line in lines:
        y += 26
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


def make_scanned_pdf(path: Path, titulo: str, lines: list[str]) -> None:
    """Image-only PDF: render text, then degrade and embed as a picture."""
    pix = _render_page(titulo, lines)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    severe = random.random() < 0.16
    if severe or random.random() < 0.35:
        angle = random.uniform(-5.0, 5.0) if severe else random.uniform(-1.6, 1.6)
        img = img.rotate(angle, expand=True, fillcolor="white")
        img = ImageEnhance.Contrast(img).enhance(0.28 if severe else 0.55)
        if severe:
            img = img.filter(ImageFilter.GaussianBlur(1.4))
        arr = np.array(img).astype("int16")
        arr += np.random.normal(0, 28, arr.shape).astype("int16")
        img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))

    buf = io.BytesIO()
    img.save(buf, "PNG")
    out = fitz.open()
    page = out.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=buf.getvalue())
    out.save(path)
    out.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    made = 0
    for i in range(1, args.n + 1):
        kind, title, fields = random.choice(TIPOS)
        value = random.randint(50_000, 9_000_000)
        lines = [f.format(n=f"{i:04d}", m=random.randint(1, 12), d=random.randint(1, 28),
                          v=value, iva=int(value * 0.19), tot=int(value * 1.19)) for f in fields]
        scanned = i % 2 == 0
        name = f"{kind.lower()}-{i:03d}{'-scan' if scanned else ''}.pdf"
        target = out / name
        if scanned:
            make_scanned_pdf(target, f"{title} No. {i:04d}", lines)
        else:
            make_text_pdf(target, f"{title} No. {i:04d}", lines)
        made += 1
    print(f"{made} PDFs in {out} (half with a text layer, half image-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
