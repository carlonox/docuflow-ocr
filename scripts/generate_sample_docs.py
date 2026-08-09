"""Generate synthetic sample documents with Pillow.

Creates fictional ID cards, invoices and forms for tests and examples.
Everything produced here is synthetic: no real names, no real ID numbers,
no personal data.

Usage:
    python scripts/generate_sample_docs.py --output tests/fixtures [--count 1]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ID_CARD_TEMPLATE = """REPUBLIC OF TESTLAND
NATIONAL IDENTIFICATION CARD

ID Number: {id_number}
Full Name: {full_name}
Date of Birth: {dob}
Place of Birth: {city}

This document is a synthetic sample for demonstration purposes.
"""

INVOICE_TEMPLATE = """TESTLAND SUPPLIES CO.
INVOICE

Invoice Number: {invoice_number}
Vendor: {vendor}
Customer: {customer}

Item                  Qty    Unit Price    Total
Widget A               2      $25.00        $50.00
Widget B               1      $75.00        $75.00

TOTAL: {amount}
Payment due within 30 days.
"""

FORM_TEMPLATE = """TESTLAND CENSUS FORM

Form Code: {form_code}
Name: {name}
Address: {address}
Signature: {signature}

Please return this form before the deadline.
"""

# ---------------------------------------------------------------------- #
# Fonts
# ---------------------------------------------------------------------- #


def _load_font(size: int):
    """Load a monospace font; fall back to default when no TTF is found."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------- #
# Document renderers
# ---------------------------------------------------------------------- #


def render_id_card(path: Path, seed: int = 1) -> Path:
    """Render a fictional ID card."""
    rng = random.Random(seed)
    text = ID_CARD_TEMPLATE.format(
        id_number=f"ID-{rng.randint(10000000, 99999999)}",
        full_name=f"TEST USER {rng.randint(1000, 9999)}",
        dob=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1960, 2000)}",
        city=rng.choice(["NORTHVILLE", "SOUTHFIELD", "EASTPORT", "WESTGATE"]),
    )
    return _render_text(path, text, seed=seed)


def render_invoice(path: Path, seed: int = 2) -> Path:
    """Render a fictional invoice."""
    rng = random.Random(seed)
    text = INVOICE_TEMPLATE.format(
        invoice_number=f"INV-{rng.randint(2020, 2026)}-{rng.randint(1, 9999):04d}",
        vendor=rng.choice(["ACME SUPPLIES", "GLOBALTECH INC", "NORTHWIND CO"]),
        customer=f"CLIENT {rng.randint(100, 999)}",
        amount=f"{rng.randint(50, 5000)}.{rng.randint(0, 99):02d}",
    )
    return _render_text(path, text, seed=seed)


def render_form(path: Path, seed: int = 3) -> Path:
    """Render a fictional census form."""
    rng = random.Random(seed)
    signature = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(18))
    text = FORM_TEMPLATE.format(
        form_code=f"FORM-{''.join(rng.choice('ABCDEF0123456789') for _ in range(6))}",
        name=f"TEST PERSON {rng.randint(1000, 9999)}",
        address=f"{rng.randint(1, 9999)} {rng.choice(['MAIN', 'OAK', 'PINE'])} ST",
        signature=signature,
    )
    return _render_text(path, text, seed=seed)


def _render_text(path: Path, text: str, seed: int = 1, degrade: bool = True) -> Path:
    """Render text into a grayscale image, optionally degrading it.

    Degradation simulates scanned documents: rotation, noise and slight blur,
    which is exactly what the cascade pipeline is designed to handle.
    """
    rng = random.Random(seed)
    font = _load_font(22)

    # Measure required canvas.
    probe = Image.new("L", (10, 10))
    draw = ImageDraw.Draw(probe)
    lines = text.splitlines()
    line_height = 30
    width = max(draw.textlength(line, font=font) for line in lines) + 80
    height = line_height * len(lines) + 80

    image = Image.new("L", (int(width), int(height)), 255)
    draw = ImageDraw.Draw(image)
    y = 40
    for line in lines:
        draw.text((40, y), line, font=font, fill=0)
        y += line_height

    if degrade:
        # Rotation (mild skew).
        image = image.rotate(rng.uniform(-1.5, 1.5), fillcolor=255)

        # Noise.
        pixels = image.load()
        if pixels is not None:
            for _ in range(int(image.width * image.height * 0.002)):
                x = rng.randint(0, image.width - 1)
                y = rng.randint(0, image.height - 1)
                pixels[x, y] = 255 if rng.random() < 0.5 else 0

        # Stamp-like blob (a circle in a corner, common on real documents).
        stamp = Image.new("L", image.size, 255)
        stamp_draw = ImageDraw.Draw(stamp)
        stamp_draw.ellipse(
            (image.width - 150, 20, image.width - 40, 130),
            outline=80, width=4,
        )
        stamp_draw.text((image.width - 140, 55), "RECEIVED", font=font, fill=90)
        image = Image.blend(image, stamp, 0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return path


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

RENDERERS = {
    "id_card": render_id_card,
    "invoice": render_invoice,
    "form": render_form,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tests/fixtures", help="Output directory")
    parser.add_argument("--count", type=int, default=1, help="Documents per type")
    parser.add_argument("--no-degrade", action="store_true", help="Render clean documents")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for doc_type, renderer in RENDERERS.items():
        for i in range(args.count):
            seed = i + 1 + (0 if args.no_degrade else 1000)
            target = output / f"sample_{doc_type}_{i + 1}.png"
            renderer(target, seed=seed)
            print(f"Generated {target}")

    print(f"\nDone. {args.count * len(RENDERERS)} synthetic documents in {output}")


if __name__ == "__main__":
    main()
