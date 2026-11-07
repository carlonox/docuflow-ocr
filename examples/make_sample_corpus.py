#!/usr/bin/env python3
"""
make_sample_corpus.py — genera un corpus sintético para probar el lote y el dashboard.

Crea PDFs con datos inventados (facturas, contratos, actas, recibos) en dos
variantes: con capa de texto y solo imagen (como un escaneo). Sirve para medir
tiempos y ver el dashboard sin usar documentos reales de nadie.

Uso:
    python examples/make_sample_corpus.py --out samples --n 40
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import fitz  # PyMuPDF

TIPOS = [
    ("FACTURA", "Factura de venta", ["N° {n}", "Fecha: 2026-{m:02d}-{d:02d}", "Subtotal: ${v:,}", "IVA (19%): ${iva:,}", "Total: ${tot:,}"]),
    ("CONTRATO", "Contrato de prestación de servicios", ["Parte A: Empresa Ejemplo S.A.S.", "Parte B: Proveedor Demo Ltda.", "Duración: {m} meses", "Valor: ${v:,}"]),
    ("ACTA", "Acta de reunión", ["Asistentes: 4", "Punto 1: revisión de avance", "Punto 2: pendientes", "Próxima sesión: 2026-{m:02d}-{d:02d}"]),
    ("RECIBO", "Recibo de pago", ["Recibido de: Cliente Demo", "Concepto: servicio mensual", "Monto: ${v:,}", "Medio: transferencia"]),
]


def make_text_pdf(path: Path, titulo: str, cuerpo: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 90
    page.insert_text((72, y), titulo, fontsize=16, fontname="helv")
    for line in cuerpo:
        y += 26
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


def make_scanned_pdf(path: Path, titulo: str, cuerpo: list[str]) -> None:
    """Sin capa de texto: primero una página con texto, luego se rasteriza."""
    tmp = fitz.open()
    page = tmp.new_page()
    y = 90
    page.insert_text((72, y), titulo, fontsize=16, fontname="helv")
    for line in cuerpo:
        y += 26
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
    pix = page.get_pixmap(dpi=150)
    tmp.close()
    # degradado: ruido + giro leve para simular escaneos malos
    from PIL import Image, ImageEnhance
    import io, random
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    severe = random.random() < 0.18
    if severe or random.random() < 0.35:
        ang = random.uniform(-5.0, 5.0) if severe else random.uniform(-1.6, 1.6)
        img = img.rotate(ang, expand=True, fillcolor="white")
        img = ImageEnhance.Contrast(img).enhance(0.28 if severe else 0.55)
        if severe:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(1.4))
        import numpy as _np
        arr = _np.array(img).astype("int16")
        arr += _np.random.normal(0, 28, arr.shape).astype("int16")
        img = Image.fromarray(_np.clip(arr, 0, 255).astype("uint8"))
    buf = io.BytesIO(); img.save(buf, "PNG"); png_bytes = buf.getvalue()
    out = fitz.open()
    p2 = out.new_page(width=612, height=792)
    p2.insert_image(p2.rect, stream=png_bytes)
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
    hechos = 0
    for i in range(1, args.n + 1):
        tipo, titulo, campos = random.choice(TIPOS)
        v = random.randint(50_000, 9_000_000)
        cuerpo = [c.format(n=f"{i:04d}", m=random.randint(1, 12), d=random.randint(1, 28),
                           v=v, iva=int(v * 0.19), tot=int(v * 1.19)) for c in campos]
        escaneado = i % 2 == 0
        nombre = f"{tipo.lower()}-{i:03d}{'-scan' if escaneado else ''}.pdf"
        destino = out / nombre
        if escaneado:
            make_scanned_pdf(destino, f"{titulo} N° {i:04d}", cuerpo)
        else:
            make_text_pdf(destino, f"{titulo} N° {i:04d}", cuerpo)
        hechos += 1
    print(f"{hechos} PDFs en {out} (mitad con capa de texto, mitad solo imagen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
