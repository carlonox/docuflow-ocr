#!/usr/bin/env python3
"""
batch_corpus.py — OCR por lotes de un corpus local con docuflow (cascade) o RapidOCR.

Estrategia (la que ahorra tiempo):
  1. Si la pagina del PDF YA tiene capa de texto  -> se usa, coste ~0.
  2. Si no la tiene                                -> render a imagen y OCR.
  3. El OCR va por el cascade de docuflow si esta en el path; si no, RapidOCR directo.

Reanudable: registra cada PDF terminado en done.txt y salta los ya hechos.
Salida: un JSONL con {path, pages:[{n, text, provider, conf}], text_chars}.

Uso:
  python examples/batch_corpus.py --root samples --out demo.jsonl
  python examples/batch_corpus.py --lista pendientes.txt --out demo.jsonl --sleep 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Limites de CPU ANTES de que onnxruntime se cargue: por defecto usa todos los
# nucleos y deja el PC inutilizable (paso el 7-nov: CPU al 100% y cuelgue).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

DOCUFLOW_SRC = r"C:\Users\cjcue\.hermes\docuflow-ocr\src"

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Falta PyMuPDF: pip install pymupdf")
    raise SystemExit(2)


def low_priority() -> None:
    """Baja la prioridad del proceso para no ahogar el escritorio."""
    if os.name != "nt":
        try:
            os.nice(10)
        except Exception:
            pass
        return
    try:
        import ctypes
        BELOW_NORMAL = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL)
    except Exception:
        pass


def to_host(path: str) -> Path:
    """Traduce rutas /opt/data/shared/... a la ruta de Windows."""
    return Path(path.strip())


def build_ocr():
    """Devuelve una funcion (pdf_path, page_index, pix) -> (texto, provider, conf)."""
    sys.path.insert(0, DOCUFLOW_SRC)
    try:
        from docuflow.ocr.cascade import CascadeConfig, CascadeOCR
        from docuflow.ocr.providers.rapid_ocr import RapidOCRProvider

        cascade = CascadeOCR(
            fast_provider=RapidOCRProvider(),
            accurate_provider=None,
            config=CascadeConfig(fast_threshold=0.90),
        )

        def run(img_path: str):
            r = cascade.extract(img_path)
            return (getattr(r, "text", "") or "", "docuflow-cascade", float(getattr(r, "confidence", 0.0) or 0.0))

        print("[ocr] usando docuflow cascade (RapidOCR)", flush=True)
        return run
    except Exception as exc:
        print(f"[ocr] docuflow no disponible ({type(exc).__name__}: {exc}); uso RapidOCR directo", flush=True)
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()

        def run(img_path: str):
            res, _ = engine(img_path)
            if not res:
                return ("", "rapidocr", 0.0)
            texts = [x[1] for x in res]
            scores = [float(x[2]) for x in res]
            return ("\n".join(texts), "rapidocr", sum(scores) / len(scores))

        return run


def process(pdf: Path, ocr_run, text_only: bool, dpi: int, max_pages: int | None):
    doc = fitz.open(pdf)
    pages = []
    try:
        for i, page in enumerate(doc):
            if max_pages and i >= max_pages:
                break
            text = page.get_text("text").strip()
            if text:
                pages.append({"n": i, "chars": len(text), "provider": "textlayer", "conf": 1.0, "text": text})
                continue
            if text_only:
                pages.append({"n": i, "chars": 0, "provider": "none", "conf": 0.0, "text": ""})
                continue
            pix = page.get_pixmap(dpi=dpi)
            tmp = Path(os.environ.get("TEMP", ".")) / f"wf_{os.getpid()}_{i}.png"
            pix.save(tmp)
            try:
                txt, provider, conf = ocr_run(str(tmp))
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
            pages.append({"n": i, "chars": len(txt), "provider": provider, "conf": round(conf, 3), "text": txt})
    finally:
        doc.close()
    return pages


def _worker(path: str, text_only: bool, dpi: int, max_pages):
    """Proceso hijo: construye su propio OCR y procesa un PDF."""
    run = None if text_only else build_ocr()
    return process(Path(path), run, text_only, dpi, max_pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lista", help="fichero con una ruta de PDF por linea")
    ap.add_argument("--root", help="carpeta a recorrer")
    ap.add_argument("--out", default="batch.jsonl")
    ap.add_argument("--done", default="done.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--allow-workers", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--text-only", action="store_true")
    args = ap.parse_args()

    if args.lista:
        paths = [to_host(l) for l in open(args.lista, encoding="utf-8", errors="replace") if l.strip()]
    elif args.root:
        # os.walk tolera enlaces rotos (WinError 1920 en .venv/lib64); rglob no.
        paths = []
        skip = {"node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages"}
        for dp, dn, fn in os.walk(args.root, onerror=lambda e: None):
            dn[:] = [d for d in dn if d not in skip]
            for f in fn:
                if f.lower().endswith(".pdf"):
                    paths.append(Path(dp) / f)
        paths.sort()
    else:
        print("Falta --lista o --root")
        return 2

    done = set()
    if os.path.exists(args.done):
        done = {l.strip() for l in open(args.done, encoding="utf-8", errors="replace") if l.strip()}

    if args.workers > 1 and not getattr(args, "allow_workers", False):
        print("[aviso] --workers>1 se ignora sin --allow-workers (se comio la CPU el 7-nov)")
        args.workers = 1
    if not args.text_only:
        low_priority()
    todo = [p for p in paths if str(p) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[plan] {len(paths)} PDFs, {len(done)} hechos, {len(todo)} por procesar", flush=True)

    t0 = time.time()
    n_ok = n_err = 0
    workers = max(1, args.workers)
    with open(args.out, "a", encoding="utf-8") as out, open(args.done, "a", encoding="utf-8") as dn:
        if workers == 1:
            ocr_run = None if args.text_only else build_ocr()
            for k, p in enumerate(todo, 1):
                try:
                    pages = process(p, ocr_run, args.text_only, args.dpi, args.max_pages or None)
                except Exception as exc:
                    n_err += 1
                    print(f"[skip] {Path(p).name}: {type(exc).__name__}", flush=True)
                    dn.write(f"{p}\n"); dn.flush()
                    continue
                out.write(json.dumps({"path": str(p), "pages": pages}, ensure_ascii=False) + "\n")
                out.flush()
                dn.write(f"{p}\n")
                dn.flush()
                n_ok += 1
                if args.sleep:
                    time.sleep(args.sleep)
                if k % 25 == 0 or k == len(todo):
                    print(f"[{k}/{len(todo)}] {k/max(time.time()-t0,1e-6):.2f} PDF/s | ok={n_ok} err={n_err}", flush=True)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_worker, str(p), args.text_only, args.dpi, args.max_pages or None): p for p in todo}
                for k, fut in enumerate(as_completed(futs), 1):
                    p = futs[fut]
                    try:
                        pages = fut.result()
                        out.write(json.dumps({"path": str(p), "pages": pages}, ensure_ascii=False) + "\n")
                        out.flush()
                        dn.write(f"{p}\n")
                        dn.flush()
                        n_ok += 1
                    except Exception as exc:
                        n_err += 1
                        print(f"[err] {p}: {type(exc).__name__}: {exc}", flush=True)
                    if k % 20 == 0 or k == len(todo):
                        print(f"[{k}/{len(todo)}] {k/max(time.time()-t0,1e-6):.2f} PDF/s | ok={n_ok} err={n_err}", flush=True)
    print(f"[fin] {n_ok} ok, {n_err} errores, {time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
