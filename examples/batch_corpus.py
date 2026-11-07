#!/usr/bin/env python3
"""
batch_corpus.py - batch OCR of a local corpus with docuflow (cascade) or RapidOCR.

Strategy (the part that saves the time):
  1. If the PDF page ALREADY has a text layer -> use it, cost ~0.
  2. If it does not                          -> render to image and OCR.
  3. OCR goes through docuflow's cascade when importable; otherwise RapidOCR.

Resumable: every finished PDF is recorded in done.txt and skipped next time.
Output: one JSONL line per document: {path, pages:[{n, text, provider, conf}]}.

Usage:
  python examples/batch_corpus.py --root samples --out demo.jsonl
  python examples/batch_corpus.py --list pending.txt --out demo.jsonl --sleep 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# CPU limits BEFORE onnxruntime loads: by default it grabs every core and makes
# the machine unusable (happened once: 100% CPU, desktop froze).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

DOCUFLOW_SRC = r"C:\Users\cjcue\.hermes\docuflow-ocr\src"

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Falta PyMuPDF: pip install pymupdf")
    raise SystemExit(2)


def low_priority() -> None:
    """Lower the process priority so the desktop stays responsive."""
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
    """Normalise a path (kept for callers that pass container-style paths)."""
    return Path(path.strip())


def build_ocr():
    """Return a callable (image_path) -> (text, provider, confidence)."""
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

        print("[ocr] using docuflow cascade (RapidOCR)", flush=True)
        return run
    except Exception as exc:
        print(f"[ocr] docuflow unavailable ({type(exc).__name__}: {exc}); falling back to RapidOCR", flush=True)
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
    """Child process: builds its own OCR engine and processes one PDF."""
    run = None if text_only else build_ocr()
    return process(Path(path), run, text_only, dpi, max_pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", help="file with one PDF path per line")
    ap.add_argument("--root", help="folder to walk")
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

    if args.list:
        paths = [to_host(l) for l in open(args.list, encoding="utf-8", errors="replace") if l.strip()]
    elif args.root:
        # os.walk tolerates broken symlinks (WinError 1920 on .venv/lib64); rglob does not.
        paths = []
        skip = {"node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages"}
        for dp, dn, fn in os.walk(args.root, onerror=lambda e: None):
            dn[:] = [d for d in dn if d not in skip]
            for f in fn:
                if f.lower().endswith(".pdf"):
                    paths.append(Path(dp) / f)
        paths.sort()
    else:
        print("Missing --list or --root")
        return 2

    done = set()
    if os.path.exists(args.done):
        done = {l.strip() for l in open(args.done, encoding="utf-8", errors="replace") if l.strip()}

    if args.workers > 1 and not getattr(args, "allow_workers", False):
        print("[warn] --workers>1 ignored without --allow-workers (it ate the CPU once)")
        args.workers = 1
    if not args.text_only:
        low_priority()
    todo = [p for p in paths if str(p) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[plan] {len(paths)} PDFs, {len(done)} done, {len(todo)} to process", flush=True)

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
    print(f"[done] {n_ok} ok, {n_err} errors, {time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
