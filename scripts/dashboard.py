#!/usr/bin/env python3
"""
dashboard.py — informe HTML autocontenido a partir de un JSONL de DocuFlow.

Lee el JSONL que produce un lote (una linea por documento, con `pages`) y
escribe un HTML sin dependencias: todo el CSS y los SVG van inline, sin CDN,
sin JavaScript. Se abre en cualquier navegador y se puede capturar para el README.

Uso:
    python scripts/dashboard.py salida.jsonl -o dashboard.html --titulo "WeaAngel"

Diseño: fondo neutro, un solo acento, numeros en monospace, cero adornos.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path):
    docs = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return docs


def summarize(docs):
    pages = [p for d in docs for p in d.get("pages", [])]
    by_provider = Counter(p.get("provider", "?") for p in pages)
    conf = defaultdict(list)
    for p in pages:
        if p.get("provider") not in (None, "textlayer"):
            conf[p["provider"]].append(float(p.get("conf") or 0.0))
    chars = sum(int(p.get("chars") or 0) for p in pages)
    review = []
    for d in docs:
        low = [p for p in d.get("pages", []) if p.get("provider") not in (None, "textlayer")
               and float(p.get("conf") or 0.0) < 0.80]
        if low:
            review.append((str(d.get("path", "")), min(float(p["conf"]) for p in low), len(low)))
    review.sort(key=lambda x: x[1])
    return {
        "docs": len(docs),
        "pages": len(pages),
        "chars": chars,
        "by_provider": by_provider,
        "conf": {k: (statistics.mean(v), len(v)) for k, v in conf.items()},
        "textlayer": by_provider.get("textlayer", 0),
        "ocr": sum(v for k, v in by_provider.items() if k != "textlayer"),
        "review": review[:12],
        "review_total": len(review),
    }


def fmt_chars(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f} M"
    if n >= 1_000:
        return f"{n/1_000:.1f} k"
    return str(n)


def bar(value, best, width=220, color="#5eead4"):
    pct = 0 if not best else max(2, round(value / best * 100))
    return (f'<svg width="{width}" height="10" role="img" aria-label="{value:.2f}">'
            f'<rect width="{width}" height="10" fill="#1e242c" rx="2"/>'
            f'<rect width="{round(width * pct / 100)}" height="10" fill="{color}" rx="2"/></svg>')


def render(s: dict, titulo: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pct_text = (s["textlayer"] / s["pages"] * 100) if s["pages"] else 0
    max_conf = max((v[0] for v in s["conf"].values()), default=1) or 1

    rows_conf = "".join(
        f"<tr><td>{k}</td><td class=n>{v[1]:,}</td><td class=n>{v[0]:.3f}</td>"
        f"<td>{bar(v[0], max_conf)}</td></tr>"
        for k, v in sorted(s["conf"].items(), key=lambda kv: -kv[1][0])
    ) or '<tr><td colspan=4 class=empty>Sin páginas OCR en este lote</td></tr>'

    rows_review = "".join(
        f"<tr><td class=path title=\"{p}\">{Path(p).name}</td><td class=n>{c:.2f}</td><td class=n>{n}</td></tr>"
        for p, c, n in s["review"]
    ) or '<tr><td colspan=3 class=empty>Nada por debajo de 0.80</td></tr>'

    return f"""<!doctype html>
<html lang="es"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DocuFlow — {titulo}</title>
<style>
  :root {{ --bg:#0f1115; --panel:#161a20; --line:#232a33; --fg:#e6edf3; --muted:#8b949e; --accent:#5eead4; }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif }}
  header {{ padding:28px 32px 20px; border-bottom:1px solid var(--line) }}
  h1 {{ margin:0; font-size:18px; font-weight:600; letter-spacing:.2px }}
  .sub {{ color:var(--muted); margin-top:4px; font-size:13px }}
  main {{ padding:24px 32px 40px; max-width:1080px }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px }}
  .kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px }}
  .kpi .v {{ font:600 22px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace; color:var(--accent) }}
  .kpi .l {{ color:var(--muted); font-size:12px; margin-top:4px; text-transform:uppercase; letter-spacing:.06em }}
  h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
        margin:26px 0 10px; font-weight:600 }}
  table {{ width:100%; border-collapse:collapse; background:var(--panel);
           border:1px solid var(--line); border-radius:8px; overflow:hidden }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); font-size:13px }}
  th {{ color:var(--muted); font-weight:500; font-size:12px }}
  tr:last-child td {{ border-bottom:none }}
  td.n {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; text-align:right; width:88px }}
  td.path {{ max-width:520px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap }}
  .empty {{ color:var(--muted); text-align:center; padding:16px }}
  footer {{ color:var(--muted); font-size:12px; padding:0 32px 28px }}
</style></head>
<body>
<header>
  <h1>DocuFlow — lote «{titulo}»</h1>
  <div class="sub">{s['docs']:,} documentos · {s['pages']:,} páginas · generado {now}</div>
</header>
<main>
  <section class="kpis">
    <div class="kpi"><div class="v">{s['docs']:,}</div><div class="l">documentos</div></div>
    <div class="kpi"><div class="v">{s['pages']:,}</div><div class="l">páginas</div></div>
    <div class="kpi"><div class="v">{pct_text:.0f}%</div><div class="l">capa de texto</div></div>
    <div class="kpi"><div class="v">{fmt_chars(s['chars'])}</div><div class="l">caracteres</div></div>
  </section>

  <h2>Confianza por proveedor</h2>
  <table><thead><tr><th>Proveedor</th><th>Páginas</th><th>Media</th><th></th></tr></thead>
  <tbody>{rows_conf}</tbody></table>

  <h2>Cola de revisión — {s['review_total']:,} documentos bajo 0.80</h2>
  <table><thead><tr><th>Documento</th><th>Conf. mín.</th><th>Págs</th></tr></thead>
  <tbody>{rows_review}</tbody></table>
</main>
<footer>Generado por <code>scripts/dashboard.py</code> · sin dependencias · datos reales del lote</footer>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("-o", "--out", default="dashboard.html")
    ap.add_argument("--titulo", default="lote")
    args = ap.parse_args()
    docs = load(Path(args.jsonl))
    if not docs:
        print("JSONL vacio")
        return 1
    html = render(summarize(docs), args.titulo)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"dashboard: {args.out} ({len(html)//1024} KB, {len(docs)} docs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
