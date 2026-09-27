"""Minimal self-contained HTML report builder (Plotly loaded once from its CDN)."""
import html

import pandas as pd

PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

CSS = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#1d2433;--muted:#5b6475;--line:#e3e6ec;--accent:#2E5E8C}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--card:#1c2029;--ink:#e6e9ef;--muted:#9aa3b2;--line:#2c3240;--accent:#7fb0e0}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Segoe UI,system-ui,sans-serif}
header{padding:28px 32px 8px;max-width:1200px;margin:auto}
h1{margin:0 0 4px;font-size:26px} .sub{color:var(--muted)}
main{max-width:1200px;margin:auto;padding:8px 32px 40px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:16px 0}
h2{font-size:18px;margin:0 0 12px;color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600} .scroll{max-height:460px;overflow:auto}
.note{color:var(--muted);font-size:13px} .badge{display:inline-block;padding:1px 8px;border-radius:9px;background:var(--line);font-size:12px}
@media (max-width:700px){header,main{padding-left:16px;padding-right:16px}}
"""


def _render(content):
    if hasattr(content, "to_html") and hasattr(content, "layout"):  # plotly Figure
        return content.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})
    return str(content)


def html_page(title, subtitle, sections, generated_at=""):
    body = "\n".join(f"<section><h2>{html.escape(t)}</h2>{_render(c)}</section>" for t, c in sections)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<script src="{PLOTLY_CDN}"></script><style>{CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1><div class="sub">{html.escape(subtitle)}{' · generated ' + html.escape(generated_at) if generated_at else ''}</div></header>
<main>{body}</main></body></html>"""


def kv_table(d):
    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in d.items())
    return f"<table>{rows}</table>"


def df_table(df, max_rows=200):
    if df is None or len(df) == 0:
        return "<p class='note'>None.</p>"
    return "<div class='scroll'>" + pd.DataFrame(df).head(max_rows).to_html(index=False, border=0, escape=True) + "</div>"


def note(text):
    return f"<p class='note'>{html.escape(text)}</p>"
