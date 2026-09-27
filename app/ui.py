"""Reusable UI pieces for the Streamlit app: styles, headers, KPI cards, chips, callouts, evidence cards.

Colour language: navy = historical fact · amber = forecast/model estimate · rose = needs review ·
green = verified / high confidence. Colours are translucent so they work in light and dark themes.
"""
import html

import streamlit as st

from src.visualization.charts import FACT, FORECAST, GOOD, REVIEW, short, subject_colors

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&display=swap');
h1,h2,h3,[data-testid="stHeadingWithActionElements"] h1,[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3{font-family:'Source Serif 4',Georgia,serif !important;font-weight:600 !important}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1320px}
[data-testid="stLogo"]{height:2.1rem !important;max-width:15rem}
[data-testid="stSidebarHeader"]{padding-bottom:0}
h1{font-size:2.05rem !important;letter-spacing:-.2px;margin-bottom:.1rem !important}
h2{font-size:1.35rem !important;margin-top:.6rem !important}
h3{font-size:1.08rem !important}
.eyebrow{font:600 11.5px/1 Inter,sans-serif;letter-spacing:.12em;text-transform:uppercase;opacity:.6;margin-bottom:.45rem}
.lede{font-size:15px;opacity:.78;max-width:860px;margin:.1rem 0 1.1rem}
.kpi{border:1px solid rgba(128,128,128,.22);border-radius:12px;padding:14px 16px 12px;height:100%;
     background:linear-gradient(180deg,rgba(128,128,128,.04),rgba(128,128,128,0));border-top:3px solid var(--acc)}
.kpi .v{font:600 28px/1.1 'Source Serif 4',Georgia,serif;font-variant-numeric:tabular-nums}
.kpi .l{font-size:13px;font-weight:500;margin-top:4px}
.kpi .h{font-size:12px;opacity:.62;margin-top:2px}
.chip{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:500;line-height:20px;
      margin:0 5px 4px 0;border:1px solid var(--c);color:var(--c);background:color-mix(in srgb,var(--c) 10%,transparent);white-space:nowrap}
.chip.solid{background:var(--c);color:#fff}
.callout{border-radius:10px;padding:12px 16px;margin:.3rem 0 1rem;border:1px solid color-mix(in srgb,var(--c) 35%,transparent);
         background:color-mix(in srgb,var(--c) 8%,transparent);font-size:14px}
.callout b.t{color:var(--c)}
.ev{border:1px solid rgba(128,128,128,.22);border-radius:10px;padding:10px 14px;margin-bottom:8px}
.ev .q{font-size:14.5px;margin:6px 0 6px;line-height:1.45}
.ev .meta{font-size:12px;opacity:.7}
.meter{height:5px;border-radius:3px;background:rgba(128,128,128,.18);overflow:hidden;margin-top:4px}
.meter>span{display:block;height:100%;background:var(--c)}
.answer{border:1px solid rgba(128,128,128,.22);border-left:4px solid #3B72AD;border-radius:10px;padding:14px 20px 6px;margin:.4rem 0 1rem}
.legend-row{font-size:12.5px;margin:3px 0;opacity:.85}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--c);margin-right:7px;vertical-align:0}
.signal{border:1px solid rgba(128,128,128,.22);border-radius:10px;padding:12px 14px;height:100%}
.signal .n{font:600 22px/1 'Source Serif 4',serif}
.signal ul{margin:.4rem 0 0 1rem;padding:0;font-size:13px}
.fam{border-bottom:1px solid rgba(128,128,128,.18);padding:9px 2px}
.fam .t{font-size:14px}
</style>
"""

KIND_COLORS = {"fact": FACT, "forecast": FORECAST, "review": REVIEW, "good": GOOD, "neutral": "#6B7485"}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def page_header(eyebrow, title, lede=None):
    st.markdown(f"<div class='eyebrow'>{esc(eyebrow)}</div>", unsafe_allow_html=True)
    st.title(title)
    if lede:
        st.markdown(f"<div class='lede'>{esc(lede)}</div>", unsafe_allow_html=True)


def chip(text, kind="neutral", color=None, solid=False):
    c = color or KIND_COLORS.get(kind, KIND_COLORS["neutral"])
    return f"<span class='chip{' solid' if solid else ''}' style='--c:{c}'>{esc(text)}</span>"


def subject_chip(subject):
    return chip(short(subject), color=subject_colors([subject])[subject])


def kpis(items):
    """items: list of (value, label, hint, kind)."""
    cols = st.columns(len(items))
    for col, (v, label, hint, kind) in zip(cols, items):
        col.markdown(f"<div class='kpi' style='--acc:{KIND_COLORS.get(kind, FACT)}'><div class='v'>{esc(v)}</div>"
                     f"<div class='l'>{esc(label)}</div><div class='h'>{esc(hint or '')}</div></div>",
                     unsafe_allow_html=True)


def callout(kind, title, body):
    c = KIND_COLORS[kind]
    st.markdown(f"<div class='callout' style='--c:{c}'><b class='t'>{esc(title)}</b> {esc(body)}</div>",
                unsafe_allow_html=True)


def meter(value, kind="fact"):
    v = max(0.0, min(1.0, float(value)))
    return f"<div class='meter'><span style='width:{v * 100:.0f}%;--c:{KIND_COLORS[kind]}'></span></div>"


def evidence_card(e):
    sub = e.get("subtopic") or ""
    part = f"Q{e.get('question_number', '?')}({e.get('sub_part', '')})"
    head = (chip(e["tag"], "fact", solid=True) + chip(e["period"], "fact") + subject_chip(e["subject"])
            + chip(f"{part} · {e.get('marks')} marks", "neutral"))
    st.markdown(
        f"<div class='ev'>{head}<div class='q'>{esc(e['question_text'])}</div>"
        f"<div class='meta'>{esc(e['primary_topic'])} › {esc(sub)} · similarity {e['similarity']:.2f}</div>"
        f"{meter(e['similarity'])}</div>", unsafe_allow_html=True)


def sidebar_legend():
    rows = [("fact", "Historical fact (from the papers)"), ("forecast", "Forecast / model estimate"),
            ("review", "Needs human review"), ("good", "Verified / high confidence")]
    st.markdown("".join(f"<div class='legend-row'><span class='dot' style='--c:{KIND_COLORS[k]}'></span>{esc(t)}</div>"
                        for k, t in rows), unsafe_allow_html=True)


TREND_META = {
    "increasing": ("Increasing", "📈", "fact"), "declining": ("Declining", "📉", "review"),
    "emerging": ("Emerging", "🌱", "good"), "dormant": ("Dormant", "💤", "neutral"),
    "consistently_recurring": ("Recurring", "🔁", "fact"), "stable": ("Stable", "➖", "neutral"),
    "sporadic": ("Sporadic", "·", "neutral"),
}


def trend_label(t):
    name, icon, _ = TREND_META.get(t, (t, "", "neutral"))
    return f"{icon} {name}"
