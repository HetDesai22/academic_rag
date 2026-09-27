"""Plotly figures shared by the HTML reports and the Streamlit app.

Visual language (matches .streamlit/config.toml):
  FACT (navy)      historical data          FORECAST (amber)  model estimates
  REVIEW (rose)    needs human review       GOOD (green)      verified / high confidence
Each subject keeps one colour everywhere. Backgrounds are transparent so figures read well in
both light and dark themes.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

FACT = "#3B72AD"   # mid-tone navy: readable on light and dark backgrounds
FACT_LIGHT = "#8FB3D6"
FORECAST = "#C98A00"
REVIEW = "#B4415A"
GOOD = "#2E7D5B"
MUTED = "#8A93A3"
SUBJECT_PALETTE = ["#3B72AD", "#2A9D8F", "#8D6CC0", "#D0654B", "#6F8BA8", "#A0822F"]
TOPIC_PALETTE = ["#0F7B8A", "#B5527A", "#6C8E2F", "#8A5A2B", "#4F5DB0", "#C07A1E", "#5E8C8A", "#9B4F9E"]
CONFIDENCE_COLORS = {"High": GOOD, "Medium": FORECAST, "Low": REVIEW}

pio.templates["aqi"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12.5, color="#5B6474"),
    title=dict(font=dict(family="Source Serif 4, Georgia, serif", size=16), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=SUBJECT_PALETTE,
    xaxis=dict(gridcolor="rgba(128,128,128,0.16)", zerolinecolor="rgba(128,128,128,0.3)", linecolor="rgba(128,128,128,0.35)"),
    yaxis=dict(gridcolor="rgba(128,128,128,0.16)", zerolinecolor="rgba(128,128,128,0.3)", linecolor="rgba(128,128,128,0.35)"),
    hoverlabel=dict(font=dict(family="Inter, sans-serif", size=12)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
))
TEMPLATE = "aqi"
_subject_colors = {}


def subject_colors(subjects):
    """Stable subject -> colour map (sorted order)."""
    for i, s in enumerate(sorted(set(subjects) | set(_subject_colors))):
        _subject_colors.setdefault(s, SUBJECT_PALETTE[i % len(SUBJECT_PALETTE)])
    return {s: _subject_colors[s] for s in subjects}


def short(subject):
    return "".join(w[0].upper() for w in str(subject).split() if w.lower() not in {"and", "of", "the"})


def _trunc(text, n=38):
    text = str(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _layout(fig, height=400, **kw):
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=8, r=16, t=40 if kw.get("title") else 12, b=8), **kw)
    return fig


def bar_counts(series, xlabel, ylabel, horizontal=False, title=None, color=FACT):
    s = pd.Series(series)
    if horizontal:
        s = s.sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=[str(i) for i in s.index], orientation="h", marker_color=color,
                               text=s.values, textposition="outside", cliponaxis=False))
        fig.update_xaxes(title=ylabel, showgrid=False, showticklabels=False)
        fig.update_yaxes(title=None)
        return _layout(fig, height=max(260, 34 * len(s) + 60), title=title)
    fig = go.Figure(go.Bar(x=[str(i) for i in s.index], y=s.values, marker_color=color, text=s.values,
                           textposition="outside", cliponaxis=False))
    fig.update_xaxes(title=xlabel, type="category")
    fig.update_yaxes(title=ylabel)
    return _layout(fig, title=title)


def histogram(series, xlabel, title=None):
    fig = go.Figure(go.Histogram(x=pd.Series(series).dropna(), marker_color=FACT, nbinsx=30))
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="Questions")
    return _layout(fig, title=title)


def heatmap(pivot, xlabel, ylabel, title=None, colorscale=None, zlabel="Questions"):
    colorscale = colorscale or [[0, "#EEF3F9"], [0.35, "#9DBBDB"], [1, "#2A5E94"]]
    fig = go.Figure(go.Heatmap(z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index],
                               colorscale=colorscale, colorbar=dict(title=zlabel, thickness=10),
                               xgap=2, ygap=2, text=pivot.values, texttemplate="%{text}",
                               hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>"))
    fig.update_xaxes(title=xlabel, type="category", side="top", showgrid=False)
    fig.update_yaxes(title=None, autorange="reversed", showgrid=False)
    return _layout(fig, height=max(300, 30 * len(pivot) + 110), title=title)


def topic_frequency(freq, top=None, title=None):
    """freq: DataFrame with columns topic, subject, questions."""
    d = freq.sort_values("questions", ascending=False)
    if top:
        d = d.head(top)
    d = d.iloc[::-1].assign(label=lambda x: x["topic"].map(_trunc) + "  · " + x["subject"].map(short))
    cmap = subject_colors(d["subject"].unique())
    fig = go.Figure()
    for s in sorted(d["subject"].unique()):
        g = d[d["subject"] == s]
        fig.add_bar(x=g["questions"], y=g["label"], orientation="h", name=f"{short(s)} — {s}", marker_color=cmap[s],
                    text=g["questions"], textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}<br>%{x} questions<extra></extra>")
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=d["label"].tolist(), automargin=True)
    fig.update_xaxes(title=None, showgrid=False, showticklabels=False)
    # Subject is encoded in the label suffix and colour (legend shown in the app sidebar).
    return _layout(fig, height=max(320, 26 * len(d) + 40), title=title, barmode="overlay", bargap=0.35,
                   showlegend=False)


def trend_lines(long, title=None, y="questions"):
    """long: DataFrame with columns period, period_index, topic, questions."""
    d = long.sort_values("period_index")
    fig = px.line(d, x="period", y=y, color="topic", markers=True, template=TEMPLATE,
                  color_discrete_sequence=TOPIC_PALETTE)
    fig.update_traces(line=dict(width=2.4), marker=dict(size=7))
    fig.update_xaxes(title=None, type="category", categoryorder="array", categoryarray=list(dict.fromkeys(d["period"])))
    fig.update_yaxes(title="Questions per exam" if y == "questions" else y, rangemode="tozero")
    return _layout(fig, height=420, title=title, legend=dict(orientation="h", y=-0.18, x=0, title=None))


def distribution_sunburst(df, title=None):
    d = df.groupby(["subject", "primary_topic", "subtopic"]).size().reset_index(name="questions")
    cmap = subject_colors(d["subject"].unique())
    fig = px.sunburst(d, path=["subject", "primary_topic", "subtopic"], values="questions", color="subject",
                      color_discrete_map=cmap, template=TEMPLATE)
    fig.update_traces(insidetextorientation="radial", hovertemplate="%{label}<br>%{value} questions<extra></extra>")
    return _layout(fig, height=620, title=title)


def forecast_bars(fc, top=15, title=None):
    """Expected questions next exam (amber, model estimate) vs. historical mean per exam (navy, fact)."""
    d = fc.sort_values("predicted_questions", ascending=False).head(top).iloc[::-1]
    hist_mean = d["historical_frequency"] / d["periods_observed"].clip(lower=1)
    labels = d["topic"].map(_trunc) + "  · " + d["subject"].map(short)
    fig = go.Figure()
    fig.add_bar(x=d["predicted_questions"], y=labels, orientation="h", name="Forecast (model estimate)",
                marker=dict(color=FORECAST, opacity=0.85), cliponaxis=False,
                text=[f"{p:.1f}  ·  {c}" for p, c in zip(d["predicted_questions"], d["forecast_confidence"])],
                textposition="outside",
                customdata=d[["forecast_score", "forecast_confidence", "trend"]].values,
                hovertemplate="%{y}<br>expected %{x:.2f} question(s)<br>P(≥1) %{customdata[0]:.2f} · "
                              "confidence %{customdata[1]} · trend %{customdata[2]}<extra></extra>")
    fig.add_scatter(x=hist_mean, y=labels, mode="markers", name="Historical average (fact)",
                    marker=dict(symbol="line-ns", size=18, line=dict(width=3, color=FACT)),
                    hovertemplate="%{y}<br>historical average %{x:.2f} per exam<extra></extra>")
    fig.update_xaxes(title="Questions in the next exam", rangemode="tozero")
    fig.update_yaxes(title=None, automargin=True)
    return _layout(fig, height=max(340, 30 * len(d) + 110), title=title,
                   legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def repeated_summary(pairs, title=None):
    order = ["exact", "near_identical", "rephrased", "semantically_similar"]
    counts = pairs["relationship_type"].value_counts().reindex(order).fillna(0).astype(int)
    colors = [FACT, "#3F74A6", FACT_LIGHT, "#C9D7E6"]
    fig = go.Figure(go.Bar(x=[o.replace("_", " ") for o in counts.index], y=counts.values, marker_color=colors,
                           text=counts.values, textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="Question pairs across papers")
    return _layout(fig, height=340, title=title)


def recycled_share(df, title=None):
    """Share of each exam's questions that repeat a question family seen in an earlier paper (subject x exam)."""
    d = df.sort_values("period_index").copy()
    fam = d["question_family"].fillna("")
    seen, flags = {}, []
    for (s_, f, pi) in zip(d["subject"], fam, d["period_index"]):
        key = (s_, f)
        flags.append(bool(f) and key in seen and seen[key] < pi)
        if f and key not in seen:
            seen[key] = pi
    d["recycled"] = flags
    order = list(dict.fromkeys(d["period"]))
    pv = (d.groupby(["subject", "period"])["recycled"].mean().unstack()[order]
            .rename(index=short).sort_index())
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=[f"{c[0]}'{c[-2:]}" for c in pv.columns], y=pv.index, zmin=0, zmax=1, xgap=3, ygap=3,
        colorscale=[[0, "#EEF3F9"], [0.5, "#9DBBDB"], [1, "#2A5E94"]],
        text=[[f"{v:.0%}" if v == v else "" for v in row] for row in pv.values], texttemplate="%{text}",
        colorbar=dict(title="Recycled", tickformat=".0%", thickness=10),
        hovertemplate="%{y} · %{x}<br>%{z:.0%} of questions asked in an earlier paper<extra></extra>"))
    fig.update_xaxes(type="category", side="top", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, height=280, title=title)


def evolution_heatmap(pivot, title=None):
    return heatmap(pivot, None, None, title=title)


def backtest_chart(metrics, title=None):
    d = metrics.sort_values("mae")
    colors = [GOOD if i == 0 else FACT_LIGHT for i in range(len(d))]
    fig = go.Figure([go.Bar(name="MAE", x=d["method"].str.replace("_", " "), y=d["mae"], marker_color=colors,
                            text=d["mae"].round(3), textposition="outside", cliponaxis=False),
                     go.Scatter(name="RMSE", x=d["method"].str.replace("_", " "), y=d["rmse"], mode="markers",
                                marker=dict(symbol="diamond", size=10, color=MUTED))])
    fig.update_yaxes(title="Error (questions per topic per exam)", rangemode="tozero")
    return _layout(fig, height=340, title=title, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
