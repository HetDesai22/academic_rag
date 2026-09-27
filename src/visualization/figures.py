"""Generate the standard report charts as standalone HTML files in reports/figures/."""
from pathlib import Path

from src.trend_analysis.evolution import evolution_pivot
from src.visualization import charts
from src.visualization.html import html_page


def _save(fig, path, title):
    Path(path).write_text(html_page(title, "Academic Question Intelligence", [(title, fig)]), encoding="utf-8")


def generate_all(df, table, freq, trend_df, pairs, forecast_df, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    subjects = sorted(df["subject"].unique())
    specs = [
        ("01_questions_per_year", "Questions per year",
         charts.bar_counts(df["year"].astype(str).value_counts().sort_index(), "Year", "Questions")),
        ("01b_questions_per_period", "Questions per exam period",
         charts.bar_counts(df.sort_values("period_index").groupby("period", sort=False).size(), "Exam period", "Questions")),
        ("02_questions_per_subject", "Questions per subject",
         charts.bar_counts(df["subject"].value_counts(), "Subject", "Questions", horizontal=True)),
        ("03_topic_frequency", "Topic frequency (top 25)", charts.topic_frequency(freq, top=25)),
        ("06_repeated_questions", "Repeated question pairs by relationship type",
         charts.repeated_summary(pairs) if len(pairs) else charts.bar_counts({}, "", "")),
        ("07_forecast", "Forecasted topic relevance (model estimate)",
         charts.forecast_bars(forecast_df.assign(subject_short=forecast_df["subject"].map(_short)), top=20)),
        ("08_topic_distribution", "Topic distribution (subject → topic → subtopic)", charts.distribution_sunburst(df)),
    ]
    for s in subjects:
        slug = _short(s)
        specs.append((f"04_topic_by_period_{slug}", f"Topic frequency by exam period — {s}",
                      charts.evolution_heatmap(evolution_pivot(table, s))))
        top = trend_df[trend_df["subject"] == s].nlargest(6, "total_questions")["topic"]
        specs.append((f"05_trend_lines_{slug}", f"Topic trend lines (top 6) — {s}",
                      charts.trend_lines(table[(table["subject"] == s) & table["topic"].isin(top)])))
    for name, title, fig in specs:
        path = out / f"{name}.html"
        _save(fig, path, title)
        made[name] = path
    return made


def _short(subject):
    stop = {"and", "of", "the"}
    return "".join(w[0].upper() for w in subject.split() if w.lower() not in stop)
