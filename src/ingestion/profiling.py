"""Automated data profiling -> reports/data_profile.json + reports/data_profile.html."""
import re

import pandas as pd

from src.utils.common import now_iso
from src.visualization import charts
from src.visualization.html import html_page, kv_table, df_table


def _norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()


def profile(df, meta, min_chars=8):
    q = df["question_text"].fillna("").astype(str)
    key = q.map(_norm)
    has = lambda c: c in df.columns  # noqa: E731

    suspicious = []
    def flag(mask, reason):
        for _, r in df[mask].iterrows():
            suspicious.append({"question_id": r["question_id"], "reason": reason, "text": str(r["question_text"])[:120]})
    flag(q.str.strip().str.len() < min_chars, f"shorter than {min_chars} characters")
    flag(q.map(lambda t: any("\U0001D400" <= ch <= "\U0001D7FF" for ch in t)),
         "mathematical alphanumeric symbols (PDF math extraction)")
    flag(q.str.match(r"^\s*-?\d+(\s+-?\d+)+\s+[A-Za-z]"), "starts with a numeric fragment (text from a neighbouring question)")
    flag(q.str.contains(r"\bPage\s*\d*\s*of\s*\d*\s*$"), "contains a page footer")
    flag(q.str.count(r"\b\d+\b") >= 12, "many standalone numbers (flattened table or matrix)")
    if has("year"):
        flag(df["year"].isna(), "missing year")
    if has("marks"):
        m = pd.to_numeric(df["marks"], errors="coerce")
        flag(m.isna(), "missing or non-numeric marks")

    repeated = key[key.duplicated(keep=False)]
    top_repeats = (df.assign(_k=key)[key.isin(repeated)].groupby("_k")
                     .agg(times=("question_id", "size"), periods=("period", lambda s: sorted(set(map(str, s)))),
                          example=("question_text", "first"))
                     .sort_values("times", ascending=False).head(15))

    rep = {
        "generated_at": now_iso(),
        "input_files": meta["input_files"],
        "rows": int(len(df)),
        "columns_in_source": len(meta["original_columns"]),
        "source_columns": meta["original_columns"],
        "column_mapping": meta["column_mapping"],
        "standardized_fields_missing": meta["missing_fields"],
        "derived_fields": meta["derived_fields"],
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n and not c.startswith("_")},
        "duplicate_rows": int(df.drop(columns=[c for c in df.columns if c.startswith("_")]).duplicated().sum()),
        "duplicate_question_ids": int(df["question_id"].duplicated().sum()),
        "duplicate_question_texts": {"rows_with_repeated_text": int(len(repeated)),
                                     "distinct_repeated_texts": int(repeated.nunique())},
        "unique_subjects": int(df["subject"].nunique()),
        "unique_years": int(df["year"].nunique()) if has("year") else None,
        "years": sorted(int(y) for y in df["year"].dropna().unique()) if has("year") else None,
        "exam_periods": sorted(df["period"].dropna().unique(), key=lambda p: df.loc[df["period"] == p, "period_index"].iloc[0]) if has("period") else None,
        "year_distribution": {int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()} if has("year") else None,
        "subject_distribution": df["subject"].value_counts().to_dict(),
        "questions_per_period": df.groupby("period", sort=False).size().to_dict() if has("period") else None,
        "questions_per_subject_year": {s: {int(y): int(n) for y, n in g["year"].value_counts().sort_index().items()}
                                       for s, g in df.groupby("subject")} if has("year") else None,
        "question_length_chars": q.str.len().describe().round(1).to_dict(),
        "question_length_words": q.str.split().str.len().describe().round(1).to_dict(),
        "marks_distribution": {str(k): int(v) for k, v in df["marks"].value_counts().sort_index().items()} if has("marks") else None,
        "top_repeated_questions": [
            {"text": r["example"][:160], "times": int(r["times"]), "periods": r["periods"]} for _, r in top_repeats.iterrows()],
        "suspicious_records": {"count": len(suspicious), "records": suspicious},
    }
    return rep


def summary_lines(rep):
    return [
        f"rows={rep['rows']}  subjects={rep['unique_subjects']}  years={rep['years']}",
        f"periods={rep['exam_periods']}",
        f"missing standardized fields: {rep['standardized_fields_missing'] or 'none'}",
        f"missing values: {rep['missing_values'] or 'none'}",
        f"duplicate rows={rep['duplicate_rows']}  repeated question texts={rep['duplicate_question_texts']}",
        f"suspicious records={rep['suspicious_records']['count']}",
    ]


def profile_html(df, rep):
    sections = [
        ("Overview", kv_table({
            "Rows": rep["rows"], "Source columns": rep["columns_in_source"], "Subjects": rep["unique_subjects"],
            "Years": ", ".join(map(str, rep["years"] or [])), "Exam periods": ", ".join(rep["exam_periods"] or []),
            "Duplicate rows": rep["duplicate_rows"], "Duplicate question IDs": rep["duplicate_question_ids"],
            "Rows with repeated question text": rep["duplicate_question_texts"]["rows_with_repeated_text"],
            "Standardized fields not available": ", ".join(rep["standardized_fields_missing"]) or "none",
            "Derived fields": ", ".join(rep["derived_fields"]) or "none",
        })),
        ("Column mapping", kv_table({k: v or "— (not in source)" for k, v in rep["column_mapping"].items()})),
        ("Missing values", kv_table(rep["missing_values"] or {"(none)": ""})),
        ("Questions per year", charts.bar_counts(df["year"].astype(str).value_counts().sort_index(), "Year", "Questions")),
        ("Questions per exam period", charts.bar_counts(df.groupby("period", sort=False).size(), "Exam period", "Questions")),
        ("Questions per subject", charts.bar_counts(df["subject"].value_counts(), "Subject", "Questions", horizontal=True)),
        ("Questions per subject and year", charts.heatmap(df.pivot_table(index="subject", columns="year", values="question_id", aggfunc="count", fill_value=0), "Year", "Subject")),
        ("Marks distribution", charts.bar_counts(df["marks"].astype(str).value_counts().sort_index(), "Marks", "Questions") if "marks" in df else "<p>No marks column.</p>"),
        ("Question length (words)", charts.histogram(df["question_text"].astype(str).str.split().str.len(), "Words per question")),
        ("Most repeated question texts", df_table(pd.DataFrame(rep["top_repeated_questions"]))),
        ("Suspicious records", df_table(pd.DataFrame(rep["suspicious_records"]["records"]))),
    ]
    return html_page("Data Profile", "Automated profile of the historical question dataset", sections, rep["generated_at"])
