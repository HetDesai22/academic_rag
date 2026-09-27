"""Topic frequency tables over exam periods.

Unit of time: exam period (Winter/Summer session, e.g. W2021, S2022) by default - each subject
has one paper per period - or calendar year (config trends.period). Counts are filled with zeros
for periods where the subject had a paper but the topic did not appear.
"""
import json

import pandas as pd


def _period_cols(df, period):
    if period == "year":
        d = df.copy()
        d["period"] = d["year"].astype(str)
        d["period_index"] = d["year"].astype(float)
        return d
    return df


def _secondary(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v) if isinstance(v, str) and v else []
    except json.JSONDecodeError:
        return []


def topic_period_table(df, level="primary_topic", period="session"):
    """Long table: subject, <level>, period, period_index, questions, mentions, marks, share."""
    d = _period_cols(df, period)
    papers = d.groupby(["subject", "period", "period_index"]).agg(paper_questions=("question_id", "size"),
                                                                  paper_marks=("marks", "sum")).reset_index()
    prim = d.groupby(["subject", level, "period"]).agg(questions=("question_id", "size"), marks=("marks", "sum")).reset_index()
    if level == "primary_topic":
        sec = d.assign(_sec=d["secondary_topics"].map(_secondary)).explode("_sec").dropna(subset=["_sec"])
        sec = sec.groupby(["subject", "_sec", "period"]).size().rename("secondary").reset_index().rename(columns={"_sec": level})
    else:
        sec = pd.DataFrame(columns=["subject", level, "period", "secondary"])
    topics = d[["subject", level]].drop_duplicates()
    grid = topics.merge(papers, on="subject")
    t = (grid.merge(prim, on=["subject", level, "period"], how="left")
             .merge(sec, on=["subject", level, "period"], how="left"))
    t[["questions", "marks", "secondary"]] = t[["questions", "marks", "secondary"]].fillna(0)
    t["questions"] = t["questions"].astype(int)
    t["secondary"] = t["secondary"].astype(int)
    t["mentions"] = t["questions"] + t["secondary"]
    t["share"] = (t["questions"] / t["paper_questions"]).round(4)
    t["marks_share"] = (t["marks"] / t["paper_marks"]).round(4)
    t = t.rename(columns={level: "topic"}) if level != "topic" else t
    return t.sort_values(["subject", "topic", "period_index"]).reset_index(drop=True)


def topic_summary(df, table, recent_periods=3):
    """One row per (subject, topic) with totals, shares, marks mix and period coverage."""
    rows = []
    for (subject, topic), g in table.groupby(["subject", "topic"]):
        g = g.sort_values("period_index")
        sub = df[(df["subject"] == subject) & (df["primary_topic"] == topic)]
        total_subject = int((df["subject"] == subject).sum())
        appeared = g.loc[g["questions"] > 0, "period"].tolist()
        recent = g.tail(recent_periods)
        rows.append({
            "subject": subject, "topic": topic,
            "total_questions": int(g["questions"].sum()),
            "share_of_subject": round(g["questions"].sum() / total_subject, 4) if total_subject else 0.0,
            "total_marks": float(g["marks"].sum()),
            "secondary_mentions": int(g["secondary"].sum()),
            "marks_mix": json.dumps({str(int(k)): int(v) for k, v in sub["marks"].value_counts().sort_index().items()}),
            "n_periods": int(len(g)), "periods_appeared": len(appeared),
            "appearance_ratio": round(len(appeared) / len(g), 3) if len(g) else 0.0,
            "first_period": appeared[0] if appeared else "", "last_period": appeared[-1] if appeared else "",
            "supporting_periods": ", ".join(appeared),
            "recent_questions": int(recent["questions"].sum()),
            "recent_periods": ", ".join(recent["period"]),
            "counts_by_period": json.dumps(dict(zip(g["period"], g["questions"].astype(int)))),
            "counts_by_year": json.dumps({str(int(y)): int(n) for y, n in sub["year"].value_counts().sort_index().items()}),
        })
    return pd.DataFrame(rows)


def global_topic_frequency(df):
    """Topic frequency across all subjects (topic names are shared where concepts are shared)."""
    return (df.groupby(["subject", "primary_topic"]).agg(questions=("question_id", "size"), marks=("marks", "sum"))
              .reset_index().rename(columns={"primary_topic": "topic"})
              .assign(share=lambda x: (x["questions"] / len(df)).round(4)))
