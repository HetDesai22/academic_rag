"""Topic evolution: which topics dominate each exam period, and how topic mix shifts."""
import pandas as pd


def dominant_topics(table, top=2):
    """Per subject and period: the most-asked topics (ties kept)."""
    rows = []
    for (subject, period), g in table.groupby(["subject", "period"], sort=False):
        g = g[g["questions"] > 0].sort_values("questions", ascending=False)
        if g.empty:
            continue
        cutoff = g["questions"].iloc[min(top, len(g)) - 1]
        best = g[g["questions"] >= cutoff]
        rows.append({"subject": subject, "period": period, "period_index": g["period_index"].iloc[0],
                     "dominant_topics": "; ".join(f"{t} ({n})" for t, n in zip(best["topic"], best["questions"])),
                     "questions_in_paper": int(g["paper_questions"].iloc[0])})
    return pd.DataFrame(rows).sort_values(["subject", "period_index"]).reset_index(drop=True)


def evolution_pivot(table, subject, value="questions"):
    g = table[table["subject"] == subject].sort_values("period_index")
    order = list(dict.fromkeys(g["period"]))
    p = g.pivot_table(index="topic", columns="period", values=value, aggfunc="sum", fill_value=0)[order]
    return p.loc[p.sum(1).sort_values(ascending=False).index]


def topic_shift(table, subject, split=None):
    """Share of questions per topic in the first vs second half of the periods."""
    g = table[table["subject"] == subject]
    periods = sorted(g["period_index"].unique())
    split = split or periods[len(periods) // 2]
    early = g[g["period_index"] < split].groupby("topic")["questions"].sum()
    late = g[g["period_index"] >= split].groupby("topic")["questions"].sum()
    d = pd.DataFrame({"early_share": early / max(early.sum(), 1), "late_share": late / max(late.sum(), 1)}).fillna(0)
    d["change"] = d["late_share"] - d["early_share"]
    return d.sort_values("change", ascending=False).round(3)
