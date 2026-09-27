"""Statistical trend classification of topic time series (thresholds in config `trends`).

For each (subject, topic) series of per-period question counts:
  slope       OLS slope (questions per period)
  tau, p      Kendall's tau rank correlation with time (monotonic trend) and its p-value
Labels (first matching rule wins; nothing is called a trend from a single appearance):
  sporadic               appeared in fewer than `min_appearances` periods  -> no trend claimed
  emerging               <= `emerging_max_early_total` questions before the recent window and
                         >= `emerging_min_recent_appearances` of the last `recent_periods` periods
  dormant                absent in the last `dormant_min_absent_periods` periods, present before
  increasing             slope >= slope_threshold and tau >= tau_threshold
  declining              slope <= -slope_threshold and tau <= -tau_threshold
  consistently_recurring appears in >= recurring_ratio of periods (and no clear slope)
  stable                 otherwise (appears regularly with small variation)
"""
import numpy as np
import pandas as pd
from scipy import stats


def series_stats(counts):
    y = np.asarray(counts, dtype=float)
    x = np.arange(len(y))
    if len(y) < 2 or np.all(y == y[0]):
        slope, tau, p = 0.0, 0.0, 1.0
    else:
        slope = float(np.polyfit(x, y, 1)[0])
        tau, p = stats.kendalltau(x, y)
        tau = 0.0 if np.isnan(tau) else float(tau)
        p = 1.0 if np.isnan(p) else float(p)
    return {"slope": round(slope, 4), "kendall_tau": round(tau, 3), "kendall_p": round(p, 4),
            "mean": round(float(y.mean()), 3) if len(y) else 0.0, "std": round(float(y.std()), 3) if len(y) else 0.0}


def classify_series(counts, tcfg):
    y = np.asarray(counts, dtype=float)
    n = len(y)
    s = series_stats(y)
    appear = int((y > 0).sum())
    r = tcfg["recent_periods"]
    recent, early = y[-r:], y[:-r]
    reason = ""
    if appear < tcfg["min_appearances"]:
        label, reason = "sporadic", f"appeared in only {appear} of {n} periods"
    elif early.sum() <= tcfg["emerging_max_early_total"] and (recent > 0).sum() >= tcfg["emerging_min_recent_appearances"]:
        label, reason = "emerging", f"{int(early.sum())} question(s) before the last {r} periods, present in {int((recent > 0).sum())} of them"
    elif n >= tcfg["dormant_min_absent_periods"] and y[-tcfg["dormant_min_absent_periods"]:].sum() == 0:
        label, reason = "dormant", f"absent in the last {tcfg['dormant_min_absent_periods']} periods"
    elif s["slope"] >= tcfg["slope_threshold"] and s["kendall_tau"] >= tcfg["kendall_tau_threshold"]:
        label, reason = "increasing", f"slope {s['slope']:+.2f}/period, Kendall tau {s['kendall_tau']:+.2f}"
    elif s["slope"] <= -tcfg["slope_threshold"] and s["kendall_tau"] <= -tcfg["kendall_tau_threshold"]:
        label, reason = "declining", f"slope {s['slope']:+.2f}/period, Kendall tau {s['kendall_tau']:+.2f}"
    elif appear / n >= tcfg["recurring_ratio"]:
        label, reason = "consistently_recurring", f"appeared in {appear} of {n} periods"
    else:
        label, reason = "stable", f"appeared in {appear} of {n} periods; no significant slope"
    trend_score = float(np.clip(s["slope"] * n / (s["mean"] * n + 1.0), -1, 1)) if n else 0.0
    return {**s, "trend": label, "trend_reason": reason, "trend_score": round(trend_score, 3)}


def classify_all(table, summary, tcfg):
    out = []
    for _, row in summary.iterrows():
        g = table[(table["subject"] == row["subject"]) & (table["topic"] == row["topic"])].sort_values("period_index")
        out.append({"subject": row["subject"], "topic": row["topic"], **classify_series(g["questions"].tolist(), tcfg)})
    return summary.merge(pd.DataFrame(out), on=["subject", "topic"])
