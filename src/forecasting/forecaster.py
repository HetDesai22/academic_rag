"""Topic relevance forecasting with rolling-origin backtesting and automatic method selection.

Target: number of questions on each (subject, topic) in the next exam period.
1. Backtest every method with rolling origins: fit on periods < t, predict period t, for
   t = min_train_periods .. last. Forecasts are never scored on data used to fit them.
2. Select the method with the lowest backtest error (config forecasting.selection_metric);
   the pooled Random Forest is only a candidate when enough observations exist.
3. Refit on all periods and forecast the next period.
Outputs are framed as model estimates from historical patterns - never as certain.
"""
import numpy as np
import pandas as pd

from src.forecasting.models import STAT_METHODS, PooledRandomForest

DISCLAIMER = ("Forecasts are statistical estimates based on observed historical patterns in this dataset. "
              "They are not predictions of actual exam content and carry uncertainty.")


def _series(table):
    periods = table[["period", "period_index"]].drop_duplicates().sort_values("period_index")
    series, meta = {}, {}
    for (subject, topic), g in table.groupby(["subject", "topic"]):
        g = g.sort_values("period_index")
        series[(subject, topic)] = g["questions"].to_numpy(dtype=float)
        meta[(subject, topic)] = {"periods": g["period"].tolist(), "seasons": [p[0] for p in g["period"]]}
    return series, meta, periods


def _next_period(last):
    season, year = last[0], int(last[1:])
    return f"W{year}" if season == "S" else f"S{year + 1}"


def _predict(method, h, seasons, target_season, fcfg, rf=None, share=0.0):
    if method == "random_forest":
        return rf.predict_one(h, seasons, target_season, share)
    fn = STAT_METHODS[method]
    return fn(h, window=fcfg["moving_average_window"], alpha=fcfg.get("exp_smoothing_alpha"),
              seasons=seasons, target_season=target_season)


def _share(series, key, upto):
    tot = sum(v[:upto].sum() for k, v in series.items() if k[0] == key[0])
    return series[key][:upto].sum() / tot if tot else 0.0


def backtest(table, fcfg, seed=42, log=None):
    series, meta, _ = _series(table)
    n_periods = max(len(s) for s in series.values())
    methods = list(STAT_METHODS)
    seasons_by_key = {k: m["seasons"] for k, m in meta.items()}
    X_all, _ = PooledRandomForest.training_set(series, seasons_by_key, n_periods)
    use_rf = len(X_all) >= fcfg["ml_min_observations"]
    if use_rf:
        methods.append("random_forest")
    elif log:
        log.info("  random forest skipped: %d pooled observations < %d", len(X_all), fcfg["ml_min_observations"])
    rows = []
    for t in range(fcfg["min_train_periods"], n_periods):
        rf = None
        if use_rf:
            X, y = PooledRandomForest.training_set(series, seasons_by_key, t)
            rf = PooledRandomForest(seed).fit(X, y)
        for key, s in series.items():
            if len(s) <= t:
                continue
            seasons = meta[key]["seasons"]
            for m in methods:
                pred = _predict(m, s[:t], seasons[:t], seasons[t], fcfg, rf, _share(series, key, t))
                rows.append({"method": m, "subject": key[0], "topic": key[1], "origin_period": meta[key]["periods"][t],
                             "train_periods": t, "actual": s[t], "predicted": round(pred, 4)})
    bt = pd.DataFrame(rows)
    return bt, methods


def backtest_metrics(bt, threshold):
    out = []
    for (m), g in bt.groupby("method"):
        err = g["predicted"] - g["actual"]
        pred_app, act_app = g["predicted"] >= threshold, g["actual"] > 0
        tp = int((pred_app & act_app).sum())
        prec = tp / max(1, int(pred_app.sum()))
        rec = tp / max(1, int(act_app.sum()))
        out.append({"method": m, "n_forecasts": int(len(g)), "mae": float(err.abs().mean()),
                    "rmse": float(np.sqrt((err ** 2).mean())),
                    "appearance_precision": prec, "appearance_recall": rec,
                    "appearance_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                    "rank_spearman_mean": float(g.groupby(["origin_period", "subject"]).apply(
                        lambda x: x["predicted"].rank().corr(x["actual"].rank(), method="pearson")
                        if x["actual"].nunique() > 1 and x["predicted"].nunique() > 1 else np.nan).mean())})
    return pd.DataFrame(out).sort_values("mae").reset_index(drop=True)


def forecast(table, trend_summary, fcfg, seed=42, log=None):
    series, meta, periods = _series(table)
    bt, methods = backtest(table, fcfg, seed, log)
    metrics = backtest_metrics(bt, fcfg["appearance_threshold"])
    metric = fcfg.get("selection_metric", "mae")
    best = metrics.sort_values(metric, ascending=metric not in ("appearance_f1",)).iloc[0]["method"]
    baseline = metrics.set_index("method").loc["historical_mean"]
    if log:
        log.info("  backtest (%d forecasts/method): %s", int(metrics["n_forecasts"].iloc[0]),
                 ", ".join(f"{r.method} MAE={r.mae:.3f}" for r in metrics.itertuples()))
        log.info("  selected method: %s", best)

    rf = None
    n = max(len(s) for s in series.values())
    if best == "random_forest":
        X, y = PooledRandomForest.training_set(series, {k: m["seasons"] for k, m in meta.items()}, n)
        rf = PooledRandomForest(seed).fit(X, y)
    series_err = bt[bt["method"] == best].assign(ae=lambda d: (d["predicted"] - d["actual"]).abs()) \
        .groupby(["subject", "topic"])["ae"].mean()
    ts = trend_summary.set_index(["subject", "topic"])
    rows = []
    for key, s in series.items():
        seasons, plist = meta[key]["seasons"], meta[key]["periods"]
        nxt = _next_period(plist[-1])
        short = len(s) < fcfg["minimum_periods"]
        method = "historical_mean" if short else best
        pred = _predict(method, s, seasons, nxt[0], fcfg, rf, _share(series, key, len(s)))
        tr = ts.loc[key] if key in ts.index else {}
        ratio = float((s > 0).mean())
        err = float(series_err.get(key, np.nan))
        # Backtest error relative to the topic's typical volume (an error of 1 question matters
        # more for a topic averaging 1 question per exam than for one averaging 6).
        rel_err = err / max(float(s.mean()), 1.0) if not np.isnan(err) else np.nan
        if short or ratio < 0.3 or (not np.isnan(rel_err) and rel_err > 0.6):
            conf = "Low"
        elif ratio >= 0.7 and (np.isnan(rel_err) or rel_err <= 0.3):
            conf = "High"
        else:
            conf = "Medium"
        appeared = [p for p, v in zip(plist, s) if v > 0]
        score = 1 - np.exp(-pred)  # P(>=1 question) if counts were Poisson with mean `pred`
        rows.append({
            "subject": key[0], "topic": key[1], "forecast_period": nxt,
            "historical_frequency": int(s.sum()), "recent_frequency": int(s[-3:].sum()),
            "periods_observed": len(s), "periods_appeared": len(appeared),
            "trend": tr.get("trend", ""), "trend_score": float(tr.get("trend_score", 0.0)),
            "predicted_questions": round(pred, 3), "forecast_score": round(float(score), 3),
            "forecast_confidence": conf, "method": method,
            "backtest_mae_topic": None if np.isnan(err) else round(err, 3),
            "backtest_relative_error": None if np.isnan(rel_err) else round(rel_err, 3),
            "supporting_periods": ", ".join(appeared),
            "evidence": (f"Historical evidence: {int(s.sum())} question(s) in {len(appeared)} of {len(s)} exams "
                         f"({', '.join(appeared) or 'none'}); last 3 exams: {int(s[-3:].sum())}. "
                         f"Based on observed patterns ({method}), the model estimates {pred:.2f} question(s) "
                         f"for {nxt}. Forecast confidence: {conf}."),
        })
    fc = pd.DataFrame(rows).sort_values(["subject", "forecast_score"], ascending=[True, False]).reset_index(drop=True)
    summary = {"selected_method": best, "candidate_methods": methods, "selection_metric": metric,
               "metrics": metrics.to_dict("records"),
               "improvement_over_historical_mean_mae": float(baseline["mae"] - metrics.set_index("method").loc[best, "mae"]),
               "backtest_origins": sorted(bt["origin_period"].unique().tolist(),
                                          key=lambda p: (p[1:], p[0])),
               "disclaimer": DISCLAIMER}
    return fc, bt, summary
