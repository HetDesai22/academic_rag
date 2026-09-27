import numpy as np
import pandas as pd
import pytest

from src.forecasting.forecaster import _next_period, backtest, forecast
from src.forecasting.models import exp_smoothing, historical_mean, linear_trend, moving_average, seasonal_mean

FCFG = {"minimum_periods": 4, "min_train_periods": 4, "ml_min_observations": 10_000, "moving_average_window": 3,
        "exp_smoothing_alpha": None, "appearance_threshold": 0.5, "selection_metric": "mae"}
PERIODS = ["W2021", "S2022", "W2022", "S2023", "W2023", "S2024", "W2024", "S2025", "W2025", "S2026"]


def table(series_by_topic):
    rows = []
    for topic, s in series_by_topic.items():
        for i, (p, v) in enumerate(zip(PERIODS, s)):
            rows.append({"subject": "ML", "topic": topic, "period": p, "period_index": i, "questions": v})
    return pd.DataFrame(rows)


def test_basic_methods():
    h = np.array([1, 2, 3, 4], dtype=float)
    assert historical_mean(h) == 2.5
    assert moving_average(h, window=2) == 3.5
    assert linear_trend(h) == pytest.approx(5.0)
    assert 1 <= exp_smoothing(h) <= 4
    assert seasonal_mean([1, 5, 1, 5], seasons=list("WSWS"), target_season="S") == 5
    assert linear_trend(np.array([5, 3, 1, 0, 0], dtype=float)) >= 0      # never negative


def test_backtest_uses_only_past_data():
    t = table({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "B": [2] * 10})
    bt, methods = backtest(t, FCFG)
    assert "random_forest" not in methods           # too few observations for the ML model
    row = bt[(bt["method"] == "historical_mean") & (bt["topic"] == "A") & (bt["origin_period"] == "S2024")].iloc[0]
    assert row["train_periods"] == 5 and row["predicted"] == pytest.approx(np.mean([1, 2, 3, 4, 5]))
    assert row["actual"] == 6


def test_forecast_output_and_selection():
    t = table({"Rising": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5], "Flat": [2] * 10, "Rare": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]})
    summary_rows = pd.DataFrame({"subject": ["ML"] * 3, "topic": ["Rising", "Flat", "Rare"],
                                 "trend": ["increasing", "consistently_recurring", "sporadic"], "trend_score": [0.5, 0, 0]})
    fc, bt, summ = forecast(t, summary_rows, FCFG)
    assert set(["topic", "historical_frequency", "recent_frequency", "trend_score", "forecast_score",
                "forecast_confidence", "supporting_periods", "evidence"]) <= set(fc.columns)
    assert fc["forecast_score"].between(0, 1).all()
    assert (fc["forecast_period"] == "W2026").all()
    by = fc.set_index("topic")
    assert by.loc["Rare", "forecast_confidence"] == "Low"
    assert by.loc["Flat", "forecast_confidence"] == "High"
    assert by.loc["Flat", "predicted_questions"] == pytest.approx(2.0)
    assert "not" in summ["disclaimer"].lower()
    assert summ["selected_method"] in summ["candidate_methods"]


def test_next_period():
    assert _next_period("S2026") == "W2026" and _next_period("W2025") == "S2026"
