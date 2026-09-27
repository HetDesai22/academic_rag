import pytest

from src.trend_analysis.frequency import topic_period_table, topic_summary
from src.trend_analysis.trends import classify_series

TCFG = {"min_appearances": 3, "slope_threshold": 0.10, "kendall_tau_threshold": 0.30, "recurring_ratio": 0.80,
        "recent_periods": 3, "emerging_max_early_total": 1, "emerging_min_recent_appearances": 2,
        "dormant_min_absent_periods": 3}


@pytest.mark.parametrize("series,label", [
    ([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], "increasing"),
    ([5, 4, 4, 3, 3, 2, 2, 1, 1, 1], "declining"),
    ([0, 0, 1, 0, 0, 0, 0, 0, 0, 1], "sporadic"),
    ([0, 0, 0, 0, 0, 0, 1, 0, 2, 1], "emerging"),
    ([2, 1, 2, 1, 2, 0, 1, 0, 0, 0], "dormant"),
    ([2, 3, 2, 3, 2, 3, 2, 3, 2, 3], "consistently_recurring"),
])
def test_trend_rules(series, label):
    assert classify_series(series, TCFG)["trend"] == label


def test_single_appearance_is_never_a_trend():
    assert classify_series([0, 0, 0, 0, 0, 0, 0, 0, 0, 6], TCFG)["trend"] == "sporadic"


def test_frequency_table_zero_fills_and_counts(questions_factory):
    rows = [{"cleaned_question_text": "a", "year": 2022, "exam_session": "Summer", "primary_topic": "T1", "marks": 3},
            {"cleaned_question_text": "b", "year": 2022, "exam_session": "Summer", "primary_topic": "T1", "marks": 4},
            {"cleaned_question_text": "c", "year": 2022, "exam_session": "Summer", "primary_topic": "T2", "marks": 7,
             "secondary_topics": '["T1"]'},
            {"cleaned_question_text": "d", "year": 2023, "exam_session": "Winter", "primary_topic": "T2", "marks": 7}]
    df = questions_factory(rows)
    t = topic_period_table(df)
    t1 = t[t["topic"] == "T1"].set_index("period")
    assert t1.loc["S2022", "questions"] == 2 and t1.loc["W2023", "questions"] == 0   # zero-filled
    assert t1.loc["S2022", "secondary"] == 1 and t1.loc["S2022", "mentions"] == 3
    assert t1.loc["S2022", "share"] == pytest.approx(2 / 3, abs=1e-3)
    s = topic_summary(df, t).set_index("topic")
    assert s.loc["T1", "total_questions"] == 2 and s.loc["T1", "periods_appeared"] == 1
    assert s.loc["T2", "supporting_periods"] == "S2022, W2023"


def test_empty_topic_series_safe():
    r = classify_series([0, 0, 0], TCFG)
    assert r["trend"] == "sporadic" and r["slope"] == 0.0
