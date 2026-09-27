import pandas as pd
import pytest

from src.ingestion.loader import detect_columns, standardize


def test_detects_alternative_column_names(log):
    raw = pd.DataFrame({"Question": ["Explain SVM."], "Course": ["ML"], "Exam_Year": [2023], "Mark": [7],
                        "Filename": ["a.pdf"]})
    m = detect_columns(raw)
    assert m["question_text"] == "Question" and m["subject"] == "Course"
    assert m["year"] == "Exam_Year" and m["marks"] == "Mark" and m["source_file"] == "Filename"
    assert m["page_number"] is None and m["semester"] is None


def test_missing_required_column_raises():
    with pytest.raises(KeyError):
        detect_columns(pd.DataFrame({"text_body": ["x"], "subject": ["ML"]}))


def test_missing_optional_metadata_is_not_invented(log):
    raw = pd.DataFrame({"question": ["Explain SVM.", "Define PCA."], "subject": ["ML", "ML"]})
    std = standardize(raw, detect_columns(raw), log)
    assert "page_number" not in std and "year" not in std and "marks" not in std
    assert std["question_id"].tolist() == ["row_00000", "row_00001"]   # row-based fallback id


def test_ids_derived_from_metadata(log):
    raw = pd.DataFrame({"question": ["A", "B"], "subject": ["ML", "ML"], "subject_code": [3170724, 3170724],
                        "exam_session": ["Winter", "Summer"], "year": [2023, 2024], "question_no": [1, 2],
                        "sub_part": ["a", "c"], "is_or_option": [False, True]})
    std = standardize(raw, detect_columns(raw), log)
    assert std["paper_id"].tolist() == ["3170724_W2023", "3170724_S2024"]
    assert std["question_id"].tolist() == ["3170724_W2023_Q1a", "3170724_S2024_Q2c_OR"]
    assert std["period"].tolist() == ["W2023", "S2024"]
    assert std["period_index"].iloc[0] < std["period_index"].iloc[1]   # W2023 before S2024


def test_missing_year_and_duplicate_ids_are_handled(log):
    raw = pd.DataFrame({"question_id": ["x", "x", "y"], "question": ["A", "A2", "B"], "subject": ["ML"] * 3,
                        "year": [2023, None, "bad"]})
    std = standardize(raw, detect_columns(raw), log)
    assert std["question_id"].tolist() == ["x", "y"]          # duplicate id dropped (first kept, logged)
    assert std["year"].isna().sum() == 1                      # invalid year -> missing, not guessed
