import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.common import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def log():
    lg = logging.getLogger("aqi-tests")
    lg.addHandler(logging.NullHandler())
    return lg


@pytest.fixture(scope="session")
def embedder(cfg, log):
    from src.embeddings.embedder import Embedder
    return Embedder(cfg, log)


@pytest.fixture(scope="session")
def taxonomy_src():
    from src.topic_modeling.taxonomy import load_source
    return load_source()


def make_questions(rows):
    """Small labelled frame in the pipeline's internal schema."""
    base = {"subject": "Machine Learning", "subject_code": "3170724", "exam_session": "Winter", "year": 2023,
            "marks": 7, "primary_topic": "Neural Network", "subtopic": "Deep Learning", "secondary_topics": "[]",
            "topic_confidence": 0.9, "review_required": False, "near_duplicate_group": "", "question_number": 1,
            "sub_part": "a", "source_file": "x.pdf"}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["period"] = df["exam_session"].str[0] + df["year"].astype(str)
    df["period_index"] = df["year"] * 2 + df["exam_session"].map({"Summer": 0, "Winter": 1})
    df["paper_id"] = df["subject_code"] + "_" + df["period"]
    default_ids = pd.Series([f"q{i}" for i in range(len(df))], index=df.index)
    df["question_id"] = df["question_id"].fillna(default_ids) if "question_id" in df else default_ids
    df["raw_question_text"] = df["cleaned_question_text"]
    return df


@pytest.fixture
def questions_factory():
    return make_questions
