from src.trend_analysis.repeated import conflicting_terms, find_pairs, question_families
from src.preprocessing.cleaning import dedup_key

TEXTS = [
    ("Explain the k-means clustering algorithm.", 2022, "Summer"),
    ("Explain the k-means clustering algorithm.", 2023, "Winter"),          # exact repeat
    ("Explain the K-means clustering algorithm with example.", 2024, "Summer"),  # near-identical/rephrased
    ("Discuss how the K-Means method groups data points into clusters.", 2025, "Winter"),  # rephrased
    ("Define CNN.", 2022, "Winter"),
    ("Define RNN.", 2024, "Winter"),
    ("State Bayes theorem.", 2023, "Summer"),
]


def _df(factory):
    return factory([{"cleaned_question_text": t, "year": y, "exam_session": s} for t, y, s in TEXTS])


def test_relationship_types(questions_factory, embedder, cfg):
    df = _df(questions_factory)
    emb = embedder.encode(df["cleaned_question_text"].tolist())
    pairs = find_pairs(df, emb, {**cfg["similarity"], "repeated_question_threshold": 0.7})
    rel = {(a, b): t for a, b, t in zip(pairs["question_id"], pairs["similar_question_id"], pairs["relationship_type"])}
    assert rel[("q0", "q1")] == "exact"
    assert rel.get(("q0", "q2")) in ("near_identical", "rephrased")
    assert rel.get(("q4", "q5"), "none") not in ("exact", "near_identical", "rephrased")   # CNN vs RNN
    assert ("q0", "q6") not in rel                                                        # unrelated


def test_same_paper_pairs_excluded(questions_factory, embedder, cfg):
    df = questions_factory([{"cleaned_question_text": "Explain SVM.", "year": 2023, "exam_session": "Winter"},
                            {"cleaned_question_text": "Explain SVM.", "year": 2023, "exam_session": "Winter"}])
    pairs = find_pairs(df, embedder.encode(df["cleaned_question_text"].tolist()), cfg["similarity"])
    assert len(pairs) == 0   # an OR alternative in the same paper is not a repetition over time


def test_families_group_repeats(questions_factory, embedder, cfg):
    df = _df(questions_factory)
    pairs = find_pairs(df, embedder.encode(df["cleaned_question_text"].tolist()), cfg["similarity"])
    fams, fam_ids = question_families(df, pairs)
    assert fam_ids.iloc[0] == fam_ids.iloc[1] != ""
    assert fams.iloc[0]["papers"] >= 2


def test_conflict_guard():
    assert conflicting_terms(dedup_key("Define CNN"), dedup_key("Define RNN"))
    assert conflicting_terms(dedup_key("advantages of KNN"), dedup_key("disadvantages of KNN"))
    assert not conflicting_terms(dedup_key("Explain k-means"), dedup_key("Describe k-means clustering"))
