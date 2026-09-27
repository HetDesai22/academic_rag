import pytest

from src.retrieval.retriever import Retriever, TfidfRetriever
from src.retrieval.vector_store import VectorStore

ROWS = [
    {"cleaned_question_text": "Explain the backpropagation algorithm with an example.", "year": 2022, "exam_session": "Summer",
     "primary_topic": "Neural Network", "subtopic": "Learning Process and Backpropagation"},
    {"cleaned_question_text": "Explain k-means clustering with a suitable example.", "year": 2023, "exam_session": "Winter",
     "primary_topic": "Unsupervised Learning", "subtopic": "K-Means and K-Medoids Clustering"},
    {"cleaned_question_text": "State Bayes theorem and explain prior and posterior probability.", "year": 2024,
     "exam_session": "Summer", "subject": "Applied Machine Learning", "subject_code": "3171617",
     "primary_topic": "Bayesian Concept Learning", "subtopic": "Bayes Theorem"},
    {"cleaned_question_text": "What is a support vector machine? Explain maximum margin.", "year": 2025, "exam_session": "Winter",
     "primary_topic": "Supervised Learning: Classification and Regression", "subtopic": "Support Vector Machine"},
]


@pytest.fixture
def store(tmp_path, embedder, log, questions_factory):
    df = questions_factory(ROWS)
    s = VectorStore(tmp_path / "vdb")
    s.build_or_update(df, embedder, log)
    return s, df


def test_semantic_search_ranks_relevant_first(store, embedder, cfg):
    s, _ = store
    r = Retriever(cfg, embedder, s)
    hits = r.retrieve("How are neural network weights learned from output errors?", top_k=2)
    assert hits[0]["primary_topic"] == "Neural Network"
    assert hits[0]["similarity"] >= hits[1]["similarity"]
    assert {"question_text", "subject", "year", "similarity", "source_file"} <= set(hits[0])


def test_metadata_filters(store, embedder, cfg):
    s, _ = store
    r = Retriever(cfg, embedder, s)
    assert all(h["subject"] == "Applied Machine Learning"
               for h in r.retrieve("probability", filters={"subject": ["Applied Machine Learning"]}))
    assert all(2023 <= h["year"] <= 2024 for h in r.retrieve("algorithm", filters={"year_range": (2023, 2024)}))
    assert [h["subtopic"] for h in r.retrieve("x", filters={"subtopic": "Bayes Theorem"})] == ["Bayes Theorem"]
    assert r.retrieve("anything", filters={"subject": ["Nonexistent"]}) == []
    assert r.retrieve("") == []


def test_persistence_and_incremental_update(store, embedder, log, questions_factory):
    s, df = store
    loaded = VectorStore(s.dir).load()
    assert loaded.index.ntotal == len(df)
    more = questions_factory(ROWS + [{"cleaned_question_text": "Explain random forest.", "year": 2026,
                                      "exam_session": "Summer", "question_id": "new1"}])
    stats = VectorStore(s.dir).build_or_update(more, embedder, log)
    assert stats == {"mode": "incremental", "added": 1, "total": 5}
    changed = more.copy()
    changed.loc[0, "cleaned_question_text"] = "Edited question text."
    assert VectorStore(s.dir).build_or_update(changed, embedder, log)["mode"] == "rebuild"


def test_tfidf_baseline(store):
    s, _ = store
    hits = TfidfRetriever(s.meta).retrieve("k-means clustering", top_k=1)
    assert "k-means" in hits[0]["question_text"].lower()
