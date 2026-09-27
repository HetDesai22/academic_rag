import pandas as pd
import pytest

from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.utils.llm import BaseProvider, NoneProvider

ROWS = [
    {"cleaned_question_text": "Explain the backpropagation algorithm with an example.", "year": 2022, "exam_session": "Summer",
     "primary_topic": "Neural Network", "subtopic": "Learning Process and Backpropagation"},
    {"cleaned_question_text": "What are the limitations of the backpropagation algorithm?", "year": 2024, "exam_session": "Winter",
     "primary_topic": "Neural Network", "subtopic": "Learning Process and Backpropagation"},
    {"cleaned_question_text": "Explain k-means clustering with a suitable example.", "year": 2023, "exam_session": "Winter",
     "primary_topic": "Unsupervised Learning", "subtopic": "K-Means and K-Medoids Clustering"},
]
TAXONOMY = {"subjects": {"Machine Learning": {"subject_code": "3170724", "topics": {
    "Neural Network": {"description": "neurons, backpropagation, deep learning", "subtopics": {}},
    "Unsupervised Learning": {"description": "clustering, association rules", "subtopics": {}}}}}}


class FakeLLM(BaseProvider):
    name, model = "fake", "fake-1"

    def __init__(self, reply):
        super().__init__(None)
        self.available, self.reply = True, reply

    def _generate(self, system, user, json_schema, max_tokens):
        return {"text": self.reply, "json": None, "model": self.model}


@pytest.fixture
def retriever(tmp_path, embedder, log, cfg, questions_factory):
    s = VectorStore(tmp_path / "vdb")
    s.build_or_update(questions_factory(ROWS), embedder, log)
    return Retriever(cfg, embedder, s)


def rag(cfg, retriever, llm, log):
    return RAGPipeline(cfg, retriever, TAXONOMY, pd.DataFrame(), pd.DataFrame(), log, llm=llm)


def test_grounded_answer_with_valid_citations(cfg, retriever, log):
    llm = FakeLLM("**Historical evidence**\nBackpropagation was asked in S2022 [E1] and again in W2024 [E2].")
    res = rag(cfg, retriever, llm, log).answer("What was asked about backpropagation?")
    assert not res["insufficient"] and res["grounded"]
    assert res["citations_used"] == ["E1", "E2"] and res["warnings"] == []
    assert res["evidence"][0]["tag"] == "E1" and "backpropagation" in res["evidence"][0]["question_text"].lower()


def test_invented_citations_and_years_are_flagged(cfg, retriever, log):
    llm = FakeLLM("Backpropagation appeared in 2019 [E1] and in S2031 [E9]. It will surely appear next year.")
    res = rag(cfg, retriever, llm, log).answer("What was asked about backpropagation?")
    assert "[E9]" not in res["answer"]                          # unknown citation removed
    assert not res["grounded"]
    joined = " ".join(res["warnings"])
    assert "E9" in joined and "2019" in joined and "S2031" in joined


def test_insufficient_evidence_skips_generation(cfg, retriever, log):
    llm = FakeLLM("should not be called")
    res = rag(cfg, retriever, llm, log).answer("How is B+ tree indexing used in database management systems?")
    assert res["insufficient"] and "Insufficient evidence" in res["answer"]
    assert llm.calls == 0


def test_extractive_fallback_without_llm(cfg, retriever, log):
    res = rag(cfg, retriever, NoneProvider(), log).answer("Questions about backpropagation")
    assert res["provider"] == "extractive" and "[E1]" in res["answer"] and res["grounded"]
    assert any("No LLM available" in w for w in res["warnings"])


def test_empty_query(cfg, retriever, log):
    assert rag(cfg, retriever, NoneProvider(), log).answer("   ")["insufficient"]


def test_query_understanding_filters(cfg, retriever, log):
    a = rag(cfg, retriever, NoneProvider(), log).analyzer
    out = a.analyze("What was asked in ML since 2023 in the winter exams?")
    assert out["filters"]["subject"] == ["Machine Learning"]
    assert out["filters"]["year_range"][0] == 2023 and out["filters"]["exam_session"] == "Winter"
    assert a.analyze("Which topics are likely in the next exam?")["intent"] == "forecast"
