"""Semantic retrieval: query -> embedding -> filtered FAISS search -> ranked sources.

Two modes:
  "query"    natural-language information need (uses the model's retrieval instruction)
  "similar"  question-to-question similarity (symmetric; used for similar / repeated questions)
A TF-IDF lexical retriever is included as a baseline for evaluation only.
"""
from src.retrieval.vector_store import VectorStore

RESULT_FIELDS = ["question_id", "question_text", "subject", "year", "period", "exam_session", "question_number",
                 "sub_part", "marks", "primary_topic", "subtopic", "secondary_topics", "paper_id", "source_file",
                 "similarity", "topic_confidence", "review_required"]


class Retriever:
    def __init__(self, cfg, embedder, store=None):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store or VectorStore(cfg["paths"]["vector_db_dir"]).load()
        rcfg = cfg["retrieval"]
        self.top_k = rcfg.get("top_k", 10)
        self.min_similarity = rcfg.get("min_similarity", 0.0)

    def retrieve(self, text, top_k=None, filters=None, mode="query", exclude_ids=()):
        if not text or not str(text).strip():
            return []
        vec = self.embedder.encode_one(str(text).strip(), query=(mode == "query"))
        hits = self.store.search(vec, k=top_k or self.top_k, filters=filters, exclude_ids=exclude_ids)
        return [{k: h.get(k) for k in RESULT_FIELDS} for h in hits if h["similarity"] >= self.min_similarity]


class TfidfRetriever:
    """Lexical baseline (evaluation only)."""

    def __init__(self, metadata):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.meta = metadata
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.X = self.vec.fit_transform([m["question_text"] or "" for m in metadata])

    def retrieve(self, text, top_k=10, exclude_ids=()):
        import numpy as np
        s = (self.X @ self.vec.transform([text]).T).toarray().ravel()
        exclude = set(exclude_ids)
        out = []
        for i in np.argsort(-s):
            if self.meta[i]["question_id"] in exclude:
                continue
            out.append({**self.meta[i], "similarity": float(s[i])})
            if len(out) == top_k:
                break
        return out
