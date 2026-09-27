"""FAISS vector database with filterable metadata and incremental updates.

vector_db/
  index.faiss       IndexFlatIP over L2-normalised question embeddings (cosine similarity)
  metadata.jsonl    one record per vector, row i <-> FAISS id i
  manifest.json     model, dimension, question ids + text hashes, build history

Filtering (subject / year / period / topic / subtopic / marks / ...) is applied *inside* the
FAISS search with an ID selector, so top-k is computed only over matching questions.
"""
import json
from pathlib import Path

import numpy as np

from src.utils.common import now_iso, sha256_text

META_FIELDS = ["question_id", "paper_id", "subject", "subject_code", "year", "exam_session", "period",
               "period_index", "question_number", "sub_part", "is_or_option", "marks", "question_text",
               "raw_question_text", "primary_topic", "subtopic", "secondary_topics", "topic_confidence",
               "review_required", "near_duplicate_group", "source_file"]


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    try:
        import pandas as pd
        if v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return None
    except (ImportError, TypeError):
        pass
    return v


def records_from_df(df):
    d = df.rename(columns={"cleaned_question_text": "question_text"})
    cols = [c for c in META_FIELDS if c in d.columns]
    recs = []
    for r in d[cols].to_dict("records"):
        r = {k: _clean(v) for k, v in r.items()}
        if isinstance(r.get("secondary_topics"), str):
            try:
                r["secondary_topics"] = json.loads(r["secondary_topics"])
            except json.JSONDecodeError:
                r["secondary_topics"] = []
        recs.append(r)
    return recs


class VectorStore:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.index = None
        self.meta = []
        self.manifest = {}

    # ---------- persistence ----------
    @property
    def exists(self):
        return all((self.dir / f).exists() for f in ("index.faiss", "metadata.jsonl", "manifest.json"))

    def load(self):
        import faiss
        if not self.exists:
            raise FileNotFoundError(f"No vector database in {self.dir}; run `python pipeline.py` first")
        self.index = faiss.read_index(str(self.dir / "index.faiss"))
        self.meta = [json.loads(l) for l in (self.dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l]
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.meta):
            raise RuntimeError(f"Vector DB inconsistent: {self.index.ntotal} vectors vs {len(self.meta)} metadata rows")
        return self

    def _save(self):
        import faiss
        self.dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.dir / "index.faiss"))
        with open(self.dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    # ---------- build / incremental update ----------
    def build_or_update(self, df, embedder, log):
        """Add only new questions when possible; rebuild if existing texts changed or were removed."""
        import faiss
        records = records_from_df(df)
        hashes = {r["question_id"]: sha256_text(r["question_text"] or "") for r in records}
        prev = {}
        if self.exists:
            try:
                self.load()
                prev = self.manifest
            except Exception as e:
                log.warning("Existing vector DB unreadable (%s); rebuilding", e)
        same_model = prev.get("model") == embedder.model_name
        prev_hashes = dict(zip(prev.get("question_ids", []), prev.get("text_hashes", [])))
        unchanged_prefix = same_model and prev_hashes and all(hashes.get(q) == h for q, h in prev_hashes.items())

        if unchanged_prefix:
            order = list(prev_hashes)                      # keep existing FAISS ids stable
            new_ids = [r["question_id"] for r in records if r["question_id"] not in prev_hashes]
            by_id = {r["question_id"]: r for r in records}
            if new_ids:
                vecs = embedder.encode([by_id[q]["question_text"] for q in new_ids])
                self.index.add(vecs)
            order += new_ids
            self.meta = [by_id[q] for q in order]          # metadata (labels) always refreshed
            mode = "incremental" if new_ids else "metadata_refresh"
            added = len(new_ids)
        else:
            if prev:
                log.info("  vector DB: model changed or existing questions edited/removed -> full rebuild")
            vecs = embedder.encode([r["question_text"] for r in records])
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
            self.meta = records
            order = [r["question_id"] for r in records]
            mode, added = "rebuild", len(records)
        history = (prev.get("history", []) if unchanged_prefix else []) + [
            {"at": now_iso(), "mode": mode, "added": added, "total": len(self.meta)}]
        self.manifest = {
            "model": embedder.model_name, "dimension": int(self.index.d), "count": len(self.meta),
            "index_type": "IndexFlatIP (cosine on normalised vectors)",
            "question_ids": order, "text_hashes": [hashes[q] for q in order], "history": history[-20:],
        }
        self._save()
        log.info("  vector DB %s: %d added, %d total -> %s", mode, added, len(self.meta), self.dir)
        return {"mode": mode, "added": added, "total": len(self.meta)}

    # ---------- search ----------
    def matching_ids(self, filters):
        if not filters:
            return None
        return np.array([i for i, m in enumerate(self.meta) if _match(m, filters)], dtype="int64")

    def search(self, qvec, k=10, filters=None, exclude_ids=()):
        import faiss
        ids = self.matching_ids(filters)
        if ids is not None and len(ids) == 0:
            return []
        exclude = set(exclude_ids)
        k_eff = min(self.index.ntotal, k + len(exclude))
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids)) if ids is not None else None
        D, I = self.index.search(np.asarray(qvec, dtype="float32").reshape(1, -1), k_eff, params=params)
        out = []
        for score, i in zip(D[0], I[0]):
            if i < 0:
                continue
            m = self.meta[i]
            if m["question_id"] in exclude:
                continue
            out.append({**m, "similarity": float(score)})
            if len(out) == k:
                break
        return out

    def vectors(self):
        return self.index.reconstruct_n(0, self.index.ntotal)


def _match(m, filters):
    for key, want in filters.items():
        if want in (None, "", [], "All"):
            continue
        field = {"topic": "primary_topic"}.get(key, key)
        val = m.get(field)
        if key == "year_range":
            lo, hi = want
            if val is None and m.get("year") is None:
                return False
            y = m.get("year")
            if y is None or not (lo <= y <= hi):
                return False
            continue
        if key == "topic" and isinstance(want, str):
            # topic filter also matches secondary topics
            if val != want and want not in (m.get("secondary_topics") or []):
                return False
            continue
        if isinstance(want, (list, tuple, set)):
            if val not in want:
                return False
        elif val != want:
            return False
    return True
