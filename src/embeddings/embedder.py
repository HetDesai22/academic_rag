"""Contextual sentence embeddings with a persistent per-text cache.

Each text is keyed by sha256(model | mode | text). Only texts never seen before are encoded,
so adding new papers later embeds just the new questions. Vectors are L2-normalised float32
(cosine similarity == inner product). Stored under models/embedding_cache/<model>/ (not CSV).
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def _slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class Embedder:
    def __init__(self, cfg, log, device=None):
        ecfg = cfg["embedding"]
        self.model_name = ecfg["model"]
        self.batch_size = ecfg.get("batch_size", 64)
        self.normalize = ecfg.get("normalize", True)
        self.query_prefix = ecfg.get("query_prefix") or ""
        self.log = log
        self.device = device or self._pick_device(ecfg.get("device", "auto"))
        self._model = None
        self.dir = Path(cfg["paths"]["embedding_cache_dir"]) / _slug(self.model_name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._keys_path, self._vec_path = self.dir / "keys.json", self.dir / "vectors.npy"
        self._index, self._vectors = {}, None
        if self._keys_path.exists() and self._vec_path.exists():
            keys = json.loads(self._keys_path.read_text(encoding="utf-8"))
            vecs = np.load(self._vec_path)
            if len(keys) == len(vecs):
                self._index = {k: i for i, k in enumerate(keys)}
                self._vectors = vecs
            else:
                log.warning("Embedding cache is inconsistent (%d keys, %d vectors); rebuilding", len(keys), len(vecs))
        self.encoded_count = 0

    @staticmethod
    def _pick_device(pref):
        if pref in ("cuda", "cpu"):
            return pref
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                if self.device == "cuda":  # e.g. CUDA runtime problem -> CPU fallback
                    self.log.warning("Could not load model on CUDA (%s); falling back to CPU", e)
                    self.device = "cpu"
                    self._model = SentenceTransformer(self.model_name, device="cpu")
                else:
                    raise
            self.log.info("Embedding model %s loaded on %s (dim %d)", self.model_name, self.device,
                          self.dim)
        return self._model

    @property
    def dim(self):
        if self._vectors is not None and len(self._vectors):
            return self._vectors.shape[1]
        m = self.model
        return m.get_embedding_dimension() if hasattr(m, "get_embedding_dimension") else m.get_sentence_embedding_dimension()

    def _key(self, text, query):
        return hashlib.sha256(f"{self.model_name}|{'q' if query else 'd'}|{text}".encode("utf-8")).hexdigest()

    def encode(self, texts, query=False, show_progress=None, persist=True):
        """Embed texts (documents by default; query=True adds the retrieval instruction)."""
        texts = ["" if t is None else str(t) for t in texts]
        keys = [self._key(t, query) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in self._index))
        if missing:
            todo = {k: t for k, t in zip(keys, texts) if k in set(missing)}
            batch = [(self.query_prefix + todo[k]) if query else todo[k] for k in missing]
            progress = show_progress if show_progress is not None else len(batch) > self.batch_size
            vecs = self.model.encode(batch, batch_size=self.batch_size, normalize_embeddings=self.normalize,
                                     show_progress_bar=progress, convert_to_numpy=True).astype(np.float32)
            start = 0 if self._vectors is None else len(self._vectors)
            self._vectors = vecs if self._vectors is None else np.vstack([self._vectors, vecs])
            for i, k in enumerate(missing):
                self._index[k] = start + i
            self.encoded_count += len(missing)
            if persist and len(missing) > 0:
                self.save()
            if len(missing) > 5:
                self.log.info("  encoded %d new texts (%s); %d served from cache", len(missing),
                              "query" if query else "document", len(texts) - len(missing))
        return self._vectors[[self._index[k] for k in keys]] if texts else np.zeros((0, self.dim), np.float32)

    def encode_one(self, text, query=True):
        return self.encode([text], query=query, persist=False)[0]

    def save(self):
        keys = [None] * len(self._index)
        for k, i in self._index.items():
            keys[i] = k
        np.save(self._vec_path, self._vectors)
        self._keys_path.write_text(json.dumps(keys), encoding="utf-8")
