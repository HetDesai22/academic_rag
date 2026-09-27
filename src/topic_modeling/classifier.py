"""Semantic topic / subtopic classification with confidence and multi-label support.

For each subject separately:
  question embedding  x  subtopic prototype embeddings  -> cosine similarity
  + small bonus when a subtopic keyword appears literally (configurable, not the main signal)
  subtopic score   = best prototype score (description, plus human-verified exemplar questions)
  topic score      = best subtopic score within the topic
  primary_topic    = argmax topic; subtopic = argmax subtopic inside that topic
  topic_confidence = softmax(topic scores / temperature)[primary]
Secondary topics come from classifying sentence-level segments of multi-part questions.
"""
import re

import numpy as np

from src.topic_modeling.taxonomy import flatten

SEGMENT_SPLIT = re.compile(
    r"(?<=[?.])\s+(?=[A-Z])"                      # sentence boundaries
    r"|\s+(?:\(?(?:i{1,3}|iv|v|vi)\)|\(?[a-e]\))\s+"  # i) ii) / a) b) sub-labels
    r"|\s+(?:Also|also)\s+"
)
MIN_SEGMENT_WORDS = 3


def split_segments(text):
    parts = [p.strip(" .?,;:") for p in SEGMENT_SPLIT.split(text)]
    parts = [p for p in parts if p and len(p.split()) >= MIN_SEGMENT_WORDS]
    return parts if len(parts) > 1 else []


def _norm(text):
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip() + " "


def keyword_matches(text, subtopics):
    """Return {subtopic_index: longest matched keyword}, keeping only the most specific matches.

    A match is dropped when its keyword is contained in a longer keyword matched for another
    subtopic, e.g. "gradient descent" loses to "stochastic gradient descent".
    """
    t = _norm(text)
    hits = {}
    for i, r in enumerate(subtopics):
        found = [kw for kw in r["keywords"] if _norm(kw) in t]
        if found:
            hits[i] = max(found, key=len)
    specific = {}
    for i, kw in hits.items():
        k = _norm(kw)
        if not any(j != i and k != _norm(other) and k in _norm(other) for j, other in hits.items()):
            specific[i] = kw
    return specific


def _softmax(x, temperature):
    z = (x - x.max()) / temperature
    e = np.exp(z)
    return e / e.sum()


class SubjectClassifier:
    """Classifier for one subject's taxonomy.

    exemplars: optional list of (topic, subtopic, question_text) verified by a human; each is
    added as an extra prototype of its subtopic.
    """

    def __init__(self, subject, taxonomy_src, embedder, cfg, exemplars=None):
        self.subject = subject
        self.subs = [r for r in flatten(taxonomy_src) if r["subject"] == subject]
        if not self.subs:
            raise KeyError(f"No taxonomy for subject '{subject}'")
        self.topics = list(dict.fromkeys(r["topic"] for r in self.subs))
        self.topic_of = np.array([self.topics.index(r["topic"]) for r in self.subs])
        sub_index = {(r["topic"], r["subtopic"]): i for i, r in enumerate(self.subs)}
        proto_texts = [r["text"] for r in self.subs]
        proto_owner = list(range(len(self.subs)))
        self.n_exemplars = 0
        for topic, subtopic, text in exemplars or []:
            if (topic, subtopic) in sub_index and text:
                proto_texts.append(text)
                proto_owner.append(sub_index[(topic, subtopic)])
                self.n_exemplars += 1
        self.proto_owner = np.array(proto_owner)
        # Taxonomy descriptions are passages; exemplar questions are embedded the same way.
        self.proto_emb = embedder.encode(proto_texts)
        self.embedder = embedder
        ccfg = cfg["classifier"]
        self.kw_bonus = ccfg["keyword_bonus"]
        self.temp = ccfg["temperature"]
        self.sec_min = ccfg["secondary_min_confidence"]
        self.sec_max = ccfg["secondary_max"]
        self.min_similarity = ccfg["min_similarity"]

    def _scores(self, texts, q_emb):
        proto_sims = q_emb @ self.proto_emb.T
        sims = np.full((len(texts), len(self.subs)), -1.0, dtype=np.float32)
        for j in range(len(self.subs)):
            sims[:, j] = proto_sims[:, self.proto_owner == j].max(1)
        bonus = np.zeros_like(sims)
        self._last_hits = []
        for i, t in enumerate(texts):
            hits = keyword_matches(t, self.subs)
            self._last_hits.append(hits)
            for j in hits:
                bonus[i, j] = self.kw_bonus
        return sims, sims + bonus

    def _decide(self, sim_row, score_row):
        topic_scores = np.array([score_row[self.topic_of == k].max() for k in range(len(self.topics))])
        probs = _softmax(topic_scores, self.temp)
        order = np.argsort(-probs)
        t = int(order[0])
        in_topic = np.where(self.topic_of == t)[0]
        sub_probs = _softmax(score_row[in_topic], self.temp)
        s_local = int(sub_probs.argmax())
        s = int(in_topic[s_local])
        conf = float(probs[t])
        # A question far from every subtopic is likely off-syllabus: cap confidence so it is
        # never auto-accepted, however clear the margin between topics is.
        below_floor = float(sim_row[in_topic].max()) < self.min_similarity
        if below_floor:
            conf = min(conf, float(sim_row[in_topic].max()))
        sub_order = in_topic[np.argsort(-score_row[in_topic])]
        return {
            "topic_idx": t,
            "primary_topic": self.topics[t],
            "subtopic": self.subs[s]["subtopic"],
            "subtopic_idx": s,
            "below_similarity_floor": below_floor,
            "topic_confidence": conf,
            "subtopic_confidence": float(sub_probs[s_local]),
            "similarity": float(sim_row[s]),
            "keyword_match": bool(score_row[s] > sim_row[s]),
            "top_topics": [self.topics[int(i)] for i in order[:3]],
            "runner_up_topic": self.topics[int(order[1])] if len(order) > 1 else "",
            "runner_up_confidence": float(probs[order[1]]) if len(order) > 1 else 0.0,
            "alternatives": [f"{self.topics[int(i)]} ({probs[int(i)]:.2f})" for i in order[1:3]]
                            + [f"{self.topics[t]} / {self.subs[int(j)]['subtopic']}" for j in sub_order[1:2]],
            "topic_probs": probs,
        }

    def classify(self, texts):
        q_emb = self.embedder.encode(texts, query=True)
        sims, scores = self._scores(texts, q_emb)
        results = [self._decide(sims[i], scores[i]) for i in range(len(texts))]

        # Multi-label: classify each segment of multi-part questions.
        seg_texts, seg_owner = [], []
        for i, t in enumerate(texts):
            for seg in split_segments(t):
                seg_texts.append(seg)
                seg_owner.append(i)
        for r in results:
            r["secondary_topics"] = []
            r["segments"] = 0
        if seg_texts:
            s_emb = self.embedder.encode(seg_texts, query=True)
            s_sims, s_scores = self._scores(seg_texts, s_emb)
            for j, i in enumerate(seg_owner):
                res = results[i]
                res["segments"] += 1
                seg = self._decide(s_sims[j], s_scores[j])
                # A secondary topic needs concrete evidence in the segment itself: a confident
                # topic AND a literal keyword of that subtopic. This rejects generic follow-ups
                # like "What are the limitations of this algorithm?".
                has_keyword = seg["subtopic_idx"] in self._last_hits[j]
                if (seg["primary_topic"] != res["primary_topic"]
                        and seg["topic_confidence"] >= self.sec_min
                        and not seg["below_similarity_floor"]
                        and has_keyword
                        and seg["primary_topic"] not in res["secondary_topics"]
                        and len(res["secondary_topics"]) < self.sec_max):
                    res["secondary_topics"].append(seg["primary_topic"])
        for r in results:
            for k in ("topic_probs", "topic_idx", "subtopic_idx"):
                r.pop(k)
        return results


def confidence_band(conf, thresholds):
    if conf >= thresholds["high"]:
        return "high"
    if conf < thresholds["low"]:
        return "low"
    return "medium"
