"""Repeated / rephrased question detection.

Pairs of questions from different papers are compared with contextual embeddings (cosine)
and a lexical ratio, then typed:
  exact                 identical after normalising case/punctuation
  near_identical        only spelling/filler-word differences, and lexical ratio >= near_identical_fuzzy
                        or cosine >= rephrased_threshold
  rephrased             cosine >= rephrased_threshold with different wording (same ask) and no
                        conflicting key terms
  semantically_similar  cosine >= repeated_question_threshold (related, not necessarily the same ask)
A conflict guard rejects look-alike pairs that ask different things (CNN vs RNN, supervised vs
unsupervised, advantages vs disadvantages): such pairs are at most "semantically_similar".
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.preprocessing.cleaning import _content_tokens, _negation_pair, dedup_key, is_variant


def conflicting_terms(key_a, key_b):
    a, b = _content_tokens(key_a), _content_tokens(key_b)
    only_a, only_b = a - b, b - a
    for x in only_a:
        for y in only_b:
            if _negation_pair(x, y):
                return True
            if len(x) <= 5 and len(y) <= 5 and x != y and fuzz.ratio(x, y) >= 60:  # CNN/RNN, KNN/ANN
                return True
    return False


def find_pairs(df, emb, scfg):
    keys = df["cleaned_question_text"].map(dedup_key).tolist()
    papers = df["paper_id"].tolist()
    sims = emb @ emb.T
    n = len(df)
    thr = scfg["repeated_question_threshold"]
    rows = []
    for i in range(n):
        cand = np.where(sims[i, i + 1:] >= thr)[0] + i + 1
        for j in cand:
            if papers[i] == papers[j]:
                continue  # an OR alternative in the same paper is not a repetition over time
            cos = float(sims[i, j])
            lex = fuzz.ratio(keys[i], keys[j])
            conflict = conflicting_terms(keys[i], keys[j])
            if keys[i] == keys[j]:
                rel = "exact"
            elif is_variant(keys[i], keys[j]) and (lex >= scfg["near_identical_fuzzy"] or cos >= scfg["rephrased_threshold"]):
                rel = "near_identical"
            elif cos >= scfg["rephrased_threshold"] and not conflict:
                rel = "rephrased"
            else:
                rel = "semantically_similar"
            rows.append((i, j, cos, lex, rel, conflict))
    cols = ["i", "j", "similarity_score", "lexical_similarity", "relationship_type", "term_conflict"]
    pairs = pd.DataFrame(rows, columns=cols)
    if pairs.empty:
        return pairs
    # Keep the strongest `max_pairs_per_question` partners per question.
    k = scfg.get("max_pairs_per_question", 10)
    pairs = pairs.sort_values("similarity_score", ascending=False)
    keep = pairs.groupby("i").head(k).index.union(pairs.groupby("j").head(k).index)
    pairs = pairs.loc[keep].sort_values(["i", "similarity_score"], ascending=[True, False])
    a, b = df.iloc[pairs["i"]].reset_index(drop=True), df.iloc[pairs["j"]].reset_index(drop=True)
    out = pd.DataFrame({
        "question_id": a["question_id"], "similar_question_id": b["question_id"],
        "similarity_score": pairs["similarity_score"].round(4).values,
        "lexical_similarity": pairs["lexical_similarity"].round(1).values,
        "relationship_type": pairs["relationship_type"].values,
        "term_conflict": pairs["term_conflict"].values,
        "subject": a["subject"], "similar_subject": b["subject"],
        "period": a["period"], "similar_period": b["period"],
        "same_subject": (a["subject"] == b["subject"]).values,
        "topic": a["primary_topic"], "similar_topic": b["primary_topic"],
        "question_text": a["cleaned_question_text"], "similar_question_text": b["cleaned_question_text"],
    })
    return out


def question_families(df, pairs, types=("exact", "near_identical", "rephrased")):
    """Connected components of strong repeat relations -> how often the 'same' question recurs."""
    ids = df["question_id"].tolist()
    pos = {q: i for i, q in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    strong = pairs[pairs["relationship_type"].isin(types)] if len(pairs) else pairs
    for q1, q2 in zip(strong.get("question_id", []), strong.get("similar_question_id", [])):
        ra, rb = find(pos[q1]), find(pos[q2])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    fam = pd.Series([find(pos[q]) for q in ids], index=df.index)
    d = df.assign(_fam=fam)
    grp = d.groupby("_fam").agg(
        times_asked=("question_id", "size"), papers=("paper_id", "nunique"),
        subjects=("subject", lambda s: "; ".join(sorted(set(s)))),
        periods=("period", lambda s: ", ".join(sorted(set(s), key=lambda p: (p[1:], p[0])))),
        topic=("primary_topic", lambda s: s.mode().iat[0]),
        example=("cleaned_question_text", "first"),
        question_ids=("question_id", lambda s: ";".join(s)))
    grp = grp[grp["papers"] > 1].sort_values(["papers", "times_asked"], ascending=False).reset_index(drop=True)
    grp.insert(0, "family_id", [f"F{i:04d}" for i in range(len(grp))])
    family_of = {q: f for f, qs in zip(grp["family_id"], grp["question_ids"]) for q in qs.split(";")}
    return grp, df["question_id"].map(family_of).fillna("")
