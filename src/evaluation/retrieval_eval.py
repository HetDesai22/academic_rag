"""Retrieval evaluation: Precision@K, Recall@K, MRR, Hit Rate@K.

Two semi-automatic test sets (no manual relevance judgements exist yet):
  A. repeat_set      - query = a question that was asked again in another paper; relevant = its
                       other occurrences (exact / near-identical groups found lexically, i.e.
                       independently of the embedding model). Lexical ground truth favours the
                       TF-IDF baseline, which is reported alongside.
  B. paraphrase_set  - an LLM rewrites sampled questions as student-style search queries with
                       different wording; relevant = the source question and its exact repeats.
                       Automatically generated (silver standard); stored for reproducibility.
"""
import json
import random

import numpy as np

from src.retrieval.retriever import TfidfRetriever
from src.utils.common import read_json, write_json


def metrics_at_k(ranked_lists, relevant_sets, ks):
    out = {}
    rr = []
    for ranked, rel in zip(ranked_lists, relevant_sets):
        rank = next((i + 1 for i, q in enumerate(ranked) if q in rel), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr"] = float(np.mean(rr)) if rr else 0.0
    for k in ks:
        p, r, h = [], [], []
        for ranked, rel in zip(ranked_lists, relevant_sets):
            top = ranked[:k]
            n_rel = len(set(top) & rel)
            p.append(n_rel / k)
            r.append(n_rel / len(rel) if rel else 0.0)
            h.append(1.0 if n_rel else 0.0)
        out[f"precision@{k}"], out[f"recall@{k}"], out[f"hit_rate@{k}"] = map(lambda x: float(np.mean(x)), (p, r, h))
    out["n_queries"] = len(ranked_lists)
    return out


def repeat_set(df, fuzzy_threshold=85):
    # Lexical-only groups (fuzzy ratio + content-word guard, no embeddings) so the ground truth is
    # independent of the embedding model being evaluated.
    from src.preprocessing.cleaning import mark_duplicates
    lexical, _ = mark_duplicates(df[["question_id", "paper_id", "cleaned_question_text"]].copy(),
                                 "cleaned_question_text", fuzzy_threshold)
    groups = lexical[lexical["near_duplicate_group"].astype(str).str.startswith(("ND", "NX"))]
    items = []
    for _, g in groups.groupby("near_duplicate_group"):
        if g["paper_id"].nunique() < 2:
            continue
        q = g.iloc[0]
        rel = set(g.loc[g["paper_id"] != q["paper_id"], "question_id"])
        if rel:
            items.append({"query_id": q["question_id"], "query": q["cleaned_question_text"], "relevant": sorted(rel)})
    return items


def paraphrase_set(df, cfg, llm, log):
    path = cfg["paths"]["evaluation_dir"] / "retrieval_paraphrase_set.json"
    existing = read_json(path)
    if existing:
        return existing["items"], existing.get("generator", "")
    if not getattr(llm, "available", False):
        log.warning("  paraphrase set skipped: no LLM available to generate queries")
        return [], ""
    rng = random.Random(cfg["random_seed"])
    pool = df.drop_duplicates(["subject", "cleaned_question_text"])
    pool = pool[pool["cleaned_question_text"].str.split().str.len() >= 4]
    sample = pool.iloc[rng.sample(range(len(pool)), min(cfg["evaluation"]["paraphrase_queries"], len(pool)))]
    schema = {"type": "object", "properties": {"queries": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "query": {"type": "string"}},
        "required": ["id", "query"], "additionalProperties": False}}}, "required": ["queries"], "additionalProperties": False}
    items = []
    for start in range(0, len(sample), 15):
        chunk = sample.iloc[start:start + 15]
        listing = "\n".join(f"{r.question_id}: {r.cleaned_question_text}" for r in chunk.itertuples())
        out = llm.generate(
            "You write realistic search queries a student would type to find past exam questions. "
            "Rewrite each question as a short natural-language query about the same concept using DIFFERENT wording "
            "(avoid copying the key phrase verbatim where a synonym or description works). One query per id.",
            listing, json_schema=schema, max_tokens=3000)
        by_id = {q["id"]: q["query"] for q in out["json"]["queries"]}
        for r in chunk.itertuples():
            if r.question_id not in by_id:
                continue
            exact = df[(df["exact_duplicate_group"] != "") & (df["exact_duplicate_group"] == r.exact_duplicate_group)]["question_id"]
            items.append({"query_id": r.question_id, "source_question": r.cleaned_question_text,
                          "query": by_id[r.question_id], "relevant": sorted({r.question_id, *exact})})
    write_json({"description": "LLM-generated paraphrase queries (silver standard). Relevant = source question + exact repeats.",
                "generator": getattr(llm, "model", ""), "items": items}, path)
    return items, getattr(llm, "model", "")


def evaluate_retrieval(df, retriever, cfg, llm, log):
    ks = cfg["evaluation"]["retrieval_k"]
    kmax = max(ks)
    tfidf = TfidfRetriever(retriever.store.meta)
    report = {"k_values": ks, "note": "Semi-automatic test sets; no manual relevance judgements."}

    rep = repeat_set(df)
    sem = [[h["question_id"] for h in retriever.retrieve(it["query"], kmax, mode="similar", exclude_ids=[it["query_id"]])] for it in rep]
    lex = [[h["question_id"] for h in tfidf.retrieve(it["query"], kmax, exclude_ids=[it["query_id"]])] for it in rep]
    rel = [set(it["relevant"]) for it in rep]
    report["repeat_set"] = {"description": "Query = a question asked again in another paper; relevant = its other occurrences "
                                           "(lexical duplicate groups). Query itself excluded.",
                            "semantic_faiss": metrics_at_k(sem, rel, ks), "tfidf_baseline": metrics_at_k(lex, rel, ks)}

    para, gen = paraphrase_set(df, cfg, llm, log)
    if para:
        sem = [[h["question_id"] for h in retriever.retrieve(it["query"], kmax, mode="query")] for it in para]
        lex = [[h["question_id"] for h in tfidf.retrieve(it["query"], kmax)] for it in para]
        rel = [set(it["relevant"]) for it in para]
        report["paraphrase_set"] = {"description": "LLM-rewritten student queries with different wording (silver standard).",
                                    "generator": gen, "semantic_faiss": metrics_at_k(sem, rel, ks),
                                    "tfidf_baseline": metrics_at_k(lex, rel, ks),
                                    "examples": [{"query": it["query"], "source": it["source_question"]} for it in para[:5]]}
    else:
        report["paraphrase_set"] = {"status": "skipped", "reason": "no LLM available to generate paraphrase queries"}
    for name in ("repeat_set", "paraphrase_set"):
        r = report[name]
        if "semantic_faiss" in r:
            log.info("  retrieval %-14s semantic MRR=%.3f Hit@5=%.3f | TF-IDF MRR=%.3f Hit@5=%.3f (n=%d)", name,
                     r["semantic_faiss"]["mrr"], r["semantic_faiss"].get("hit_rate@5", 0),
                     r["tfidf_baseline"]["mrr"], r["tfidf_baseline"].get("hit_rate@5", 0), r["semantic_faiss"]["n_queries"])
    return report
