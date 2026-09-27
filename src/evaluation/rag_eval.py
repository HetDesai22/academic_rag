"""RAG evaluation.

Deterministic metrics (computed from the pipeline output itself):
  context_keyword_precision  share of retrieved questions containing an expected keyword (lexical proxy)
  citation_coverage          share of answer sentences carrying a citation tag
  citation_validity          no citation outside the context; grounded flag from the grounding checks
  refusal_accuracy           out-of-scope controls answered with "insufficient evidence", in-scope not
Automatic LLM-judge metrics (separate, silver standard): faithfulness, answer relevance and context
relevance on a 1-5 scale, plus a list of unsupported claims. The judge sees only the query, the
context and the answer.
"""
import numpy as np

from src.utils.common import ROOT, read_json

JUDGE_SYSTEM = """You are a strict evaluator of retrieval-augmented answers about historical exam questions.
Judge ONLY against the provided context. Score 1-5:
- faithfulness: every factual claim (questions, years, counts, trends) is supported by the context (5 = fully supported)
- answer_relevance: the answer addresses the user's question (5 = fully)
- context_relevance: the retrieved context is relevant to the question (5 = highly relevant)
List any claims not supported by the context."""

JUDGE_SCHEMA = {"type": "object", "properties": {
    "faithfulness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    "answer_relevance": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    "context_relevance": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    "rationale": {"type": "string"}},
    "required": ["faithfulness", "answer_relevance", "context_relevance", "unsupported_claims", "rationale"],
    "additionalProperties": False}


def _kw_precision(evidence, keywords):
    if not keywords or not evidence:
        return None
    hits = [any(k in (e["question_text"] or "").lower() for k in keywords) for e in evidence]
    return float(np.mean(hits))


def evaluate_rag(rag, cfg, judge, log):
    spec = read_json(ROOT / cfg["evaluation"]["rag_queries"])
    if not spec:
        return {"status": "skipped", "reason": "no RAG evaluation queries file"}
    per_query = []
    for q in spec["queries"]:
        res = rag.answer(q["query"])
        row = {"id": q["id"], "query": q["query"], "provider": res.get("provider"), "model": res.get("model"),
               "insufficient": bool(res.get("insufficient")), "expect_insufficient": q["expect_insufficient"],
               "refusal_correct": bool(res.get("insufficient")) == q["expect_insufficient"],
               "best_similarity": round(res.get("best_similarity", 0.0), 3),
               "n_evidence": len(res.get("evidence", [])),
               "context_keyword_precision": _kw_precision(res.get("evidence", []), q["expected_keywords"]),
               "citation_coverage": res.get("citation_coverage"), "evidence_cited_ratio": res.get("evidence_cited_ratio"),
               "grounded": res.get("grounded"), "warnings": res.get("warnings", []),
               "answer": res.get("answer", "")}
        if judge is not None and getattr(judge, "available", False) and not res.get("insufficient"):
            try:
                out = judge.generate(JUDGE_SYSTEM, f"QUESTION:\n{q['query']}\n\nCONTEXT:\n{res['context']}\n\nANSWER:\n{res['answer']}",
                                     json_schema=JUDGE_SCHEMA, max_tokens=1500)
                row["judge"] = out["json"]
            except Exception as e:
                log.warning("  judge failed for %s: %s", q["id"], e)
        per_query.append(row)

    def mean(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None
    answered = [r for r in per_query if not r["insufficient"]]
    judged = [r for r in answered if "judge" in r]
    report = {
        "n_queries": len(per_query), "generator_provider": answered[0]["provider"] if answered else None,
        "generator_model": answered[0]["model"] if answered else None,
        "deterministic": {
            "refusal_accuracy": mean("refusal_correct", [{**r, "refusal_correct": float(r["refusal_correct"])} for r in per_query]),
            "context_keyword_precision": mean("context_keyword_precision", answered),
            "citation_coverage": mean("citation_coverage", answered),
            "evidence_cited_ratio": mean("evidence_cited_ratio", answered),
            "grounded_rate": mean("grounded", [{**r, "grounded": float(bool(r["grounded"]))} for r in answered]),
            "answers_with_grounding_warnings": sum(1 for r in answered if r["warnings"]),
        },
        "llm_judge": ({
            "type": "AUTOMATIC - LLM judge (silver standard, not human evaluation)",
            "judge_model": getattr(judge, "model", ""), "n_judged": len(judged),
            "faithfulness_mean": round(float(np.mean([r["judge"]["faithfulness"] for r in judged])), 3),
            "answer_relevance_mean": round(float(np.mean([r["judge"]["answer_relevance"] for r in judged])), 3),
            "context_relevance_mean": round(float(np.mean([r["judge"]["context_relevance"] for r in judged])), 3),
            "answers_with_unsupported_claims": sum(1 for r in judged if r["judge"]["unsupported_claims"]),
        } if judged else {"status": "skipped", "reason": "no LLM judge available or no answered queries"}),
        "per_query": per_query,
    }
    log.info("  RAG: refusal accuracy %s, citation coverage %s, grounded rate %s, judge faithfulness %s",
             report["deterministic"]["refusal_accuracy"], report["deterministic"]["citation_coverage"],
             report["deterministic"]["grounded_rate"], report["llm_judge"].get("faithfulness_mean"))
    return report
