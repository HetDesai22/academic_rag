"""Retrieval-Augmented Generation over the historical question bank.

User query -> query understanding (filters, intent, topics) -> query embedding -> filtered FAISS
retrieval -> context (retrieved questions [E#] + computed statistics [S#] + model forecasts [F#])
-> LLM (configurable provider) -> citation/grounding checks -> answer with inspectable sources.

Grounding safeguards:
  * insufficient evidence (no hits / best similarity below threshold) -> explicit statement, no LLM call
  * citations are validated against the context; unknown tags are removed and reported
  * years / exam periods mentioned in the answer must appear in the context, else flagged
  * without an available LLM, an extractive answer is assembled from the same evidence
"""
import json
import re

import pandas as pd

from src.rag.prompts import SYSTEM_PROMPT, user_prompt
from src.rag.query_understanding import QueryAnalyzer
from src.utils.llm import LLMUnavailable, get_llm

TAG = re.compile(r"\[(?P<tags>[ESF]\d+(?:\s*,\s*[ESF]\d+)*)\]")
INSUFFICIENT = re.compile(r"(does not|doesn't) contain (enough|sufficient|any) (evidence|questions|data)|insufficient evidence", re.I)


class RAGPipeline:
    def __init__(self, cfg, retriever, taxonomy, trend_df=None, forecast_df=None, log=None, llm=None):
        self.cfg = cfg
        self.retriever = retriever
        self.log = log
        self.trends = trend_df if trend_df is not None else pd.DataFrame()
        self.forecasts = forecast_df if forecast_df is not None else pd.DataFrame()
        meta = retriever.store.meta
        periods = sorted({m["period"] for m in meta if m.get("period")}, key=lambda p: (p[1:], p[0]))
        years = {m["year"] for m in meta if m.get("year") is not None}
        self.scope = {"subjects": sorted({m["subject"] for m in meta}), "periods": periods}
        self.analyzer = QueryAnalyzer(taxonomy, retriever.embedder, periods, years)
        self.llm = llm or get_llm(cfg["llm"], log, cache_path=cfg["paths"]["cache_dir"] / "rag_llm_cache.jsonl")
        self.rcfg = cfg["rag"]

    # ------------------------------------------------------------------ context
    def _stat_rows(self, analysis, hits):
        if self.trends.empty or not self.rcfg.get("include_statistics", True):
            return []
        keys = [(t["subject"], t["topic"]) for t in analysis["topics"]]
        for h in hits[:6]:
            k = (h["subject"], h["primary_topic"])
            if k not in keys:
                keys.append(k)
        idx = self.trends.set_index(["subject", "topic"])
        return [(k, idx.loc[k]) for k in keys[:4] if k in idx.index]

    def build_context(self, query, analysis, hits):
        lines = [f"Dataset scope: subjects {', '.join(self.scope['subjects'])}; exam periods "
                 f"{self.scope['periods'][0]}..{self.scope['periods'][-1]} ({len(self.scope['periods'])} periods, one paper per subject per period)."]
        if analysis["filters"]:
            lines.append(f"Filters applied from the question: {json.dumps(analysis['filters'], default=str)}")
        lines.append("\nRETRIEVED HISTORICAL QUESTIONS (ranked by semantic similarity):")
        evidence = []
        for i, h in enumerate(hits, 1):
            tag = f"E{i}"
            part = f"Q{h.get('question_number', '?')}({h.get('sub_part', '')})"
            lines.append(f"[{tag}] {h['period']} | {h['subject']} | {part}, {h.get('marks')} marks | "
                         f"topic: {h['primary_topic']} / {h['subtopic']} | similarity {h['similarity']:.2f}\n"
                         f"     \"{h['question_text']}\"")
            evidence.append({**h, "tag": tag})
        stats = []
        stat_rows = self._stat_rows(analysis, hits)
        subj = analysis["filters"].get("subject")
        if analysis["intent"] in ("forecast", "trend") and subj and not analysis["topics"] and not self.trends.empty:
            # Subject-level trend/forecast question: evidence = the subject's topic statistics,
            # not whichever questions happen to be nearest to a generic query.
            idx = self.trends.set_index(["subject", "topic"])
            if analysis["intent"] == "forecast" and not self.forecasts.empty:
                keys = list(self.forecasts[self.forecasts["subject"].isin(subj)]
                            .nlargest(5, "forecast_score")[["subject", "topic"]].itertuples(index=False, name=None))
            else:
                t = self.trends[self.trends["subject"].isin(subj)]
                keys = list(t.nlargest(5, "total_questions")[["subject", "topic"]].itertuples(index=False, name=None))
            stat_rows = [(k, idx.loc[k]) for k in keys if k in idx.index]
        if stat_rows:
            lines.append("\nCOMPUTED STATISTICS (exact counts from the full labelled dataset, not only the retrieved items):")
            for j, (k, r) in enumerate(stat_rows, 1):
                tag = f"S{j}"
                counts = json.loads(r["counts_by_period"])
                lines.append(f"[{tag}] {k[0]} / topic '{k[1]}': {int(r['total_questions'])} questions in total "
                             f"({r['share_of_subject']:.1%} of the subject's questions), appeared in "
                             f"{int(r['periods_appeared'])} of {int(r['n_periods'])} exams; per exam: "
                             + ", ".join(f"{p}={c}" for p, c in counts.items())
                             + f"; trend label: {r['trend']} ({r['trend_reason']}).")
                stats.append({"tag": tag, "subject": k[0], "topic": k[1], "total_questions": int(r["total_questions"]),
                              "trend": r["trend"], "counts_by_period": counts})
        forecasts = []
        if analysis["intent"] == "forecast" and self.rcfg.get("include_forecast", True) and not self.forecasts.empty:
            fidx = self.forecasts.set_index(["subject", "topic"])
            keys = [(s["subject"], s["topic"]) for s in stats]
            if not keys and analysis["filters"].get("subject"):
                keys = list(self.forecasts[self.forecasts["subject"].isin(analysis["filters"]["subject"])]
                            .nlargest(5, "forecast_score")[["subject", "topic"]].itertuples(index=False, name=None))
            if keys:
                lines.append("\nMODEL FORECASTS (statistical estimates, NOT historical facts, NOT guarantees):")
            for j, k in enumerate(keys[:5], 1):
                if k not in fidx.index:
                    continue
                r = fidx.loc[k]
                tag = f"F{j}"
                lines.append(f"[{tag}] {k[0]} / '{k[1]}' for {r['forecast_period']}: estimated {r['predicted_questions']:.2f} "
                             f"question(s), forecast score {r['forecast_score']:.2f}, confidence {r['forecast_confidence']} "
                             f"(method {r['method']}, trend {r['trend']}).")
                forecasts.append({"tag": tag, "subject": k[0], "topic": k[1], **{c: r[c] for c in
                                  ("forecast_period", "predicted_questions", "forecast_score", "forecast_confidence", "method")}})
        return "\n".join(lines), evidence, stats, forecasts

    # ------------------------------------------------------------------ answer
    def answer(self, query, filters=None, top_k=None, provider=None):
        query = (query or "").strip()
        if not query:
            return {"answer": "Please enter a question.", "evidence": [], "insufficient": True, "warnings": []}
        analysis = self.analyzer.analyze(query)
        f = {**analysis["filters"], **{k: v for k, v in (filters or {}).items() if v not in (None, "", [], "All")}}
        analysis["filters"] = f
        k = min(top_k or self.cfg["retrieval"]["top_k"], self.rcfg["max_context_questions"])
        hits = self.retriever.retrieve(query, top_k=k, filters=f)
        best = max((h["similarity"] for h in hits), default=0.0)
        context, evidence, stats, forecasts = self.build_context(query, analysis, hits)
        result = {"query": query, "analysis": analysis, "evidence": evidence, "statistics": stats,
                  "forecasts": forecasts, "context": context, "best_similarity": best, "warnings": []}
        threshold = self.cfg["retrieval"]["insufficient_evidence_similarity"]
        # Subject-level trend/forecast questions are answered from computed statistics, so the
        # question-similarity gate does not apply when such statistics exist.
        stats_question = analysis["intent"] in ("forecast", "trend") and bool(stats)
        if not hits or (best < threshold and not stats_question):
            result.update(insufficient=True, provider="none", model="", grounded=True, answer=(
                "**Insufficient evidence.** The historical dataset does not contain questions closely matching this "
                f"query (best similarity {best:.2f} < {threshold:.2f}"
                + (f", with filters {json.dumps(f, default=str)}" if f else "") + "). "
                f"The dataset covers {', '.join(self.scope['subjects'])} for exam periods "
                f"{self.scope['periods'][0]}–{self.scope['periods'][-1]}. The closest retrieved items are listed below for inspection."))
            return result

        llm = self.llm if provider is None else get_llm(self.cfg["llm"], self.log, provider=provider,
                                                          cache_path=self.cfg["paths"]["cache_dir"] / "rag_llm_cache.jsonl")
        text, used_provider, model = None, "extractive", ""
        if llm.available:
            try:
                out = llm.generate(SYSTEM_PROMPT, user_prompt(query, context))
                text, used_provider, model = out["text"], llm.name, out["model"]
            except LLMUnavailable as e:
                result["warnings"].append(f"LLM unavailable ({e}); showing an extractive answer instead.")
        else:
            result["warnings"].append(f"No LLM available ({llm.reason}); showing an extractive answer.")
        if text is None:
            text = self.extractive_answer(analysis, evidence, stats, forecasts)
        text, check = self.check_grounding(text, context, evidence, stats, forecasts)
        result["warnings"] += check.pop("grounding_warnings")
        # The generator may itself conclude that the retrieved context does not answer the query.
        llm_refused = bool(INSUFFICIENT.search(text[:400]))
        result.update(answer=text, provider=used_provider, model=model, insufficient=llm_refused,
                      insufficient_source="generator" if llm_refused else "", **check)
        return result

    @staticmethod
    def extractive_answer(analysis, evidence, stats, forecasts):
        parts = ["**Historical evidence**"]
        for e in evidence[:6]:
            parts.append(f"- {e['period']} · {e['subject']} · {e['primary_topic']}: \"{e['question_text']}\" [{e['tag']}]")
        if stats:
            parts.append("\n**Observed trend**")
            for s in stats:
                parts.append(f"- {s['topic']} ({s['subject']}): {s['total_questions']} questions in the dataset; "
                             f"trend label '{s['trend']}' [{s['tag']}]")
        if forecasts:
            parts.append("\n**Forecast / inference** (model estimate, not a guarantee)")
            for fc in forecasts:
                parts.append(f"- {fc['topic']}: forecast score {fc['forecast_score']:.2f} for {fc['forecast_period']}, "
                             f"confidence {fc['forecast_confidence']} [{fc['tag']}]")
        return "\n".join(parts)

    @staticmethod
    def check_grounding(text, context, evidence, stats, forecasts):
        valid = {e["tag"] for e in evidence} | {s["tag"] for s in stats} | {f["tag"] for f in forecasts}
        used, invalid = [], []

        def fix(m):
            tags = [t.strip() for t in m.group("tags").split(",")]
            good = [t for t in tags if t in valid]
            invalid.extend(t for t in tags if t not in valid)
            used.extend(good)
            return f"[{', '.join(good)}]" if good else ""
        text = TAG.sub(fix, text)
        warnings = []
        if invalid:
            warnings.append(f"Removed citations not present in the context: {sorted(set(invalid))}")
        ctx_years = set(re.findall(r"\b20\d\d\b", context))
        bad_years = sorted(set(re.findall(r"\b20\d\d\b", text)) - ctx_years)
        if bad_years:
            warnings.append(f"Answer mentions years not present in the retrieved context: {bad_years}")
        ctx_periods = set(re.findall(r"\b[SW]20\d\d\b", context))
        bad_periods = sorted(set(re.findall(r"\b[SW]20\d\d\b", text)) - ctx_periods)
        if bad_periods:
            warnings.append(f"Answer mentions exam periods not present in the context: {bad_periods}")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", text) if len(s.split()) >= 6 and not s.startswith("**")]
        cited = [s for s in sentences if TAG.search(s)]
        return text, {"citations_used": sorted(set(used), key=lambda t: (t[0], int(t[1:]))),
                      "citation_coverage": round(len(cited) / len(sentences), 3) if sentences else 0.0,
                      "evidence_cited_ratio": round(len({u for u in used if u.startswith("E")}) / len(evidence), 3) if evidence else 0.0,
                      "grounded": not (invalid or bad_years or bad_periods) and bool(used),
                      "grounding_warnings": warnings}
