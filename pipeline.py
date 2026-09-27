"""Academic Question Intelligence - end-to-end pipeline.

    python pipeline.py                 # run everything (unchanged stages are skipped)
    python pipeline.py --force         # recompute every stage
    python pipeline.py --skip-eval     # everything except the evaluation stage
    python pipeline.py --stage evaluation   # (re)run a single stage and its report outputs

Stages: 1 profiling · 2 cleaning · 3 schema normalisation · 4 topic discovery · 5 topic
classification · 6 validation · 7 embeddings · 8 vector DB · 9 trend analysis · 10 repeated
questions · 11 forecasting · 12 evaluation · 13 reports.
The original dataset is only read. New papers: drop CSVs into data/incoming/ (same columns), or
extract PDFs with `python -m src.ingestion.extract_questions --only-new`.
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from src.embeddings.embedder import Embedder
from src.evaluation import classification_eval, classification_reference, rag_eval, retrieval_eval
from src.evaluation.report import evaluation_html
from src.forecasting.forecaster import forecast
from src.ingestion.loader import load_questions
from src.ingestion.profiling import profile, profile_html, summary_lines
from src.preprocessing.cleaning import PREPROCESSING_VERSION, mark_duplicates, preprocess
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.topic_modeling import labeling
from src.topic_modeling.label_validation import labeling_report, validate
from src.topic_modeling.taxonomy import (build_taxonomy_json, discover_clusters, draft_taxonomy_from_clusters,
                                         flatten, load_source, validate_source)
from src.trend_analysis import evolution, frequency, repeated, trends
from src.utils.common import (ROOT, fingerprint, get_logger, load_config, now_iso, read_json, rel, sha256_file,
                              write_json)
from src.utils.llm import get_llm
from src.utils.state import PipelineState
from src.visualization.figures import generate_all

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*n_jobs value.*")

LABEL_COLUMNS = ["primary_topic", "secondary_topics", "subtopic", "topic_confidence", "subtopic_confidence",
                 "similarity", "keyword_match", "confidence_band", "labeling_method", "review_required", "review_reason",
                 "runner_up_topic", "runner_up_confidence", "top_topics", "alternatives", "llm_decision", "llm_topic",
                 "llm_subtopic", "llm_confidence", "below_similarity_floor"]


def file_hash(p):
    return sha256_file(p) if p.exists() else "missing"


def code_hash(*packages):
    """Hash of a stage's source code, so code changes invalidate cached stage outputs."""
    files = sorted(f for pkg in packages for f in (ROOT / "src" / pkg).rglob("*.py"))
    return fingerprint([sha256_file(f) for f in files])


class Pipeline:
    def __init__(self, force=False, skip_eval=False, only=None):
        self.cfg = load_config()
        self.log = get_logger(self.cfg)
        self.state = PipelineState(self.cfg["paths"]["pipeline_state"], force=force)
        self.skip_eval, self.only = skip_eval, only
        self.timings = {}
        np.random.seed(self.cfg["random_seed"])
        for k, p in self.cfg["paths"].items():
            if hasattr(p, "mkdir"):  # create every output folder (globs are plain strings)
                (p if k.endswith("_dir") else p.parent).mkdir(parents=True, exist_ok=True)

    def stage(self, n, name):
        self.log.info("[%2d/13] %s", n, name)
        self._t = time.time()
        self._name = name

    def done(self):
        self.timings[self._name] = round(time.time() - self._t, 2)

    # --------------------------------------------------------------------------------------
    def run(self):
        cfg, log, P = self.cfg, self.log, self.cfg["paths"]
        started = now_iso()
        log.info("=== Academic Question Intelligence pipeline started ===")

        # 1 + 3. Load, schema normalisation, profiling
        self.stage(1, "Data profiling")
        df, meta = load_questions(cfg, log)
        in_fp = fingerprint(meta["input_files"], cfg["columns"], code_hash("ingestion", "visualization"))
        if not self.state.is_fresh("profile", in_fp, [P["profile_report"], P["profile_html"]]):
            rep = profile(df, meta, cfg["preprocessing"]["min_question_chars"])
            write_json(rep, P["profile_report"])
            P["profile_html"].write_text(profile_html(df, rep), encoding="utf-8")
            self.state.mark("profile", in_fp)
        else:
            rep = read_json(P["profile_report"])
            log.info("  unchanged input - profile reused")
        for line in summary_lines(rep):
            log.info("  %s", line)
        self.done()

        prev_labeled = pd.read_csv(P["output_csv"], usecols=["question_id"]) if P["output_csv"].exists() else None
        if prev_labeled is not None:
            new_ids = set(df["question_id"]) - set(prev_labeled["question_id"])
            if new_ids:
                log.info("  incremental: %d new question(s) since the last run: %s", len(new_ids), sorted(new_ids)[:10])

        # 2. Cleaning
        self.stage(2, "Cleaning")
        df = preprocess(df, "question_text", cfg, log)
        df, lex_dups = mark_duplicates(df, "cleaned_question_text", cfg["preprocessing"]["near_duplicate_fuzzy_threshold"])
        clean_cols = [c for c in df.columns if not c.startswith("_")]
        df[clean_cols].to_csv(P["cleaned_csv"], index=False, encoding="utf-8-sig")
        clean_fp = fingerprint(in_fp, PREPROCESSING_VERSION, cfg["preprocessing"])
        self.done()

        # 3. Schema normalisation (done during loading) - report it
        self.stage(3, "Schema normalisation")
        log.info("  mapping: %s", {k: v for k, v in meta["column_mapping"].items() if v})
        log.info("  not available in source (left empty, not invented): %s", meta["missing_fields"] or "none")
        log.info("  derived: %s", meta["derived_fields"] or "none")
        self.done()

        # Embeddings are needed from stage 4 on; the per-text cache makes this cheap after the first run.
        embedder = Embedder(cfg, log)

        # 4. Topic discovery / taxonomy
        self.stage(4, "Topic discovery & taxonomy")
        src = load_source(ROOT / cfg["paths"]["taxonomy_source"])
        tax_warnings = validate_source(src, set(df["subject"]), log)
        llm_small = get_llm(cfg["llm"], log, cache_path=P["cache_dir"] / "aux_llm_cache.jsonl")
        for subject in sorted(set(df["subject"]) - set(src["subjects"])):
            g = df[df["subject"] == subject]
            src["subjects"][subject] = draft_taxonomy_from_clusters(
                subject, g["cleaned_question_text"].tolist(), embedder.encode(g["cleaned_question_text"].tolist()), cfg, log, llm_small)
        rows = flatten(src)
        rows_by_subject = {s: [r for r in rows if r["subject"] == s] for s in src["subjects"]}
        tax_fp = fingerprint(clean_fp, sha256_file(ROOT / cfg["paths"]["taxonomy_source"]), cfg["discovery"], cfg["embedding"]["model"],
                             code_hash("topic_modeling"))
        if cfg["discovery"]["enabled"] and not self.state.is_fresh("discovery", tax_fp, [P["cluster_report"]]):
            disc = {}
            for subject, g in df.groupby("subject"):
                texts = g["cleaned_question_text"].tolist()
                disc[subject] = discover_clusters(embedder.encode(texts, query=True), texts, rows_by_subject[subject],
                                                  embedder.encode([r["text"] for r in rows_by_subject[subject]]), cfg, log)
            write_json(disc, P["cluster_report"])
            self.state.mark("discovery", tax_fp)
        else:
            log.info("  cluster discovery unchanged - reused %s", rel(P["cluster_report"]))
        taxonomy = build_taxonomy_json(src)
        write_json(taxonomy, P["taxonomy_path"])
        log.info("  taxonomy v%s: %d subjects, %d topics, %d subtopics", src["version"], len(src["subjects"]),
                 sum(len(s["topics"]) for s in src["subjects"].values()), len(rows))
        self.done()

        # 5. Topic classification (+ LLM check + human review)
        self.stage(5, "Topic classification")
        prev_review = labeling.load_review(cfg)
        exemplars = labeling.review_exemplars(prev_review, df, cfg)
        df = labeling.classify_all(df, src, embedder, cfg, log, exemplars)
        df, llm_stats = labeling.apply_llm(df, rows_by_subject, cfg, log)
        df.loc[df["flag_empty"], ["review_required", "review_reason"]] = [True, "empty_question"]
        df, applied = labeling.apply_human_review(df, prev_review, taxonomy, log)
        review = labeling.write_review_csv(df, prev_review, cfg)
        log.info("  review file: %d rows (%d pending) -> %s", len(review), int((review["review_status"] == "pending").sum()),
                 rel(P["review_csv"]))
        low = df[df["confidence_band"] == "low"]
        for _, r in low.head(10).iterrows():
            log.info("  low confidence %.2f %s: %s -> %s", r["topic_confidence"], r["question_id"],
                     r["cleaned_question_text"][:60], r["primary_topic"])
        self.done()

        # 6. Validation
        self.stage(6, "Label validation")
        df["secondary_topics"] = df["secondary_topics"].map(lambda x: json.dumps(list(x)))
        checks, errors = validate(df, taxonomy, cfg["confidence_thresholds"])
        for e in errors:
            log.error("  validation failed: %s (%d rows, e.g. %s)", e["check"], e["count"], e["question_ids"][:3])
        log.info("  %d/%d checks passed", sum(v == 0 for v in checks.values()), len(checks))
        self.done()

        # 7. Embeddings
        self.stage(7, "Embeddings")
        doc_emb = embedder.encode(df["cleaned_question_text"].tolist())
        log.info("  %d question embeddings (dim %d, %s, cache %s); %d newly encoded this run",
                 len(doc_emb), doc_emb.shape[1], embedder.device, rel(embedder.dir), embedder.encoded_count)
        df, sem_dups = mark_duplicates(df, "cleaned_question_text", cfg["preprocessing"]["near_duplicate_fuzzy_threshold"],
                                       doc_emb, cfg["preprocessing"]["near_duplicate_semantic_threshold"])
        self.done()

        # 8. Vector DB
        self.stage(8, "Vector database (FAISS)")
        store = VectorStore(P["vector_db_dir"])
        vstats = store.build_or_update(df, embedder, log)
        self.done()

        # 9. Trend analysis
        self.stage(9, "Trend analysis")
        tcfg = cfg["trends"]
        table = frequency.topic_period_table(df, "primary_topic", tcfg["period"])
        sub_table = frequency.topic_period_table(df, "subtopic", tcfg["period"])
        summary = frequency.topic_summary(df, table, tcfg["recent_periods"])
        trend_df = trends.classify_all(table, summary, tcfg)
        trend_df = trend_df.sort_values(["subject", "total_questions"], ascending=[True, False])
        trend_df.to_csv(P["trend_csv"], index=False, encoding="utf-8-sig")
        table.to_csv(P["topic_period_csv"], index=False, encoding="utf-8-sig")
        sub_table.to_csv(P["reports_dir"] / "subtopic_frequency_by_period.csv", index=False, encoding="utf-8-sig")
        dom = evolution.dominant_topics(table)
        dom.to_csv(P["reports_dir"] / "topic_evolution.csv", index=False, encoding="utf-8-sig")
        by_label = {lab: g[["subject", "topic", "total_questions", "trend_reason"]].to_dict("records")
                    for lab, g in trend_df.groupby("trend")}
        write_json({"generated_at": now_iso(), "period_unit": tcfg["period"], "rules": tcfg,
                    "counts": trend_df["trend"].value_counts().to_dict(), "topics_by_trend": by_label,
                    "top_topics_by_subject": {s: g.nlargest(5, "total_questions")[["topic", "total_questions", "trend"]].to_dict("records")
                                              for s, g in trend_df.groupby("subject")}}, P["trend_summary_json"])
        log.info("  %d topic series; trends: %s", len(trend_df), trend_df["trend"].value_counts().to_dict())
        self.done()

        # 10. Repeated questions
        self.stage(10, "Repeated-question detection")
        pairs = repeated.find_pairs(df, doc_emb, cfg["similarity"])
        families, fam_ids = repeated.question_families(df, pairs)
        df["question_family"] = fam_ids
        pairs.to_csv(P["repeated_csv"], index=False, encoding="utf-8-sig")
        families.to_csv(P["reports_dir"] / "question_families.csv", index=False, encoding="utf-8-sig")
        log.info("  %d related pairs %s; %d question families asked in 2+ papers",
                 len(pairs), pairs["relationship_type"].value_counts().to_dict() if len(pairs) else {}, len(families))
        self.done()

        # 11. Forecasting
        self.stage(11, "Forecasting")
        fcfg = cfg["forecasting"]
        fc_fp = fingerprint(table[["subject", "topic", "period", "questions"]].to_csv(index=False), fcfg, cfg["random_seed"],
                            code_hash("forecasting"))
        fc_summary_path = P["reports_dir"] / "forecast_summary.json"
        if self.state.is_fresh("forecast", fc_fp, [P["forecast_csv"], P["forecast_backtest_csv"], fc_summary_path]):
            fc_df, fc_summary = pd.read_csv(P["forecast_csv"]), read_json(fc_summary_path)
            log.info("  topic series unchanged - forecast reused (method %s)", fc_summary["selected_method"])
        else:
            fc_df, bt, fc_summary = forecast(table, trend_df, fcfg, cfg["random_seed"], log)
            fc_df.to_csv(P["forecast_csv"], index=False, encoding="utf-8-sig")
            bt.to_csv(P["forecast_backtest_csv"], index=False, encoding="utf-8-sig")
            write_json(fc_summary, fc_summary_path)
            self.state.mark("forecast", fc_fp)
        self.done()

        # Final labelled outputs (written before evaluation so the app works even if evaluation is skipped)
        out = self.write_outputs(df, meta, llm_stats, checks, errors, vstats, lex_dups, sem_dups, tax_warnings, started, embedder)

        # 12. Evaluation
        self.stage(12, "Evaluation")
        ev = read_json(P["evaluation_report"], {}) or {}
        label_content = out[["question_id", "cleaned_question_text", "primary_topic", "subtopic", "secondary_topics",
                             "review_required", "confidence_band", "labeling_method"]].to_csv(index=False)
        vdb = read_json(P["vector_db_dir"] / "manifest.json", {}) or {}
        eval_fp = fingerprint(label_content, vdb.get("model"), vdb.get("text_hashes"),
                              file_hash(P["validation_sample_csv"]), file_hash(ROOT / cfg["evaluation"]["rag_queries"]),
                              cfg["llm"], cfg["evaluation"], cfg["retrieval"], cfg["rag"], fc_summary.get("metrics"),
                              code_hash("evaluation", "rag", "retrieval", "embeddings", "utils"))
        if self.skip_eval:
            log.info("  skipped (--skip-eval)")
        elif self.state.is_fresh("evaluation", eval_fp, [P["evaluation_report"], P["evaluation_html"]]):
            log.info("  inputs unchanged - evaluation reused")
        else:
            ev = self.evaluate(out, src, taxonomy, store, embedder, trend_df, fc_df, fc_summary)
            self.state.mark("evaluation", eval_fp)
        self.done()

        # 13. Reports & figures
        self.stage(13, "Reports & visualisations")
        freq = frequency.global_topic_frequency(out)
        figs = generate_all(out, table, freq, trend_df, pairs, fc_df, P["figures_dir"])
        if ev:
            P["evaluation_html"].write_text(evaluation_html(ev), encoding="utf-8")
        self.done()
        self.verify_outputs()
        log.info("=== Done in %.1fs: %d questions, %d topics, %d need review; validation %s ===",
                 sum(self.timings.values()), len(out), out["primary_topic"].nunique(), int(out["review_required"].sum()),
                 "PASSED" if all(v == 0 for v in checks.values()) else "FAILED")
        log.info("Stage timings (s): %s", self.timings)
        return 0 if all(v == 0 for v in checks.values()) else 1

    # --------------------------------------------------------------------------------------
    def write_outputs(self, df, meta, llm_stats, checks, errors, vstats, lex_dups, sem_dups, tax_warnings, started, embedder):
        cfg, P = self.cfg, self.cfg["paths"]
        df["taxonomy_version"] = load_source(ROOT / cfg["paths"]["taxonomy_source"])["version"]
        df["embedding_model"] = embedder.model_name
        df["labeled_at"] = started
        drop = {"top_topics", "alternatives", "below_similarity_floor", "segments", "llm_reason", "llm_model", "_input_file"}
        first = ["question_id", "paper_id", "university", "subject", "subject_code", "semester", "year", "exam_session",
                 "period", "period_index", "exam_date", "question_number", "sub_part", "is_or_option", "marks",
                 "raw_question_text", "cleaned_question_text", "primary_topic", "secondary_topics", "subtopic",
                 "topic_confidence", "labeling_method", "review_required"]
        cols = [c for c in first if c in df.columns] + [c for c in df.columns if c not in first and c not in drop]
        out = df[cols].copy()
        for c in ("topic_confidence", "subtopic_confidence", "similarity", "runner_up_confidence"):
            if c in out:
                out[c] = out[c].astype(float).round(4)
        out.to_csv(P["output_csv"], index=False, encoding="utf-8-sig")
        records = json.loads(out.to_json(orient="records", force_ascii=False))
        for r in records:
            r["secondary_topics"] = json.loads(r["secondary_topics"]) if r.get("secondary_topics") else []
        write_json({"generated_at": now_iso(), "count": len(records), "questions": records}, P["output_json"])
        import sentence_transformers
        import torch
        report = labeling_report(out, checks, errors, {
            "run": {"started_at": started, "python": sys.version.split()[0], "random_seed": cfg["random_seed"]},
            "inputs": meta["input_files"], "column_mapping": meta["column_mapping"],
            "versions": {"taxonomy": out["taxonomy_version"].iloc[0], "preprocessing": PREPROCESSING_VERSION,
                         "embedding_model": embedder.model_name, "embedding_device": embedder.device,
                         "sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__},
            "classifier_config": cfg["classifier"], "llm_topic_check": llm_stats, "taxonomy_warnings": tax_warnings,
            "duplicate_detection": {"lexical": lex_dups, "lexical_plus_semantic": sem_dups}, "vector_db": vstats,
        }, cfg["confidence_thresholds"])
        write_json(report, P["labeling_report"])
        # Validation sample for human verification (created once; never overwritten)
        classification_eval.create_sample(out.assign(secondary_topics=out["secondary_topics"]), cfg, self.log)
        return out

    def evaluate(self, out, src, taxonomy, store, embedder, trend_df, fc_df, fc_summary):
        cfg, P, log = self.cfg, self.cfg["paths"], self.log
        judge = get_llm({**cfg["llm"], "provider": "anthropic" if cfg["evaluation"]["llm_judge"]["enabled"] else "none",
                         "anthropic": {**cfg["llm"]["anthropic"], "model": cfg["evaluation"]["llm_judge"]["model"],
                                       "effort": cfg["evaluation"]["llm_judge"]["effort"]}},
                        log, cache_path=P["cache_dir"] / "eval_llm_cache.jsonl")
        ev = {"generated_at": now_iso(), "notes": "Human-verified metrics and automatic (silver) metrics are reported separately."}
        sample = pd.read_csv(P["validation_sample_csv"], dtype=str, encoding="utf-8-sig")
        ev["classification_manual"] = classification_eval.evaluate(sample, taxonomy, log)
        vsample = pd.read_csv(P["validation_sample_csv"], encoding="utf-8-sig")
        labels, model = classification_reference.reference_labels(vsample, src, cfg, judge, log)
        ev["classification_llm_reference"] = classification_reference.agreement(vsample, labels, model)
        if ev["classification_llm_reference"].get("status") == "ok":
            log.info("  classification (silver, LLM reference): topic agreement %.3f (n=%d)",
                     ev["classification_llm_reference"]["all"]["topic_agreement"], ev["classification_llm_reference"]["all"]["n"])
        retriever = Retriever(cfg, embedder, store)
        ev["retrieval"] = retrieval_eval.evaluate_retrieval(out, retriever, cfg, judge, log)
        rag = RAGPipeline(cfg, retriever, taxonomy, trend_df, fc_df, log)
        ev["rag"] = rag_eval.evaluate_rag(rag, cfg, judge, log)
        ev["forecasting"] = fc_summary
        write_json(ev, P["evaluation_report"])
        return ev

    def verify_outputs(self):
        P = self.cfg["paths"]
        expected = [P["cleaned_csv"], P["output_csv"], P["output_json"], P["taxonomy_path"], P["review_csv"],
                    P["profile_report"], P["profile_html"], P["labeling_report"], P["trend_csv"], P["forecast_csv"],
                    P["repeated_csv"], P["vector_db_dir"] / "index.faiss", P["vector_db_dir"] / "metadata.jsonl"]
        if not self.skip_eval:
            expected += [P["evaluation_report"], P["evaluation_html"]]
        missing = [rel(p) for p in expected if not p.exists()]
        if missing:
            self.log.error("Missing expected outputs: %s", missing)
        else:
            self.log.info("  all %d expected output files present", len(expected))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="recompute every stage")
    ap.add_argument("--skip-eval", action="store_true", help="skip the evaluation stage")
    ap.add_argument("--stage", choices=["evaluation"], help="force re-running a specific stage")
    a = ap.parse_args()
    p = Pipeline(force=a.force, skip_eval=a.skip_eval)
    if a.stage == "evaluation":
        p.state.state.pop("evaluation", None)
    return p.run()


if __name__ == "__main__":
    sys.exit(main())
