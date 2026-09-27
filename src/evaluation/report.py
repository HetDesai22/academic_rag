"""reports/evaluation_report.html from the evaluation JSON."""
import pandas as pd

from src.visualization import charts
from src.visualization.html import df_table, html_page, kv_table, note


def _metrics_frame(block):
    rows = []
    for system in ("semantic_faiss", "tfidf_baseline"):
        if system in block:
            rows.append({"system": system, **{k: round(v, 3) if isinstance(v, float) else v for k, v in block[system].items()}})
    return pd.DataFrame(rows)


def evaluation_html(ev):
    sections = []
    cm = ev.get("classification_manual", {})
    if cm.get("status") == "ok":
        sections.append(("Topic classification — human-verified (ground truth)", kv_table({
            "Verified questions": cm["evaluated_rows"], "Topic accuracy": f"{cm['topic']['accuracy']:.3f}",
            "Macro precision / recall / F1": f"{cm['topic']['macro_precision']:.3f} / {cm['topic']['macro_recall']:.3f} / {cm['topic']['macro_f1']:.3f}",
            "Subtopic accuracy": f"{cm['subtopic']['accuracy']:.3f}",
            "Multi-label micro F1": f"{cm['multi_label_topics']['micro_f1']:.3f}",
        })))
        for s, m in cm.get("confusion_matrices", {}).items():
            sections.append((f"Confusion matrix — {s}", charts.heatmap(
                pd.DataFrame(m["rows_true_cols_pred"], index=m["labels"], columns=m["labels"]), "Predicted", "True")))
    else:
        sections.append(("Topic classification — human-verified (ground truth)", note(
            "Not yet available: no rows of data/review/validation_sample.csv have been verified by a person. "
            "Fill the 'verified' column and rerun `python pipeline.py --stage evaluation`. No accuracy is claimed until then.")))
    cr = ev.get("classification_llm_reference", {})
    if cr.get("status") == "ok":
        sections.append(("Topic classification — automatic LLM reference agreement (silver, NOT ground truth)", kv_table({
            "Reference model": cr["reference_model"],
            "All (n)": f"topic {cr['all']['topic_agreement']} · subtopic {cr['all']['subtopic_agreement']} (n={cr['all']['n']})",
            "Semantic-only labels": f"topic {cr['semantic_only_labels']['topic_agreement']} (n={cr['semantic_only_labels']['n']})",
            "LLM-checked labels": f"topic {cr['llm_checked_labels']['topic_agreement']} (n={cr['llm_checked_labels']['n']})",
            **{f"Band: {b}": f"topic {v['topic_agreement']} (n={v['n']})" for b, v in cr["by_confidence_band"].items()},
        }) + df_table(pd.DataFrame(cr["disagreements"]))))
    rt = ev.get("retrieval", {})
    for name in ("repeat_set", "paraphrase_set"):
        blk = rt.get(name, {})
        if "semantic_faiss" in blk:
            sections.append((f"Retrieval — {name.replace('_', ' ')}", note(blk["description"]) + df_table(_metrics_frame(blk))))
        elif blk:
            sections.append((f"Retrieval — {name.replace('_', ' ')}", note(f"Skipped: {blk.get('reason')}")))
    rg = ev.get("rag", {})
    if rg.get("per_query"):
        det = rg["deterministic"]
        sections.append(("RAG — deterministic checks", kv_table({"Generator": f"{rg['generator_provider']} / {rg['generator_model']}", **det})))
        sections.append(("RAG — automatic LLM judge (silver, not human evaluation)", kv_table(rg["llm_judge"])))
        pq = pd.DataFrame([{k: r.get(k) for k in ("id", "query", "insufficient", "expect_insufficient", "refusal_correct",
                                                   "best_similarity", "context_keyword_precision", "citation_coverage", "grounded")}
                           | ({"faithfulness": r["judge"]["faithfulness"], "answer_relevance": r["judge"]["answer_relevance"]}
                              if "judge" in r else {}) for r in rg["per_query"]])
        sections.append(("RAG — per query", df_table(pq)))
    fc = ev.get("forecasting", {})
    if fc.get("metrics"):
        m = pd.DataFrame(fc["metrics"])
        sections.append(("Forecasting — rolling-origin backtest", note(
            f"Each method is fit on exams before the origin and scored on the origin exam it never saw. "
            f"Origins: {', '.join(fc['backtest_origins'])}. Selected: {fc['selected_method']}. {fc['disclaimer']}")
            + df_table(m.round(3))))
        sections.append(("Forecasting — error by method", charts.backtest_chart(m)))
    return html_page("Evaluation Report", "Retrieval · classification · RAG · forecasting", sections, ev.get("generated_at", ""))
