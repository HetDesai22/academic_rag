"""Automatic (silver) classification check: agreement with an independent LLM reference labeler.

This is NOT ground truth and is reported separately from the human-verified evaluation
(classification_eval.py). The LLM labels the validation-sample questions blind (it does not see
the classifier's prediction). Because the same LLM family also adjudicated medium-confidence
items during labeling, agreement is reported separately for semantic-only labels.
"""
from src.topic_modeling.taxonomy import flatten
from src.utils.common import read_json, write_json


def reference_labels(sample, src, cfg, llm, log):
    path = cfg["paths"]["evaluation_dir"] / "llm_reference_labels.json"
    cached = read_json(path)
    if cached and set(cached["labels"]) >= set(sample["question_id"]):
        return cached["labels"], cached.get("model", "")
    if not getattr(llm, "available", False):
        return None, ""
    rows = flatten(src)
    labels = {}
    for subject, g in sample.groupby("subject"):
        sr = [r for r in rows if r["subject"] == subject]
        topics = list(dict.fromkeys(r["topic"] for r in sr))
        tax = "\n".join(f"TOPIC: {t}\n" + "\n".join(f"  - {r['subtopic']}: {r['description']}" for r in sr if r["topic"] == t)
                        for t in topics)
        schema = {"type": "object", "properties": {"labels": {"type": "array", "items": {
            "type": "object", "properties": {"id": {"type": "string"}, "topic": {"type": "string", "enum": topics + ["NONE_OF_THESE"]},
                                             "subtopic": {"type": "string"}},
            "required": ["id", "topic", "subtopic"], "additionalProperties": False}}},
            "required": ["labels"], "additionalProperties": False}
        for start in range(0, len(g), 10):
            chunk = g.iloc[start:start + 10]
            listing = "\n".join(f"{r.question_id}: {r.cleaned_question_text}" for r in chunk.itertuples())
            try:
                out = llm.generate(
                    "You label exam questions with the syllabus topic and subtopic they primarily examine. "
                    "Use only names from the taxonomy, spelled exactly. Answer NONE_OF_THESE if nothing fits.",
                    f"SUBJECT: {subject}\n\nTAXONOMY:\n{tax}\n\nQUESTIONS:\n{listing}", json_schema=schema, max_tokens=3000)
            except Exception as e:
                log.error("  LLM reference labeling failed for %s: %s", subject, e)
                continue
            for lab in out["json"]["labels"]:
                labels[lab["id"]] = {"topic": lab["topic"], "subtopic": lab["subtopic"]}
    write_json({"description": "Blind LLM reference labels (silver standard, not ground truth).",
                "model": getattr(llm, "model", ""), "labels": labels}, path)
    return labels, getattr(llm, "model", "")


def agreement(sample, labels, model):
    if not labels:
        return {"status": "skipped", "reason": "no LLM available for reference labels"}
    d = sample[sample["question_id"].isin(labels)].copy()
    d["ref_topic"] = d["question_id"].map(lambda q: labels[q]["topic"])
    d["ref_subtopic"] = d["question_id"].map(lambda q: labels[q]["subtopic"])
    d["topic_agree"] = d["ref_topic"] == d["predicted_topic"]
    d["sub_agree"] = d["topic_agree"] & (d["ref_subtopic"] == d["predicted_subtopic"])
    semantic_only = d["labeling_method"] == "semantic_classification"

    def rate(mask):
        x = d[mask]
        return {"n": int(len(x)), "topic_agreement": round(float(x["topic_agree"].mean()), 3) if len(x) else None,
                "subtopic_agreement": round(float(x["sub_agree"].mean()), 3) if len(x) else None}
    return {
        "status": "ok", "type": "AUTOMATIC - agreement with an LLM reference labeler (silver standard, NOT ground truth)",
        "reference_model": model,
        "all": rate(d.index == d.index),
        "semantic_only_labels": rate(semantic_only),
        "llm_checked_labels": rate(~semantic_only),
        "by_confidence_band": {b: rate(d["confidence_band"] == b) for b in sorted(d["confidence_band"].unique())},
        "reference_none_of_these": int((d["ref_topic"] == "NONE_OF_THESE").sum()),
        "disagreements": d.loc[~d["topic_agree"], ["question_id", "cleaned_question_text", "predicted_topic", "ref_topic"]]
                          .head(25).to_dict("records"),
    }
