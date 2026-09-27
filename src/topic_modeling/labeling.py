"""Topic labeling stage: classify -> confidence routing -> LLM check -> human review merge.

Human decisions in data/review/topic_review.csv are applied on every run and kept across runs.
Corrected/approved questions also become exemplar prototypes for the classifier (excluding
validation-sample questions, so evaluation stays unbiased).
"""
import json

import numpy as np
import pandas as pd

from src.topic_modeling.classifier import SubjectClassifier, confidence_band
from src.topic_modeling.label_validation import parse_list
from src.topic_modeling.llm_validator import LLMValidator

REVIEW_COLUMNS = [
    "question_id", "subject", "question_text", "predicted_topic", "predicted_subtopic",
    "predicted_secondary_topics", "confidence", "suggested_alternatives", "review_required", "review_reason",
    "llm_suggestion", "applies_to_question_ids", "review_status", "corrected_topic", "corrected_subtopic",
    "corrected_secondary_topics", "reviewer_notes",
]
REVIEWER_FIELDS = ["review_status", "corrected_topic", "corrected_subtopic", "corrected_secondary_topics", "reviewer_notes"]


def load_review(cfg):
    path = cfg["paths"]["review_csv"]
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    prev = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    for c in REVIEW_COLUMNS:
        if c not in prev.columns:
            prev[c] = ""
    return prev


def review_exemplars(prev, df, cfg):
    """(subject -> [(topic, subtopic, text)]) from human decisions, excluding the validation sample."""
    if not cfg["classifier"].get("use_review_exemplars", True) or prev.empty:
        return {}
    excluded = set()
    vs = cfg["paths"]["validation_sample_csv"]
    if vs.exists():
        excluded = set(pd.read_csv(vs, dtype=str)["question_id"])
    text_of = df.set_index("question_id")["cleaned_question_text"].to_dict()
    out = {}
    for _, r in prev.iterrows():
        status = r["review_status"].strip().lower()
        if status not in ("approved", "corrected") or r["question_id"] in excluded:
            continue
        topic = r["predicted_topic"] if status == "approved" else r["corrected_topic"].strip()
        sub = r["predicted_subtopic"] if status == "approved" else r["corrected_subtopic"].strip()
        text = text_of.get(r["question_id"], r["question_text"])
        out.setdefault(r["subject"], []).append((topic, sub, text))
    return out


def classify_all(df, src, embedder, cfg, log, exemplars=None):
    parts = []
    for subject, g in df.groupby("subject"):
        clf = SubjectClassifier(subject, src, embedder, cfg, (exemplars or {}).get(subject))
        res = pd.DataFrame(clf.classify(g["cleaned_question_text"].tolist()), index=g.index)
        parts.append(res)
        log.info("  %s: %d questions classified%s", subject, len(g),
                 f" ({clf.n_exemplars} human exemplars)" if clf.n_exemplars else "")
    return df.join(pd.concat(parts).loc[df.index])


def apply_llm(df, rows_by_subject, cfg, log):
    th = cfg["confidence_thresholds"]
    df["confidence_band"] = df["topic_confidence"].map(lambda c: confidence_band(c, th))
    df["labeling_method"] = "semantic_classification"
    df["review_required"] = df["confidence_band"] == "low"
    df["review_reason"] = np.where(df["review_required"], "low_confidence", "")
    for c in ("llm_topic", "llm_subtopic", "llm_confidence", "llm_decision", "llm_reason", "llm_model"):
        df[c] = ""

    llm = LLMValidator(cfg, log)
    lcfg = cfg["topic_llm_check"]
    bands = set(lcfg.get("apply_to", []))
    hint_bands = set(lcfg.get("suggest_for", [])) - bands
    targets = df.index[df["confidence_band"].isin(bands)]
    hint_targets = df.index[df["confidence_band"].isin(hint_bands)]
    if not llm.enabled:
        df.loc[targets, "review_required"] = True
        df.loc[targets, "review_reason"] = "medium_confidence_no_llm"
        log.warning("LLM topic check unavailable: %d medium-confidence questions go to human review", len(targets))
        return df, {"enabled": False, "sent": 0}

    stats = {"enabled": True, "model": lcfg["model"], "candidates": int(len(targets))}
    for i in targets:
        r = df.loc[i]
        cand = {"primary_topic": r["primary_topic"], "subtopic": r["subtopic"], "top_topics": r["top_topics"]}
        res = llm.validate(r["subject"], r["cleaned_question_text"], rows_by_subject[r["subject"]], cand)
        if res["status"] != "ok":
            df.loc[i, ["review_required", "review_reason", "llm_decision"]] = [True, f"llm_{res['status']}", res["status"]]
            continue
        df.loc[i, ["llm_topic", "llm_subtopic", "llm_confidence", "llm_decision", "llm_reason", "llm_model"]] = [
            res["llm_topic"], res["llm_subtopic"], res["llm_confidence"], res["decision"], res["llm_reason"], res["llm_model"]]
        decision = res["decision"]
        if decision == "confirmed":
            df.at[i, "labeling_method"] = "semantic+llm_confirmed"
            if res["llm_subtopic"] != r["subtopic"]:
                if res["llm_confidence"] == "high":
                    df.at[i, "subtopic"] = res["llm_subtopic"]
                    df.at[i, "labeling_method"] = "semantic+llm_confirmed_subtopic_adjusted"
                else:
                    df.loc[i, ["review_required", "review_reason"]] = [True, "llm_subtopic_disagreement"]
        elif decision == "adjudicated":
            df.at[i, "primary_topic"] = res["llm_topic"]
            df.at[i, "subtopic"] = res["llm_subtopic"]
            df.at[i, "secondary_topics"] = [t for t in r["secondary_topics"] if t != res["llm_topic"]]
            df.at[i, "labeling_method"] = "semantic+llm_adjudicated"
        else:
            df.loc[i, ["review_required", "review_reason"]] = [True, decision]
        if decision in ("confirmed", "adjudicated") and res["llm_confidence"] == "high":
            sec = list(dict.fromkeys(list(df.at[i, "secondary_topics"]) + res["llm_secondary"]))
            df.at[i, "secondary_topics"] = [t for t in sec if t != df.at[i, "primary_topic"]][: cfg["classifier"]["secondary_max"]]
    # Low-confidence items: the LLM answer is only a hint for the human reviewer.
    for i in hint_targets:
        r = df.loc[i]
        cand = {"primary_topic": r["primary_topic"], "subtopic": r["subtopic"], "top_topics": r["top_topics"]}
        res = llm.validate(r["subject"], r["cleaned_question_text"], rows_by_subject[r["subject"]], cand)
        if res["status"] == "ok":
            df.loc[i, ["llm_topic", "llm_subtopic", "llm_confidence", "llm_decision", "llm_reason", "llm_model"]] = [
                res["llm_topic"], res["llm_subtopic"], res["llm_confidence"], "hint_only", res["llm_reason"], res["llm_model"]]
    stats.update({"hint_candidates": int(len(hint_targets)), "api_calls": llm.calls,
                  "decisions": df.loc[targets, "llm_decision"].value_counts().to_dict()})
    log.info("  LLM check: %d candidates + %d hints, %d API calls (rest cached), decisions %s",
             len(targets), len(hint_targets), llm.calls, stats["decisions"])
    return df, stats


def apply_human_review(df, prev, taxonomy, log):
    subjects = taxonomy["subjects"]
    applied = 0
    for _, r in prev.iterrows():
        status = r["review_status"].strip().lower()
        if status not in ("approved", "corrected"):
            continue
        ids = [x for x in r.get("applies_to_question_ids", "").split(";") if x] or [r["question_id"]]
        idx = df.index[df["question_id"].isin(ids)]
        if len(idx) == 0:
            log.error("Review row %s matches no question; skipped", r["question_id"])
            continue
        applied += _apply_decision(df, idx, r, status, subjects, log)
    log.info("  applied human review decisions to %d questions", applied)
    return df, applied


def _apply_decision(df, idx, r, status, subjects, log):
    n = 0
    for i in idx:
        if status == "approved":
            if (df.at[i, "primary_topic"], df.at[i, "subtopic"]) != (r["predicted_topic"], r["predicted_subtopic"]):
                log.warning("%s was approved for %s / %s but the prediction changed; left for re-review",
                            r["question_id"], r["predicted_topic"], r["predicted_subtopic"])
                continue
            topic, sub, sec = r["predicted_topic"], r["predicted_subtopic"], df.at[i, "secondary_topics"]
        else:
            topic, sub = r["corrected_topic"].strip(), r["corrected_subtopic"].strip()
            try:
                sec = parse_list(r["corrected_secondary_topics"]) if r["corrected_secondary_topics"].strip() else []
            except json.JSONDecodeError:
                log.error("%s: corrected_secondary_topics is not a JSON list; skipped", r["question_id"])
                continue
        topics = subjects[df.at[i, "subject"]]["topics"]
        if topic not in topics or sub not in topics[topic]["subtopics"] or any(t not in topics for t in sec):
            log.error("%s: reviewer label %s / %s is not in the taxonomy; skipped", r["question_id"], topic, sub)
            continue
        df.at[i, "primary_topic"], df.at[i, "subtopic"] = topic, sub
        df.at[i, "secondary_topics"] = [t for t in sec if t != topic]
        df.at[i, "labeling_method"] = "human_review"
        df.at[i, "review_required"] = False
        df.at[i, "review_reason"] = f"human_{status}"
        n += 1
    return n


def write_review_csv(df, prev, cfg):
    """Rewrite the review file with currently flagged items, keeping every reviewer input."""
    flagged = df[df["review_required"] | df["labeling_method"].eq("human_review")]
    key = flagged["subject"] + " || " + flagged["cleaned_question_text"]
    also = flagged.groupby(key)["question_id"].transform(lambda s: ";".join(s))
    flagged = flagged.assign(_also=also)[~key.duplicated()]
    llm_hint = flagged.apply(
        lambda r: f"{r['llm_topic']} / {r['llm_subtopic']} ({r['llm_confidence']})" if r["llm_topic"] else "", axis=1)
    out = pd.DataFrame({
        "question_id": flagged["question_id"],
        "subject": flagged["subject"],
        "question_text": flagged["cleaned_question_text"],
        "predicted_topic": flagged["primary_topic"],
        "predicted_subtopic": flagged["subtopic"],
        "predicted_secondary_topics": flagged["secondary_topics"].map(lambda x: json.dumps(list(x))),
        "confidence": flagged["topic_confidence"].round(3),
        "suggested_alternatives": flagged["alternatives"].map(lambda a: " | ".join(a)),
        "review_required": flagged["review_required"],
        "review_reason": flagged["review_reason"],
        "llm_suggestion": llm_hint,
        "applies_to_question_ids": flagged["_also"],
        "review_status": "pending",
        "corrected_topic": "", "corrected_subtopic": "", "corrected_secondary_topics": "", "reviewer_notes": "",
    })
    if not prev.empty:
        p = prev.drop_duplicates("question_id").set_index("question_id")
        seen = out["question_id"].isin(p.index)
        for f in REVIEWER_FIELDS:
            out.loc[seen, f] = out.loc[seen, "question_id"].map(p[f]).values
        decided = seen & out["review_status"].str.lower().isin(["approved", "corrected"])
        for f in ("predicted_topic", "predicted_subtopic"):
            out.loc[decided, f] = out.loc[decided, "question_id"].map(p[f]).values
    out = out[REVIEW_COLUMNS].sort_values(["subject", "confidence"])
    path = cfg["paths"]["review_csv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out
