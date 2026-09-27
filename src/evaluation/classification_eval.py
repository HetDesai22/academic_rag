"""Stage 7: sample-based quality evaluation against human-verified labels.

1. `create_sample` draws a stratified random sample (fixed seed) and writes it with the
   pipeline's predictions to data/review/validation_sample.csv. An existing sample is never
   overwritten, so human edits are safe across reruns.
2. A person fills `verified` for each row: "y" if the predicted topic AND subtopic are right,
   otherwise "n" plus `correct_topic` / `correct_subtopic` (and `correct_secondary_topics`,
   a JSON list, if the secondary topics are wrong).
3. `evaluate` computes metrics ONLY from rows a person has verified. Nothing is estimated
   or invented; with no verified rows it reports that no metrics are available.

Note: showing predictions to the reviewer is faster than blind labeling but can bias toward
agreement; the report records this.

Run:  python src/evaluation.py            (computes metrics from the filled sample)
"""
import json

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config, now_iso, read_json, write_json
from src.topic_modeling.label_validation import parse_list

SAMPLE_COLUMNS = [
    "question_id", "subject", "cleaned_question_text", "predicted_topic", "predicted_subtopic",
    "predicted_secondary_topics", "topic_confidence", "confidence_band", "labeling_method",
    "review_required", "verified", "correct_topic", "correct_subtopic", "correct_secondary_topics", "notes",
]


def create_sample(labeled, cfg, log):
    path = cfg["paths"]["validation_sample_csv"]
    if path.exists():
        log.info("Validation sample exists (%s); not overwriting human edits", path.name)
        return pd.read_csv(path, encoding="utf-8-sig")
    ecfg = cfg["evaluation"]
    n, by = ecfg["sample_size"], ecfg["stratify_by"]
    # Proportional stratified sample over subjects; unique question texts only, so a question
    # repeated across years is not verified several times.
    pool = labeled.drop_duplicates(["subject", "cleaned_question_text"])
    frac = min(1.0, n / len(pool))
    sample = pd.concat([g.sample(max(1, round(len(g) * frac)), random_state=cfg["random_seed"])
                        for _, g in pool.groupby(by)])
    sample = sample.sample(frac=1, random_state=cfg["random_seed"])  # shuffle order across subjects
    out = pd.DataFrame({
        "question_id": sample["question_id"],
        "subject": sample["subject"],
        "cleaned_question_text": sample["cleaned_question_text"],
        "predicted_topic": sample["primary_topic"],
        "predicted_subtopic": sample["subtopic"],
        "predicted_secondary_topics": sample["secondary_topics"],
        "topic_confidence": sample["topic_confidence"].round(3),
        "confidence_band": sample["confidence_band"],
        "labeling_method": sample["labeling_method"],
        "review_required": sample["review_required"],
        "verified": "", "correct_topic": "", "correct_subtopic": "",
        "correct_secondary_topics": "", "notes": "",
    })[SAMPLE_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("Wrote validation sample of %d questions to %s", len(out), path)
    return out


def _gold(row):
    v = str(row["verified"]).strip().lower()
    if v in ("y", "yes", "true", "1"):
        return row["predicted_topic"], row["predicted_subtopic"], parse_list(row["predicted_secondary_topics"])
    topic = str(row["correct_topic"]).strip() or row["predicted_topic"]
    sub = str(row["correct_subtopic"]).strip() or row["predicted_subtopic"]
    sec_raw = str(row["correct_secondary_topics"]).strip()
    sec = parse_list(sec_raw) if sec_raw else parse_list(row["predicted_secondary_topics"])
    return topic, sub, sec


def evaluate(sample, taxonomy, log):
    from sklearn.metrics import (accuracy_score, confusion_matrix, hamming_loss, jaccard_score,
                                 precision_recall_fscore_support)
    from sklearn.preprocessing import MultiLabelBinarizer

    sample = sample.fillna("")
    done = sample[sample["verified"].astype(str).str.strip().str.lower().isin(["y", "yes", "true", "1", "n", "no", "false", "0"])]
    report = {"generated_at": now_iso(), "sample_size": int(len(sample)), "verified_rows": int(len(done)),
              "method_note": "Gold labels = human-verified predictions (reviewer saw predictions)."}
    if done.empty:
        report["status"] = "no_verified_rows"
        report["message"] = "Fill the 'verified' column in the validation sample, then rerun evaluation."
        log.warning(report["message"])
        return report

    gold = done.apply(_gold, axis=1, result_type="expand")
    gold.columns = ["gold_topic", "gold_subtopic", "gold_secondary"]
    d = pd.concat([done.reset_index(drop=True), gold.reset_index(drop=True)], axis=1)

    invalid = []
    for _, r in d.iterrows():
        topics = taxonomy["subjects"][r["subject"]]["topics"]
        if r["gold_topic"] not in topics or r["gold_subtopic"] not in topics[r["gold_topic"]]["subtopics"]:
            invalid.append({"question_id": r["question_id"], "gold_topic": r["gold_topic"], "gold_subtopic": r["gold_subtopic"]})
    if invalid:
        report["invalid_gold_labels"] = invalid
        log.warning("%d verified rows use labels not in the taxonomy; they are excluded", len(invalid))
        bad = {x["question_id"] for x in invalid}
        d = d[~d["question_id"].isin(bad)]

    y_true, y_pred = d["gold_topic"], d["predicted_topic"]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    sub_true = d["gold_topic"] + " / " + d["gold_subtopic"]
    sub_pred = d["predicted_topic"] + " / " + d["predicted_subtopic"]

    # Multi-label view: the set {primary} U secondary.
    mlb = MultiLabelBinarizer()
    true_sets = [[a] + list(b) for a, b in zip(d["gold_topic"], d["gold_secondary"])]
    pred_sets = [[a] + parse_list(b) for a, b in zip(d["predicted_topic"], d["predicted_secondary_topics"])]
    mlb.fit(true_sets + pred_sets)
    T, P = mlb.transform(true_sets), mlb.transform(pred_sets)
    mp, mr, mf, _ = precision_recall_fscore_support(T, P, average="micro", zero_division=0)

    report.update({
        "status": "ok",
        "evaluated_rows": int(len(d)),
        "topic": {"accuracy": accuracy_score(y_true, y_pred),
                  "macro_precision": p, "macro_recall": r, "macro_f1": f,
                  "weighted_precision": pw, "weighted_recall": rw, "weighted_f1": fw},
        "subtopic": {"accuracy": accuracy_score(sub_true, sub_pred)},
        "multi_label_topics": {"micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
                               "hamming_loss": hamming_loss(T, P),
                               "jaccard_samples": jaccard_score(T, P, average="samples", zero_division=0)},
        # Accuracy per confidence band: the basis for re-tuning thresholds/temperature.
        "calibration_by_band": {
            band: {"n": int(len(g)), "topic_accuracy": float((g["gold_topic"] == g["predicted_topic"]).mean())}
            for band, g in d.groupby("confidence_band")
        },
        "auto_accepted_accuracy": (
            float((d.loc[~d["review_required"].astype(str).str.lower().eq("true"), "gold_topic"]
                   == d.loc[~d["review_required"].astype(str).str.lower().eq("true"), "predicted_topic"]).mean())
            if (~d["review_required"].astype(str).str.lower().eq("true")).any() else None),
        "per_subject_topic_accuracy": {
            s: {"n": int(len(g)), "accuracy": float((g["gold_topic"] == g["predicted_topic"]).mean())}
            for s, g in d.groupby("subject")
        },
        "confusion_matrices": {},
        "errors": d.loc[d["gold_topic"] != d["predicted_topic"],
                        ["question_id", "cleaned_question_text", "predicted_topic", "gold_topic"]].to_dict("records"),
    })
    for s, g in d.groupby("subject"):
        labels = sorted(set(g["gold_topic"]) | set(g["predicted_topic"]))
        cm = confusion_matrix(g["gold_topic"], g["predicted_topic"], labels=labels)
        report["confusion_matrices"][s] = {"labels": labels, "rows_true_cols_pred": cm.tolist()}
    return json.loads(json.dumps(report, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))


def main():
    cfg = load_config()
    log = get_logger(cfg)
    path = cfg["paths"]["validation_sample_csv"]
    if not path.exists():
        log.error("No validation sample at %s; run the pipeline first", path)
        return
    sample = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    report = evaluate(sample, read_json(cfg["paths"]["taxonomy_path"]), log)
    write_json(report, cfg["paths"]["evaluation_report"])
    if report.get("status") == "ok":
        log.info("Evaluated %d verified questions: topic accuracy %.3f, macro F1 %.3f, subtopic accuracy %.3f",
                 report["evaluated_rows"], report["topic"]["accuracy"], report["topic"]["macro_f1"],
                 report["subtopic"]["accuracy"])
    log.info("Evaluation report -> %s", cfg["paths"]["evaluation_report"])


if __name__ == "__main__":
    main()
