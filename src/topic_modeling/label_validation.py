"""Stage 6: automated validation of the labeled dataset and the labeling report."""
import json

from src.utils.common import now_iso

LIST_COLUMNS = ("secondary_topics",)


def parse_list(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    return json.loads(value)


def validate(df, taxonomy, thresholds, id_col="question_id", subject_col="subject"):
    """Return (checks, errors). Each check: name -> number of offending rows (0 = pass)."""
    subjects = taxonomy["subjects"]
    errors = []

    def fail(name, mask):
        idx = df.index[mask]
        if len(idx):
            errors.append({"check": name, "count": int(len(idx)),
                           "question_ids": df.loc[idx, id_col].head(20).tolist()})
        return int(len(idx))

    q = df["cleaned_question_text"].fillna("")
    topic_ok = df.apply(lambda r: r[subject_col] in subjects and r["primary_topic"] in subjects[r[subject_col]]["topics"], axis=1)
    sub_ok = df.apply(lambda r: topic_ok[r.name] and r["subtopic"] in subjects[r[subject_col]]["topics"][r["primary_topic"]]["subtopics"], axis=1)

    def secondary_ok(r):
        try:
            sec = parse_list(r["secondary_topics"])
        except (json.JSONDecodeError, TypeError):
            return False
        topics = subjects.get(r[subject_col], {}).get("topics", {})
        return (isinstance(sec, list) and all(isinstance(t, str) and t in topics for t in sec)
                and r["primary_topic"] not in sec and len(sec) == len(set(sec)))

    conf = df["topic_confidence"]
    checks = {
        "missing_question_text": fail("missing_question_text", q.str.strip() == ""),
        "missing_subject": fail("missing_subject", df[subject_col].isna() | (df[subject_col].astype(str).str.strip() == "")),
        "empty_primary_topic": fail("empty_primary_topic", df["primary_topic"].fillna("").str.strip() == ""),
        "empty_subtopic": fail("empty_subtopic", df["subtopic"].fillna("").str.strip() == ""),
        "subject_not_in_taxonomy": fail("subject_not_in_taxonomy", ~df[subject_col].isin(subjects)),
        "topic_not_in_subject_taxonomy": fail("topic_not_in_subject_taxonomy", ~topic_ok),
        "subtopic_not_under_topic": fail("subtopic_not_under_topic", ~sub_ok),
        "invalid_secondary_topics": fail("invalid_secondary_topics", ~df.apply(secondary_ok, axis=1)),
        "invalid_confidence": fail("invalid_confidence", conf.isna() | (conf < 0) | (conf > 1)),
        "duplicate_question_ids": fail("duplicate_question_ids", df[id_col].duplicated(keep=False)),
        "low_confidence_not_flagged": fail("low_confidence_not_flagged",
                                           (conf < thresholds["low"]) & ~df["review_required"].astype(bool)
                                           & (df["labeling_method"] != "human_review")),
        "invalid_review_required": fail("invalid_review_required", ~df["review_required"].isin([True, False])),
    }
    return checks, errors


def labeling_report(df, checks, errors, meta, thresholds, subject_col="subject"):
    passed = all(v == 0 for v in checks.values())
    sec = df["secondary_topics"].map(parse_list)
    return {
        "generated_at": now_iso(),
        **meta,
        "validation_passed": passed,
        "validation_checks": checks,
        "validation_errors": errors,
        "total_questions": int(len(df)),
        "successfully_labeled": int(((df["primary_topic"].fillna("") != "") & (df["subtopic"].fillna("") != "")).sum()),
        "auto_accepted": int((~df["review_required"]).sum()),
        "review_required": int(df["review_required"].sum()),
        "review_reasons": df.loc[df["review_required"], "review_reason"].value_counts().to_dict(),
        "missing_questions": int((df["cleaned_question_text"].fillna("").str.strip() == "").sum()),
        "confidence_thresholds": thresholds,
        "confidence_band_counts": df["confidence_band"].value_counts().to_dict(),
        "low_confidence_count": int((df["confidence_band"] == "low").sum()),
        "labeling_method_counts": df["labeling_method"].value_counts().to_dict(),
        "questions_with_secondary_topics": int((sec.str.len() > 0).sum()),
        "duplicate_count": {
            "rows_in_exact_duplicate_groups": int((df["exact_duplicate_group"] != "").sum()),
            "exact_duplicate_groups": int(df.loc[df["exact_duplicate_group"] != "", "exact_duplicate_group"].nunique()),
            "rows_in_near_duplicate_groups": int((df["near_duplicate_group"] != "").sum()),
            "near_duplicate_groups": int(df.loc[df["near_duplicate_group"] != "", "near_duplicate_group"].nunique()),
        },
        "subject_distribution": df[subject_col].value_counts().to_dict(),
        "topic_distribution": {
            s: g["primary_topic"].value_counts().to_dict() for s, g in df.groupby(subject_col)
        },
        "subtopic_distribution": {
            s: {t: gt["subtopic"].value_counts().to_dict() for t, gt in g.groupby("primary_topic")}
            for s, g in df.groupby(subject_col)
        },
        "extraction_artifact_flags": {
            c: int(df[c].sum()) for c in df.columns if c.startswith("flag_")
        },
    }
