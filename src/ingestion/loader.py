"""Load question CSVs, detect/map columns, and build the standardized internal schema.

Inputs are only read. Missing fields are reported and left absent - never invented.
New papers can be added later as extra CSVs matching `paths.incoming_glob`.
"""
import glob
import re

import pandas as pd

from src.utils.common import ROOT, rel, sha256_file

# Candidate source-column names per standardized field (case-insensitive, first match wins).
COLUMN_CANDIDATES = {
    "question_id": ["question_id", "qid", "id"],
    "paper_id": ["paper_id", "paper", "paper_code"],
    "university": ["university", "board", "institution"],
    "subject": ["subject", "subject_name", "course", "course_name"],
    "subject_code": ["subject_code", "course_code", "code"],
    "year": ["year", "exam_year"],
    "exam_session": ["exam_session", "session", "term", "season"],
    "semester": ["semester", "sem"],
    "exam_date": ["exam_date", "date"],
    "question_number": ["question_number", "question_no", "qno", "q_no"],
    "sub_part": ["sub_part", "part", "subpart"],
    "is_or_option": ["is_or_option", "or_option", "is_alternative"],
    "question_text": ["question_text", "question", "text", "questions"],
    "marks": ["marks", "mark", "score", "max_marks"],
    "source_file": ["source_file", "file", "filename", "source"],
    "page_number": ["page_number", "page", "page_no"],
}
REQUIRED = ["question_text", "subject"]
STANDARD_FIELDS = list(COLUMN_CANDIDATES)


def detect_columns(df, overrides=None):
    lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for field, candidates in COLUMN_CANDIDATES.items():
        override = (overrides or {}).get(field)
        if override:
            if override not in df.columns:
                raise KeyError(f"Configured column '{override}' for '{field}' not in CSV")
            mapping[field] = override
            continue
        mapping[field] = next((lower[c] for c in candidates if c in lower), None)
    missing = [f for f in REQUIRED if mapping[f] is None]
    if missing:
        raise KeyError(f"Required fields not found in CSV: {missing}. Columns: {list(df.columns)}")
    return mapping


def read_inputs(cfg, log):
    """Read the original CSV plus any incoming CSVs. Returns (raw_df, input_files)."""
    files = [cfg["paths"]["input_csv"]]
    files += sorted(ROOT / p for p in glob.glob(cfg["paths"]["incoming_glob"], root_dir=ROOT))
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, encoding="utf-8-sig")
        except Exception as e:
            log.error("Could not read %s: %s", rel(f), e)
            continue
        d["_input_file"] = rel(f)
        frames.append(d)
        log.info("  read %s: %d rows", rel(f), len(d))
    if not frames:
        raise FileNotFoundError("No readable input CSV found")
    raw = pd.concat(frames, ignore_index=True)
    return raw, [{"file": rel(f), "sha256": sha256_file(f)} for f in files if f.exists()]


def standardize(raw, mapping, log):
    """Build the standardized schema. Original columns are kept alongside (prefixed `orig_`
    only where names collide)."""
    m = mapping
    df = raw.copy()
    std = pd.DataFrame(index=df.index)
    for field in STANDARD_FIELDS:
        if m[field] is not None:
            std[field] = df[m[field]]

    # Derived fields: built only from columns that exist.
    if "exam_session" in std:
        std["exam_session"] = std["exam_session"].astype(str).str.strip().str.title()
    if "year" in std:
        std["year"] = pd.to_numeric(std["year"], errors="coerce").astype("Int64")
        if std["year"].isna().any():
            log.warning("  %d rows have a missing/invalid year", int(std["year"].isna().sum()))
    if "paper_id" not in std:
        if {"subject_code", "exam_session", "year"} <= set(std):
            std["paper_id"] = (std["subject_code"].astype(str) + "_" + std["exam_session"].str[0].str.upper()
                               + std["year"].astype(str))
            log.info("  derived paper_id from subject_code + exam_session + year")
        elif "source_file" in std:
            std["paper_id"] = std["source_file"].map(lambda s: re.sub(r"\W+", "_", str(s)))
            log.info("  derived paper_id from source_file")
    if "question_id" not in std:
        if {"paper_id", "question_number", "sub_part"} <= set(std):
            qid = std["paper_id"] + "_Q" + std["question_number"].astype(str) + std["sub_part"].astype(str)
            if "is_or_option" in std:
                qid = qid + std["is_or_option"].map({True: "_OR", False: "", "True": "_OR", "False": ""}).fillna("")
            std["question_id"] = qid
            log.info("  derived question_id from paper_id + question_number + sub_part (+ _OR)")
        else:
            std["question_id"] = [f"row_{i:05d}" for i in range(len(std))]
            log.warning("  not enough metadata for a semantic question_id; using row index")
    # Chronological exam period: sessions within a year ordered Summer (May/Jun) -> Winter (Nov/Dec).
    if {"year", "exam_session"} <= set(std):
        order = std["exam_session"].map({"Summer": 0, "Winter": 1})
        std["period_index"] = std["year"].astype("float") * 2 + order
        std["period"] = std["exam_session"].str[0].str.upper() + std["year"].astype(str)
    elif "year" in std:
        std["period_index"] = std["year"].astype("float")
        std["period"] = std["year"].astype(str)

    # Keep original columns that are not already represented.
    used = {c for c in m.values() if c}
    for c in df.columns:
        if c not in used and c not in std.columns:
            std[c] = df[c]
    dupe_ids = std["question_id"].duplicated(keep="first")
    if dupe_ids.any():
        log.warning("  %d duplicate question_ids across inputs (kept first occurrence): %s",
                    int(dupe_ids.sum()), std.loc[dupe_ids, "question_id"].head(5).tolist())
        std = std[~dupe_ids]
    return std.reset_index(drop=True)


def load_questions(cfg, log):
    raw, files = read_inputs(cfg, log)
    mapping = detect_columns(raw, cfg.get("columns"))
    df = standardize(raw, mapping, log)
    missing = [f for f in STANDARD_FIELDS if f not in df.columns]
    derived = [f for f in ("question_id", "paper_id") if mapping.get(f) is None and f in df.columns]
    return df, {"column_mapping": mapping, "missing_fields": missing, "derived_fields": derived,
                "input_files": files, "original_columns": list(raw.columns.drop("_input_file"))}
