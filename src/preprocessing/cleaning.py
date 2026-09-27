"""Stage 2: clean question text and mark duplicates / extraction artifacts.

raw_question_text is kept verbatim; cleaned_question_text is derived from it. Cleaning is
deliberately light (no stemming/stopword removal) so meaning is preserved for embeddings.
Repeated questions are grouped, never dropped.
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

PREPROCESSING_VERSION = "1.1.0"

CHAR_MAP = {
    "−": "-", "–": "-", "—": "-", "‐": "-", "‑": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "×": "x", " ": " ", "�": "",
}
# Conservative domain spelling fixes (source-paper typos). Word-boundary, case-insensitive.
SPELLING_FIXES = {
    r"\blogostic\b": "logistic",
    r"\bpropog(ation|ate|ating)\b": r"propag\1",
    r"\bbackpropog(ation)?\b": r"backpropag\1",
    r"\bgradient decent\b": "gradient descent",
    r"\bna[iï]ve bays\b": "Naive Bayes",
    r"\bbays\b": "Bayes",
    r"\bco-relation\b": "correlation",
    r"\bperformance (matrices|metrices)\b": "performance metrics",
    r"\bmetrices\b": "metrics",
}
# Stray "Q.1" / "(a)" prefixes. A leading "a)" that starts an a)/b)/c) list is kept.
LEADING_NUMBERING = re.compile(r"^\s*(Q\.?\s*\d+\s*)?(\(?[a-c]\)\s+(?!.*\bb\)))?", re.I)
# Exam-paper header/footer text that PDF extraction can leak into a question.
HEADER_FOOTER = re.compile(
    r"(GUJARAT TECHNOLOGICAL UNIVERSITY|Seat No\.?[:_ ]*|Enrolment No\.?[_ ]*|Subject Code\s*:\s*\d+|"
    r"Total Marks\s*:\s*\d+|\*{3,}|Page \d+ of \d+)", re.I)
REPEATED_SENTENCE = re.compile(r"\b([^.?!]{8,}[.?!])\s+\1", re.I)  # "X. X." -> "X."
PAGE_FOOTER = re.compile(r"\s*\bPage\s*\d*\s*of\s*\d*\s*$")  # stray PDF footer "Page of"
HYPHEN_BREAK = re.compile(r"(\w)- (\w)")          # "K- Means" -> "K-Means"
MATH_ALNUM = re.compile(r"[\U0001D400-\U0001D7FF]")  # mathematical italic/bold letters
NUMBER_RUN = re.compile(r"(?:-?\d+(?:\.\d+)?\s+){5,}-?\d+")
SUBLABELS = re.compile(r"(?:^|\s)(?:\(?(?:i{1,3}|iv|v|vi)\)|\(?[a-e]\))\s", re.I)


def clean_text(text):
    """Return (cleaned_text, list_of_rules_applied)."""
    rules = []
    if not isinstance(text, str):
        return "", ["empty"]
    t = text
    nfkc = unicodedata.normalize("NFKC", t)
    if nfkc != t:
        rules.append("unicode_nfkc")
        t = nfkc
    mapped = "".join(CHAR_MAP.get(c, c) for c in t)
    if mapped != t:
        rules.append("char_map")
        t = mapped
    stripped = LEADING_NUMBERING.sub("", t, count=1)
    if stripped != t:
        rules.append("leading_numbering")
        t = stripped
    hf = HEADER_FOOTER.sub(" ", t)
    if hf != t:
        rules.append("header_footer")
        t = hf
    dedup = REPEATED_SENTENCE.sub(r"\1", t)
    if dedup != t:
        rules.append("repeated_sentence")
        t = dedup
    footer = PAGE_FOOTER.sub("", t)
    if footer != t:
        rules.append("page_footer")
        t = footer
    fixed = HYPHEN_BREAK.sub(r"\1-\2", t)
    if fixed != t:
        rules.append("hyphen_break")
        t = fixed
    for pat, rep in SPELLING_FIXES.items():
        new = re.sub(pat, rep, t, flags=re.I)
        if new != t:
            rules.append("spelling")
            t = new
    new = re.sub(r"\s+", " ", t).strip()
    new = re.sub(r"\s+([?.,;:])", r"\1", new)
    if new != t:
        rules.append("whitespace")
        t = new
    return t, sorted(set(rules))


def artifact_flags(raw):
    raw = raw if isinstance(raw, str) else ""
    return {
        "flag_math_artifact": bool(MATH_ALNUM.search(raw)) or ("matri" in raw.lower() and bool(NUMBER_RUN.search(raw))),
        "flag_table_data": bool(NUMBER_RUN.search(raw)) and "matri" not in raw.lower(),
        "flag_has_sublabels": bool(SUBLABELS.search(" " + raw)),
        "flag_leading_fragment": bool(re.match(r"^\s*-?\d+(\s+-?\d+)+\s+[A-Za-z]", raw)),
    }


def dedup_key(text):
    t = text.lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# Instruction/filler words that do not change what a question is about.
FILLER = set("""a an the of in on to for with vs versus between is are what how why do does
explain define discuss describe write short note notes list down out state give enlist illustrate
compare comparison differentiate distinguish difference differences detail details detailed brief
briefly suitable example examples proper necessary any one two three four various different types
type about method algorithm algorithms concept s its it can you mean by""".split())


def _content_tokens(key):
    return {t for t in key.split() if t not in FILLER and not t.isdigit()}


def is_variant(key_a, key_b):
    """True if two dedup keys differ only by filler words, numbers or spelling variants.

    Guards the fuzzy/semantic matchers against pairs like "CNN" vs "RNN" or
    "supervised" vs "unsupervised" that look alike but ask different things.
    """
    if key_a.replace(" ", "") == key_b.replace(" ", ""):
        return True
    a, b = _content_tokens(key_a), _content_tokens(key_b)
    only_a, only_b = a - b, b - a
    if "".join(sorted(only_a)) == "".join(sorted(only_b)):
        return True
    for x, others in ((only_a, only_b), (only_b, only_a)):
        for tok in x:
            partners = [o for o in others if fuzz.ratio(tok, o) >= 80 and not _negation_pair(tok, o)]
            if not partners:
                return False
    return True


def _negation_pair(a, b):
    """'supervised'/'unsupervised', 'advantages'/'disadvantages', 'linear'/'nonlinear'."""
    return any(x == p + y for p in ("un", "non", "dis", "in") for x, y in ((a, b), (b, a)))


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _group_ids(keys, pairs, prefix):
    """Union-find over unique keys; returns {key: group_id} (only for keys in a group of >= 2)."""
    uf = _UnionFind(len(keys))
    for a, b in pairs:
        uf.union(a, b)
    roots = {}
    for i in range(len(keys)):
        roots.setdefault(uf.find(i), []).append(i)
    out = {}
    for n, members in enumerate(sorted((m for m in roots.values() if len(m) > 1), key=min)):
        for i in members:
            out[keys[i]] = f"{prefix}{n:04d}"
    return out


def mark_duplicates(df, text_col, fuzzy_threshold, embeddings=None, semantic_threshold=None):
    """Add exact / near-duplicate group ids across the whole dataset (all subjects).

    exact_duplicate_group:  identical after lowercasing and stripping punctuation.
    near_duplicate_group:   exact groups merged with pairs whose fuzz.ratio >= fuzzy_threshold
                            or (if embeddings given) cosine >= semantic_threshold.
    """
    keys = df[text_col].map(dedup_key)
    uniq = sorted(set(keys))
    counts = keys.value_counts()
    exact = {k: f"EX{n:04d}" for n, k in enumerate(k for k in uniq if counts[k] > 1)}

    scores = cdist(uniq, uniq, scorer=fuzz.ratio, workers=-1)
    iu = np.triu_indices(len(uniq), k=1)
    pairs = [(a, b) for a, b in zip(*iu)
             if scores[a, b] >= fuzzy_threshold and is_variant(uniq[a], uniq[b])]
    n_fuzzy = len(pairs)
    n_sem = 0
    if embeddings is not None and semantic_threshold is not None:
        idx = {k: i for i, k in enumerate(uniq)}
        first_row = {}
        for row, k in enumerate(keys):
            first_row.setdefault(k, row)
        E = embeddings[[first_row[k] for k in uniq]]
        sim = E @ E.T
        sem = [(a, b) for a, b in zip(*iu) if sim[a, b] >= semantic_threshold
               and scores[a, b] < fuzzy_threshold and is_variant(uniq[a], uniq[b])]
        n_sem = len(sem)
        pairs += sem
    near = _group_ids(uniq, pairs, "ND")

    df = df.copy()
    df["exact_duplicate_group"] = keys.map(exact).fillna("")
    near_ids = keys.map(near)
    # Exact duplicates with no fuzzy partner still form a near-duplicate group of their own.
    df["near_duplicate_group"] = near_ids.fillna(df["exact_duplicate_group"].str.replace("EX", "NX")).fillna("")
    df["times_asked"] = df.groupby(df["near_duplicate_group"].where(df["near_duplicate_group"] != "", keys))[text_col].transform("size")
    stats = {"exact_groups": len(exact), "fuzzy_pairs": n_fuzzy, "semantic_pairs": n_sem,
             "near_duplicate_groups": int(df.loc[df.near_duplicate_group != "", "near_duplicate_group"].nunique()),
             "rows_in_exact_groups": int((df.exact_duplicate_group != "").sum()),
             "rows_in_near_groups": int((df.near_duplicate_group != "").sum())}
    return df, stats


def preprocess(df, text_col, cfg, log):
    """Add cleaned_question_text next to the untouched raw text (renamed raw_question_text)."""
    df = df.rename(columns={text_col: "raw_question_text"}).copy()
    bad = ~df["raw_question_text"].map(lambda v: isinstance(v, str))
    if bad.any():
        log.warning("  %d rows have a missing/non-text question; kept and flagged", int(bad.sum()))
    cleaned = df["raw_question_text"].map(clean_text)
    df.insert(df.columns.get_loc("raw_question_text") + 1, "cleaned_question_text", cleaned.str[0])
    df["cleaning_rules"] = cleaned.str[1].map(lambda r: ";".join(r))
    flags = pd.DataFrame([artifact_flags(t) for t in df["raw_question_text"]], index=df.index)
    df = pd.concat([df, flags], axis=1)
    df["flag_empty"] = df["cleaned_question_text"].str.len() == 0
    min_chars = cfg["preprocessing"].get("min_question_chars", 8)
    df["flag_short"] = (df["cleaned_question_text"].str.len() < min_chars) & ~df["flag_empty"]
    df["preprocessing_version"] = PREPROCESSING_VERSION

    for col in ["flag_empty", "flag_short", *flags.columns]:
        n = int(df[col].sum())
        if n:
            log.info("  %s: %d rows", col, n)
    if df["flag_empty"].any():
        log.error("Empty questions: %s", df.loc[df.flag_empty, "question_id"].tolist())
    rule_counts = df["cleaning_rules"].str.split(";").explode().value_counts()
    log.info("  cleaning rules applied: %s", rule_counts.drop("", errors="ignore").to_dict())
    return df
