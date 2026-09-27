"""Query understanding: explicit filters (subject, years, session), intent, and likely topics.

Filters are only applied when the user states them explicitly (subject code/abbreviation/full
name, years, Winter/Summer). Topic detection is soft: it selects which computed statistics to
add to the context, it does not filter retrieval.
"""
import re

import numpy as np

STOP = {"and", "of", "the", "for", "in", "to"}
INTENTS = {
    "forecast": r"\b(forecast|predict|likely|expected|upcoming|next (exam|paper|year|session)|future|important for)\b",
    "trend": r"\b(trend|trends|how often|frequen\w*|over the years|over time|increas\w*|decreas\w*|evolv\w*|recurr\w*|pattern)\b",
    "repeated": r"\b(repeat\w*|asked again|same question|rephras\w*|duplicate)\b",
}


def subject_aliases(subjects, codes):
    aliases = {}
    for s in subjects:
        words = [w for w in re.findall(r"[A-Za-z]+", s)]
        abbr = "".join(w[0].upper() for w in words if w.lower() not in STOP)
        aliases.setdefault(s, set()).add(abbr)
        if len(words) >= 3:
            aliases[s].add(s.lower())
        if codes.get(s):
            aliases[s].add(str(codes[s]))
    return aliases


class QueryAnalyzer:
    def __init__(self, taxonomy, embedder, periods, years):
        self.taxonomy = taxonomy
        self.embedder = embedder
        self.subjects = list(taxonomy["subjects"])
        codes = {s: d.get("subject_code") for s, d in taxonomy["subjects"].items()}
        self.aliases = subject_aliases(self.subjects, codes)
        self.periods, self.years = periods, sorted(years)
        self.topic_rows = [(s, t, f"{t}. {td.get('description', '')}")
                           for s, sd in taxonomy["subjects"].items() for t, td in sd["topics"].items()]
        self.topic_emb = embedder.encode([r[2] for r in self.topic_rows]) if self.topic_rows else None

    def detect_subject(self, q):
        found = []
        for s, al in self.aliases.items():
            for a in al:
                if a.isdigit() or a.isupper():
                    if re.search(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", q):
                        found.append(s)
                elif a in q.lower():
                    found.append(s)
        return sorted(set(found))

    def detect_years(self, q):
        ql = q.lower()
        ys = [int(y) for y in re.findall(r"\b(20\d\d)\b", q)]
        if m := re.search(r"last (\d+|two|three|four|five) years", ql):
            n = {"two": 2, "three": 3, "four": 4, "five": 5}.get(m.group(1), None) or int(m.group(1))
            return {"year_range": (self.years[-1] - n + 1, self.years[-1])}
        if re.search(r"\b(since|after|from)\s+20\d\d", ql) and ys:
            return {"year_range": (min(ys), self.years[-1])}
        if re.search(r"\b(between|from)\b.*\b(and|to)\b", ql) and len(ys) >= 2:
            return {"year_range": (min(ys), max(ys))}
        if ys:
            return {"year": sorted(set(ys))}
        return {}

    def detect_intent(self, q):
        ql = q.lower()
        for name, pat in INTENTS.items():
            if re.search(pat, ql):
                return name
        return "lookup"

    def detect_topics(self, q, subjects=None, k=3, min_sim=0.55):
        if self.topic_emb is None:
            return []
        v = self.embedder.encode_one(q, query=True)
        sims = self.topic_emb @ v
        out = []
        for i in np.argsort(-sims):
            s, t, _ = self.topic_rows[i]
            if subjects and s not in subjects:
                continue
            if sims[i] < min_sim:
                break
            out.append({"subject": s, "topic": t, "similarity": round(float(sims[i]), 3)})
            if len(out) == k:
                break
        return out

    def analyze(self, q):
        subjects = self.detect_subject(q)
        filters = {}
        if subjects:
            filters["subject"] = subjects
        filters.update(self.detect_years(q))
        if re.search(r"\bsummer\b", q, re.I):
            filters["exam_session"] = "Summer"
        elif re.search(r"\bwinter\b", q, re.I):
            filters["exam_session"] = "Winter"
        return {"filters": filters, "intent": self.detect_intent(q), "topics": self.detect_topics(q, subjects or None)}
