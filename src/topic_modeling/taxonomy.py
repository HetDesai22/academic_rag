"""Stage 3: build, validate and save the Subject -> Topic -> Subtopic taxonomy.

Source of truth is config/taxonomy_source.yaml (derived from the official GTU syllabi in
data/syllabus). This module validates it against the dataset, checks coverage with
unsupervised clustering of the questions (topic discovery), and writes
data/final/topic_taxonomy.json.
"""
from collections import Counter

import numpy as np
import yaml

from src.utils.common import ROOT, now_iso, sha256_file, write_json

SOURCE_PATH = ROOT / "config" / "taxonomy_source.yaml"


def load_source(path=SOURCE_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_source(src, dataset_subjects, log):
    """Structural checks. Raises on errors that would make labeling ambiguous."""
    errors, warnings = [], []
    subjects = src["subjects"]
    for s in dataset_subjects:
        if s not in subjects:
            warnings.append(f"Dataset subject '{s}' has no syllabus taxonomy; a draft will be discovered from clusters")
    for s in subjects:
        if s not in dataset_subjects:
            warnings.append(f"Taxonomy subject '{s}' not in dataset")
    for s, sdata in subjects.items():
        syllabus = ROOT / "data" / "syllabus" / f"{sdata['subject_code']}.pdf"
        if not syllabus.exists():
            warnings.append(f"{s}: syllabus file {syllabus.name} missing")
        seen_sub = Counter()
        units = [t.get("syllabus_unit") for t in sdata["topics"].values() if t.get("syllabus_unit") is not None]
        if len(set(units)) != len(units):
            errors.append(f"{s}: duplicate syllabus_unit numbers {units}")
        for t, tdata in sdata["topics"].items():
            if not tdata.get("subtopics"):
                errors.append(f"{s} / {t}: no subtopics")
            for st, stdata in tdata["subtopics"].items():
                seen_sub[st] += 1
                if not stdata.get("description"):
                    errors.append(f"{s} / {t} / {st}: missing description")
                bad = [k for k in stdata.get("keywords", []) if not isinstance(k, str) or not k.strip()]
                if bad:
                    errors.append(f"{s} / {t} / {st}: non-string keywords {bad} (quote YAML literals)")
        for st, n in seen_sub.items():
            if n > 1:
                errors.append(f"{s}: subtopic '{st}' appears under {n} topics (ambiguous)")
    for w in warnings:
        log.warning("taxonomy: %s", w)
    if errors:
        for e in errors:
            log.error("taxonomy: %s", e)
        raise ValueError(f"Taxonomy has {len(errors)} error(s); see log")
    return warnings


def flatten(src):
    """One row per subtopic: dict(subject, topic, subtopic, description, keywords, text)."""
    rows = []
    for s, sdata in src["subjects"].items():
        for t, tdata in sdata["topics"].items():
            for st, stdata in tdata["subtopics"].items():
                kws = stdata.get("keywords", [])
                rows.append({
                    "subject": s, "topic": t, "subtopic": st,
                    "description": stdata["description"], "keywords": kws,
                    # Prototype text embedded for semantic matching.
                    "text": f"{st}. {stdata['description']}",
                })
    return rows


def discover_clusters(question_emb, texts, subtopic_rows, subtopic_emb, cfg, log):
    """Unsupervised topic discovery for one subject, used to validate taxonomy coverage.

    Clusters the questions (UMAP + HDBSCAN), then reports for each cluster its representative
    questions and the closest taxonomy subtopic. Clusters with weak best-match similarity point
    to missing taxonomy entries; subtopics that attract no question are reported as unused.
    """
    import hdbscan
    import umap

    dcfg = cfg["discovery"]
    seed = cfg["random_seed"]
    reducer = umap.UMAP(n_neighbors=dcfg["umap_neighbors"], n_components=dcfg["umap_components"],
                        metric="cosine", random_state=seed)
    reduced = reducer.fit_transform(question_emb)
    labels = hdbscan.HDBSCAN(min_cluster_size=dcfg["min_cluster_size"], metric="euclidean").fit_predict(reduced)

    sims = question_emb @ subtopic_emb.T
    nearest = sims.argmax(1)
    clusters = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        centroid = question_emb[idx].mean(0)
        centroid /= np.linalg.norm(centroid)
        reps = idx[np.argsort(-(question_emb[idx] @ centroid))[:5]]
        csim = centroid @ subtopic_emb.T
        best = int(csim.argmax())
        votes = Counter(subtopic_rows[i]["subtopic"] for i in nearest[idx])
        clusters.append({
            "cluster": int(c), "size": int(len(idx)),
            "representative_questions": [texts[i] for i in reps],
            "closest_subtopic": f"{subtopic_rows[best]['topic']} / {subtopic_rows[best]['subtopic']}",
            "closest_similarity": round(float(csim[best]), 3),
            "member_nearest_subtopics": dict(votes.most_common(3)),
        })
    used = Counter(subtopic_rows[i]["subtopic"] for i in nearest)
    unused = [f"{r['topic']} / {r['subtopic']}" for r in subtopic_rows if used[r["subtopic"]] == 0]
    report = {
        "n_questions": int(len(texts)),
        "n_clusters": len(clusters),
        "noise_points": int((labels == -1).sum()),
        "clusters": sorted(clusters, key=lambda c: c["closest_similarity"]),
        "subtopics_without_nearest_question": unused,
    }
    log.info("  discovery: %d clusters, %d noise, weakest cluster match %.3f, %d unused subtopics",
             len(clusters), report["noise_points"],
             min((c["closest_similarity"] for c in clusters), default=float("nan")), len(unused))
    return report


def draft_taxonomy_from_clusters(subject, texts, emb, cfg, log, llm=None):
    """Fallback when a subject has no syllabus taxonomy: discover candidate topics.

    Agglomerative clustering on question embeddings; each cluster is named from its most
    distinctive terms (class-based TF-IDF), optionally refined by an LLM from representative
    questions. The result is a DRAFT for human review, not trusted blindly.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_clusters = max(2, min(12, len(texts) // 15))
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(emb)
    docs = [" ".join(t for t, l in zip(texts, labels) if l == c) for c in range(n_clusters)]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    topics = {}
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        top_terms = terms[X[c].toarray()[0].argsort()[::-1][:6]].tolist()
        centroid = emb[idx].mean(0)
        reps = [texts[i] for i in idx[np.argsort(-(emb[idx] @ centroid))[:5]]]
        name = " / ".join(t.title() for t in top_terms[:2])
        if llm is not None and getattr(llm, "available", False):
            try:
                out = llm.generate(
                    "You name clusters of exam questions with a short syllabus-style topic name (2-5 words).",
                    "Representative questions:\n- " + "\n- ".join(reps) + f"\nDistinctive terms: {', '.join(top_terms)}",
                    json_schema={"type": "object", "properties": {"name": {"type": "string"}},
                                 "required": ["name"], "additionalProperties": False}, max_tokens=200)
                name = out["json"]["name"].strip() or name
            except Exception as e:  # naming is optional; keep the TF-IDF name
                log.warning("LLM cluster naming failed: %s", e)
        base, k = name, 2
        while name in topics:
            name, k = f"{base} ({k})", k + 1
        topics[name] = {"syllabus_unit": None, "description": "Discovered cluster: " + "; ".join(reps[:3]),
                        "subtopics": {name: {"description": " ".join(reps[:3]), "keywords": top_terms[:4]}}}
    log.warning("%s: no syllabus taxonomy - drafted %d topics from clusters (review before trusting)",
                subject, len(topics))
    return {"subject_code": "", "topics": topics, "discovered": True}


def build_taxonomy_json(src, discovery=None):
    out = {
        "taxonomy_version": src["version"],
        "generated_at": now_iso(),
        "source_file": SOURCE_PATH.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(SOURCE_PATH),
        "method": ("Topics = official GTU syllabus units; subtopics = syllabus items refined to the "
                   "concepts recurring in past papers; coverage validated by embedding clustering."),
        "subjects": {},
    }
    for s, sdata in src["subjects"].items():
        code = sdata["subject_code"]
        subj = {
            "subject_code": code,
            "taxonomy_origin": "discovered_clusters (draft)" if sdata.get("discovered") else "official_syllabus",
            "syllabus_file": f"data/syllabus/{code}.pdf" if code and not sdata.get("discovered") else None,
            "topics": {},
        }
        for t, tdata in sdata["topics"].items():
            subj["topics"][t] = {
                "syllabus_unit": tdata.get("syllabus_unit"),
                "description": tdata.get("description", ""),
                "subtopics": {
                    st: {"description": d["description"], "keywords": d.get("keywords", [])}
                    for st, d in tdata["subtopics"].items()
                },
            }
        out["subjects"][s] = subj
    # Compact Subject -> Topic -> [Subtopics] view for quick reading.
    out["hierarchy"] = {
        s: {t: list(td["subtopics"]) for t, td in sd["topics"].items()}
        for s, sd in out["subjects"].items()
    }
    return out
