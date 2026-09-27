"""Academic Question Intelligence - Streamlit application.

    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import ui  # noqa: E402
from src.services import build_engines, load_artifacts  # noqa: E402
from src.trend_analysis.evolution import evolution_pivot, topic_shift  # noqa: E402
from src.utils.common import load_config  # noqa: E402
from src.visualization import charts  # noqa: E402
from src.visualization.charts import short  # noqa: E402

st.set_page_config(page_title="Academic Question Intelligence", page_icon=":material/school:", layout="wide")
ui.inject_css()


def _secrets_to_env():
    """On Streamlit Cloud, keys live in st.secrets; the pipeline code reads the environment."""
    import os
    try:
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "HF_TOKEN"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:  # no secrets file locally -> .env / environment are used instead
        pass


_secrets_to_env()


@st.cache_resource(show_spinner="Loading the question bank…")
def get_cfg_and_artifacts():
    cfg = load_config()
    return cfg, load_artifacts(cfg)


@st.cache_resource(show_spinner="Loading the embedding model and vector database…")
def get_engines():
    cfg, art = get_cfg_and_artifacts()
    return build_engines(cfg, art)


try:
    cfg, art = get_cfg_and_artifacts()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

Q = art["questions"]
SUBJECTS = sorted(Q["subject"].unique())
PERIODS = Q.sort_values("period_index")["period"].unique().tolist()
YEARS = sorted(Q["year"].dropna().astype(int).unique().tolist())
SUBJECT_OPTS = {s: f"{short(s)} — {s}" for s in SUBJECTS}
charts.subject_colors(SUBJECTS)

COLS = {  # human-readable column headers
    "question_text": "Question", "cleaned_question_text": "Question", "period": "Exam", "subject": "Subject",
    "primary_topic": "Topic", "subtopic": "Subtopic", "marks": "Marks", "question_number": "Q", "sub_part": "Part",
    "topic_confidence": "Confidence", "labeling_method": "Labelled by", "review_required": "Needs review",
    "times_asked": "Times asked", "question_family": "Family", "similarity": "Similarity", "tag": "Ref",
    "secondary_topics": "Also covers", "question_id": "ID",
}


def table(df, cols=None, height=None, extra=None):
    d = df[cols] if cols else df
    config = {c: COLS.get(c, c.replace("_", " ").capitalize()) for c in d.columns}
    config.update(extra or {})
    st.dataframe(d, hide_index=True, width="stretch", height=height or "auto", column_config=config)


def with_short(d):
    return d.assign(subject=d["subject"].map(short))


# ============================================================================== Dashboard
def dashboard():
    ui.page_header("Overview", "Question bank at a glance",
                   f"{len(Q)} questions from {Q['paper_id'].nunique()} GTU papers across {len(SUBJECTS)} machine-learning "
                   f"subjects, {PERIODS[0]} → {PERIODS[-1]}. Every number on this page is computed from the labelled dataset.")
    fams = art["families"]
    ui.kpis([
        (len(Q), "questions", f"{Q['paper_id'].nunique()} papers", "fact"),
        (len(PERIODS), "exam sessions", f"{YEARS[0]}–{YEARS[-1]} · Winter & Summer", "fact"),
        (Q["primary_topic"].nunique(), "syllabus topics", f"{Q['subtopic'].nunique()} subtopics", "fact"),
        (len(fams), "recurring questions", "asked in 2+ papers", "fact"),
        (f"{(~Q['review_required']).mean():.0%}", "labels auto-accepted", "semantic + LLM check", "good"),
        (int(Q["review_required"].sum()), "awaiting review", "human-in-the-loop queue", "review"),
    ])
    st.write("")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Most-examined topics")
        pick = st.segmented_control("Subject", ["All"] + SUBJECTS, default="All", key="dash_subj",
                                    format_func=lambda s: "All subjects" if s == "All" else short(s),
                                    label_visibility="collapsed")
        freq = (Q.groupby(["subject", "primary_topic"]).size().reset_index(name="questions")
                  .rename(columns={"primary_topic": "topic"}))
        if pick and pick != "All":
            freq = freq[freq["subject"] == pick]
        st.plotly_chart(charts.topic_frequency(freq, top=12), width="stretch", key="dash_freq")
    with right:
        st.subheader("How much of each exam is recycled?")
        st.caption("Share of each exam's questions that repeat (exactly or rephrased) a question from an earlier paper "
                   "of the same subject.")
        st.plotly_chart(charts.recycled_share(Q), width="stretch", key="dash_recycled")

    st.subheader("Trend signals")
    t = art["trends"]
    cols = st.columns(4)
    for col, lab in zip(cols, ["increasing", "declining", "emerging", "consistently_recurring"]):
        sub = t[t["trend"] == lab].sort_values("total_questions", ascending=False)
        name, icon, kind = ui.TREND_META[lab]
        items = "".join(f"<li>{ui.esc(x['topic'])} <span class='chip' style='--c:#6B7485'>{short(x['subject'])}</span></li>"
                        for _, x in sub.head(5).iterrows()) or "<li style='opacity:.6'>None under the configured rules</li>"
        col.markdown(f"<div class='signal'><div class='eyebrow'>{icon} {name}</div><div class='n'>{len(sub)}</div>"
                     f"<ul>{items}</ul></div>", unsafe_allow_html=True)
    st.caption("Rules: increasing/declining need slope ≥ 0.10 questions per exam and Kendall τ ≥ 0.30; "
               "a single appearance is never a trend. See Trend Analysis for details.")

    st.subheader("Most-repeated questions")
    for _, f in fams.head(6).iterrows():
        subj = [s.strip() for s in str(f["subjects"]).split(";")]
        st.markdown(f"<div class='fam'><div class='t'>{ui.esc(f['example'])}</div>"
                    + ui.chip(f"asked {int(f['times_asked'])}×", "fact", solid=True)
                    + ui.chip(f"{int(f['papers'])} papers", "fact")
                    + "".join(ui.subject_chip(s) for s in subj)
                    + f"<span class='meta' style='font-size:12px;opacity:.65'>{ui.esc(f['periods'])}</span></div>",
                    unsafe_allow_html=True)

    st.subheader("Topic map")
    st.caption("Subject → topic → subtopic. Click a segment to zoom in.")
    st.plotly_chart(charts.distribution_sunburst(Q), width="stretch", key="dash_sun")


# ============================================================================== Ask
def ask():
    ui.page_header("Explore", "Ask the question bank",
                   "Answers are generated only from retrieved historical questions and computed statistics. Every claim "
                   "cites its evidence — [E] past question, [S] computed statistic, [F] model forecast.")
    examples = ["What has been asked about the backpropagation algorithm?",
                "How often has K-means clustering been asked in ML?",
                "Which topics are likely to be important for the next AML exam?",
                "What was asked about support vector machines in AIML since 2023?",
                "Has principal component analysis been asked repeatedly?"]
    pick = st.pills("Try an example", examples, key="ask_example")
    with st.form("ask_form", border=False):
        c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
        query = c1.text_input("Your question", value=pick or "", placeholder="e.g. What has been asked about Bayes theorem?")
        submitted = c2.form_submit_button("Ask", type="primary", width="stretch")
        with st.expander("Filters · detected automatically from your question, or set them here"):
            f1, f2, f3 = st.columns(3)
            fs = f1.multiselect("Subject", SUBJECTS, format_func=SUBJECT_OPTS.get)
            fy = f2.select_slider("Years", options=YEARS, value=(YEARS[0], YEARS[-1]))
            k = f3.slider("Questions to retrieve", 3, 12, 10)
    if not (submitted and query.strip()):
        return
    eng = get_engines()
    filters = {}
    if fs:
        filters["subject"] = fs
    if fy != (YEARS[0], YEARS[-1]):
        filters["year_range"] = fy
    with st.spinner("Retrieving evidence and writing a grounded answer…"):
        res = eng["rag"].answer(query, filters=filters, top_k=k)

    a = res.get("analysis", {})
    chips = [ui.chip(f"intent · {a.get('intent', '-')}", "neutral")]
    if res.get("insufficient"):
        chips.append(ui.chip("insufficient evidence", "review", solid=True))
    elif res.get("grounded"):
        chips.append(ui.chip("✓ citations verified", "good", solid=True))
    else:
        chips.append(ui.chip("grounding warnings", "review", solid=True))
    chips.append(ui.chip(f"{res.get('provider')} {res.get('model', '')}".strip(), "neutral"))
    chips.append(ui.chip(f"best match {res.get('best_similarity', 0):.2f}", "fact"))
    for key, val in (a.get("filters") or {}).items():
        chips.append(ui.chip(f"{key}: {', '.join(map(str, val)) if isinstance(val, (list, tuple)) else val}", "fact"))
    st.markdown("".join(chips), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(res["answer"])
    for w in res.get("warnings", []):
        st.warning(w, icon=":material/warning:")

    ev_tab, st_tab, ctx_tab = st.tabs([f"Evidence ({len(res.get('evidence', []))})",
                                       f"Statistics & forecasts ({len(res.get('statistics', [])) + len(res.get('forecasts', []))})",
                                       "Full context"])
    with ev_tab:
        if not res.get("evidence"):
            st.caption("No questions retrieved.")
        cols = st.columns(2, gap="medium")
        for i, e in enumerate(res.get("evidence", [])):
            with cols[i % 2]:
                ui.evidence_card(e)
    with st_tab:
        if res.get("statistics"):
            st.markdown(ui.chip("computed from the full dataset", "fact"), unsafe_allow_html=True)
            table(pd.DataFrame(res["statistics"]).drop(columns=["counts_by_period"]))
        if res.get("forecasts"):
            ui.callout("forecast", "Model estimates, not facts.", "Forecasts come from historical frequency patterns.")
            table(pd.DataFrame(res["forecasts"]))
        if not res.get("statistics") and not res.get("forecasts"):
            st.caption("No statistics were needed for this question.")
    with ctx_tab:
        st.code(res.get("context", ""), language="text", wrap_lines=True)


# ============================================================================== Trends
def trends():
    ui.page_header("Analyse", "Topic trends over time",
                   "Question counts per exam session (one paper per subject per session). Trend labels follow explicit "
                   "statistical rules — a topic appearing once is never called a trend.")
    c1, c2 = st.columns([1, 2], gap="large")
    subject = c1.selectbox("Subject", SUBJECTS, format_func=SUBJECT_OPTS.get)
    pr = c2.select_slider("Exam range", options=PERIODS, value=(PERIODS[0], PERIODS[-1]))
    sel = PERIODS[PERIODS.index(pr[0]):PERIODS.index(pr[1]) + 1]
    tp = art["topic_period"]
    tp = tp[(tp["subject"] == subject) & tp["period"].isin(sel)]
    tsum = art["trends"][art["trends"]["subject"] == subject].sort_values("total_questions", ascending=False)

    counts = tsum["trend"].value_counts()
    ui.kpis([(len(tsum), "topics tracked", f"{int(tsum['total_questions'].sum())} questions", "fact"),
             (int(counts.get("consistently_recurring", 0)), "consistently recurring", "≥ 80% of exams", "fact"),
             (int(counts.get("increasing", 0)), "increasing", "positive slope + Kendall τ", "good"),
             (int(counts.get("declining", 0)), "declining", "negative slope + Kendall τ", "review")])
    st.write("")
    topics = st.multiselect("Topics to plot", tsum["topic"].tolist(), default=tsum["topic"].head(5).tolist())
    tab1, tab2, tab3 = st.tabs(["Trend lines", "Evolution heatmap", "Dominant topic per exam"])
    with tab1:
        st.plotly_chart(charts.trend_lines(tp[tp["topic"].isin(topics)] if topics else tp), width="stretch", key="tr_lines")
    with tab2:
        st.plotly_chart(charts.evolution_heatmap(evolution_pivot(tp, subject)), width="stretch", key="tr_heat")
    with tab3:
        evo = art["evolution"]
        table(evo[(evo["subject"] == subject) & evo["period"].isin(sel)], ["period", "dominant_topics", "questions_in_paper"])

    st.subheader("Topic frequency and trend classification")
    show = tsum.assign(trend=tsum["trend"].map(ui.trend_label))
    table(show, ["topic", "total_questions", "share_of_subject", "periods_appeared", "recent_questions", "trend",
                 "trend_reason", "marks_mix", "supporting_periods"], extra={
        "share_of_subject": st.column_config.ProgressColumn("Share of subject", min_value=0,
                                                            max_value=float(show["share_of_subject"].max() or 1), format="percent"),
        "total_questions": "Questions", "periods_appeared": "Exams present", "recent_questions": "Last 3 exams",
        "trend": "Trend", "trend_reason": "Why", "marks_mix": "Marks mix", "supporting_periods": "Appeared in"})

    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Topic mix shift")
        st.caption("Share of the subject's questions: first half vs second half of the exam range.")
        shift = topic_shift(art["topic_period"], subject).reset_index().rename(columns={"index": "topic"})
        table(shift, extra={"early_share": st.column_config.NumberColumn("Early share", format="percent"),
                            "late_share": st.column_config.NumberColumn("Late share", format="percent"),
                            "change": st.column_config.NumberColumn("Change", format="percent")})
    with b:
        st.subheader("Subtopic drill-down")
        if topics:
            topic = st.selectbox("Topic", topics)
            subs = Q[(Q["subject"] == subject) & (Q["primary_topic"] == topic)]["subtopic"].unique()
            sp = art["subtopic_period"]
            sp = sp[(sp["subject"] == subject) & sp["topic"].isin(subs) & sp["period"].isin(sel)]
            st.plotly_chart(charts.evolution_heatmap(evolution_pivot(sp, subject)), width="stretch", key="tr_sub")
    with st.expander("Trend rules (config.yaml → trends)"):
        st.json(cfg["trends"])


# ============================================================================== Forecast
def forecast():
    fs = art["forecast_summary"]
    ui.page_header("Analyse", "Forecast for the next exam",
                   "Which topics historical patterns suggest will carry weight in the next paper.")
    ui.callout("forecast", "Forecasts are model estimates, not facts or guarantees.",
               f"Method chosen by rolling-origin backtesting: {fs.get('selected_method', '?').replace('_', ' ')} "
               f"(trained only on earlier exams, scored on the next). Historical facts are shown in navy, estimates in amber.")
    subject = st.segmented_control("Subject", SUBJECTS, default=SUBJECTS[0], format_func=short, key="fc_subj") or SUBJECTS[0]
    fc = art["forecast"][art["forecast"]["subject"] == subject].sort_values("predicted_questions", ascending=False)
    conf = fc["forecast_confidence"].value_counts()
    ui.kpis([(fc["forecast_period"].iloc[0], "forecast exam", subject, "forecast"),
             (f"{fc['predicted_questions'].sum():.0f}", "questions estimated", "sum across topics", "forecast"),
             (int(conf.get("High", 0)), "high-confidence topics", "regular + low backtest error", "good"),
             (int(conf.get("Low", 0)), "low-confidence topics", "rare or erratic history", "review")])
    st.write("")
    st.plotly_chart(charts.forecast_bars(fc, top=15), width="stretch", key="fc_bars")

    st.subheader("Topic-by-topic evidence")
    table(fc.assign(forecast_confidence=fc["forecast_confidence"].map({"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}),
                    trend=fc["trend"].map(ui.trend_label)),
          ["topic", "historical_frequency", "recent_frequency", "trend", "predicted_questions", "forecast_score",
           "forecast_confidence", "evidence"], extra={
              "historical_frequency": "Asked (total)", "recent_frequency": "Last 3 exams", "trend": "Trend",
              "predicted_questions": st.column_config.NumberColumn("Expected next exam", format="%.1f"),
              "forecast_score": st.column_config.ProgressColumn("P(≥1 question)", min_value=0, max_value=1, format="%.2f"),
              "forecast_confidence": "Confidence", "evidence": st.column_config.TextColumn("Evidence", width="large")})

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.subheader("History vs forecast")
        topic = st.selectbox("Topic", fc["topic"].tolist())
        h = art["topic_period"]
        h = h[(h["subject"] == subject) & (h["topic"] == topic)].sort_values("period_index")
        row = fc[fc["topic"] == topic].iloc[0]
        fig = go.Figure([
            go.Bar(x=h["period"], y=h["questions"], name="Historical (fact)", marker_color=charts.FACT),
            go.Bar(x=[row["forecast_period"]], y=[row["predicted_questions"]], name="Forecast (estimate)",
                   marker=dict(color=charts.FORECAST, pattern_shape="/"))])
        fig.update_layout(template=charts.TEMPLATE, height=320, margin=dict(l=8, r=8, t=10, b=8), yaxis_title="Questions",
                          xaxis_type="category", legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig, width="stretch", key="fc_hist")
    with c2:
        st.subheader("How reliable is this?")
        if fs.get("metrics"):
            m = pd.DataFrame(fs["metrics"])
            st.plotly_chart(charts.backtest_chart(m), width="stretch", key="fc_bt")
            st.caption(f"Backtest over exams {', '.join(fs.get('backtest_origins', []))}. With only 10 exams per subject "
                       "no method beats the long-run average — topic weights are largely set by the syllabus.")


# ============================================================================== Similar
def similar():
    ui.page_header("Explore", "Find similar questions",
                   "Semantic search over every past question — catches rephrased versions that keyword search misses.")
    with st.form("sim_form", border=False):
        q = st.text_area("Question", "Explain the concept of normalization of features and why it is needed.", height=90)
        c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="bottom")
        fs = c1.multiselect("Subject", SUBJECTS, format_func=SUBJECT_OPTS.get)
        k = c2.slider("Results", 5, 30, 10)
        go_ = c3.form_submit_button("Search", type="primary", width="stretch")
    if go_ and q.strip():
        eng = get_engines()
        hits = eng["retriever"].retrieve(q, top_k=k, filters={"subject": fs} if fs else None, mode="similar")
        thr = cfg["similarity"]
        d = pd.DataFrame(hits)
        if d.empty:
            st.info("No results.")
        else:
            d["relation"] = d["similarity"].map(lambda s: "rephrased / near-identical" if s >= thr["rephrased_threshold"]
                                                else "semantically similar" if s >= thr["repeated_question_threshold"] else "related")
            table(with_short(d), ["similarity", "relation", "period", "subject", "question_text", "primary_topic", "marks"], extra={
                "similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.3f"),
                "relation": "Relation", "question_text": st.column_config.TextColumn("Question", width="large")})

    st.divider()
    st.subheader("Known repeats of a past question")
    qtext = Q.set_index("question_id")["cleaned_question_text"]
    qid = st.selectbox("Pick a question from the dataset", Q["question_id"],
                       format_func=lambda i: f"{i} — {qtext[i][:95]}")
    p = art["pairs"]
    rel = p[(p["question_id"] == qid) | (p["similar_question_id"] == qid)].copy()
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        if rel.empty:
            st.caption("No related questions above the similarity threshold in other papers.")
        else:
            mine = rel["question_id"] == qid
            rel["question"] = rel["similar_question_text"].where(mine, rel["question_text"])
            rel["exam"] = rel["similar_period"].where(mine, rel["period"])
            rel["relationship_type"] = rel["relationship_type"].str.replace("_", " ")
            table(rel.sort_values("similarity_score", ascending=False),
                  ["relationship_type", "similarity_score", "exam", "question"], extra={
                      "relationship_type": "Relation",
                      "similarity_score": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.2f"),
                      "exam": "Exam", "question": st.column_config.TextColumn("Question", width="large")})
    with c2:
        st.plotly_chart(charts.repeated_summary(p, title="All related pairs in the dataset"), width="stretch", key="sim_rep")


# ============================================================================== Explorer
def explorer():
    ui.page_header("Explore", "Question explorer", "Browse, filter and export the labelled question bank.")
    with st.container(border=True):
        c = st.columns(5)
        fs = c[0].multiselect("Subject", SUBJECTS, format_func=short)
        fp = c[1].multiselect("Exam", PERIODS)
        d = Q[Q["subject"].isin(fs)] if fs else Q
        ft = c[2].multiselect("Topic", sorted(d["primary_topic"].unique()))
        d = d[d["primary_topic"].isin(ft)] if ft else d
        fsub = c[3].multiselect("Subtopic", sorted(d["subtopic"].unique()))
        fm = c[4].multiselect("Marks", sorted(Q["marks"].dropna().astype(int).unique().tolist()))
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        text = c1.text_input("Search text", placeholder="e.g. kernel, entropy, gradient")
        only_review = c2.toggle("Only labels needing review")
    if fp:
        d = d[d["period"].isin(fp)]
    if fsub:
        d = d[d["subtopic"].isin(fsub)]
    if fm:
        d = d[d["marks"].isin(fm)]
    if text:
        d = d[d["cleaned_question_text"].str.contains(text, case=False, regex=False)]
    if only_review:
        d = d[d["review_required"]]
    l, r = st.columns([4, 1], vertical_alignment="center")
    l.markdown(ui.chip(f"{len(d)} questions", "fact", solid=True) + ui.chip(f"{d['paper_id'].nunique()} papers", "fact"),
               unsafe_allow_html=True)
    r.download_button("Download CSV", d.to_csv(index=False).encode("utf-8-sig"), "questions_filtered.csv", "text/csv",
                      icon=":material/download:", width="stretch")
    table(with_short(d), ["period", "subject", "question_number", "sub_part", "marks", "cleaned_question_text", "primary_topic",
                          "subtopic", "topic_confidence", "labeling_method", "review_required", "times_asked"], height=560, extra={
        "topic_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
        "cleaned_question_text": st.column_config.TextColumn("Question", width="large"),
        "review_required": st.column_config.CheckboxColumn("Needs review")})


# ============================================================================== Review
def review():
    ui.page_header("Quality", "Human review queue",
                   "Topic labels the system was unsure about. Your decisions are applied on the next pipeline run, to every "
                   "copy of the question, and corrected questions become examples that improve the classifier.")
    rv = art["review"]
    if rv.empty:
        st.success("Nothing to review.", icon=":material/task_alt:")
        return
    rv = rv.fillna("")
    status = rv["review_status"].str.lower()
    ui.kpis([(len(rv), "questions in queue", "unique question texts", "review"),
             (int((status == "pending").sum()), "pending", "", "review"),
             (int((status == "approved").sum()), "approved", "", "good"),
             (int((status == "corrected").sum()), "corrected", "", "good")])
    st.write("")
    tax = art["taxonomy"]["subjects"]
    all_topics = sorted({t for s in tax.values() for t in s["topics"]})
    all_subs = sorted({x for s in tax.values() for t in s["topics"].values() for x in t["subtopics"]})
    editable = ["review_status", "corrected_topic", "corrected_subtopic", "reviewer_notes"]
    edited = st.data_editor(
        rv[["question_id", "subject", "question_text", "predicted_topic", "predicted_subtopic", "confidence",
            "suggested_alternatives", "llm_suggestion", "review_reason"] + editable],
        hide_index=True, width="stretch", height=520, disabled=[c for c in rv.columns if c not in editable],
        column_config={"review_status": st.column_config.SelectboxColumn("Decision", options=["pending", "approved", "corrected"]),
                       "corrected_topic": st.column_config.SelectboxColumn("Corrected topic", options=[""] + all_topics),
                       "corrected_subtopic": st.column_config.SelectboxColumn("Corrected subtopic", options=[""] + all_subs),
                       "reviewer_notes": "Notes", "question_text": st.column_config.TextColumn("Question", width="large"),
                       "predicted_topic": "Predicted topic", "predicted_subtopic": "Predicted subtopic",
                       "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                       "suggested_alternatives": "Alternatives", "llm_suggestion": "LLM suggestion", "review_reason": "Why flagged",
                       "question_id": "ID", "subject": "Subject"})
    if st.button("Save decisions", type="primary", icon=":material/save:"):
        full = art["review"].fillna("").set_index("question_id")
        for _, r in edited.iterrows():
            for c in editable:
                full.at[r["question_id"], c] = r[c]
        full.reset_index().to_csv(cfg["paths"]["review_csv"], index=False, encoding="utf-8-sig")
        st.success("Saved. Run `python pipeline.py` to apply the decisions.", icon=":material/task_alt:")
        st.cache_resource.clear()


# ============================================================================== Evaluation
def evaluation():
    ui.page_header("Quality", "Evaluation",
                   "How well each component works on this dataset. Human-verified and automatic (silver-standard) results "
                   "are kept separate.")
    ev = art["evaluation"]
    if not ev:
        st.info("No evaluation yet — run `python pipeline.py`.")
        return
    rt, rg, fcs = ev.get("retrieval", {}), ev.get("rag", {}), ev.get("forecasting", {})
    para = rt.get("paraphrase_set", {}).get("semantic_faiss", {})
    det = rg.get("deterministic", {})
    ui.kpis([(f"{para.get('mrr', 0):.2f}", "retrieval MRR", "reworded queries (silver)", "fact"),
             (f"{para.get('hit_rate@10', 0):.0%}", "hit rate @10", "reworded queries", "fact"),
             (f"{det.get('refusal_accuracy', 0):.0%}", "correct refusals", "out-of-scope handled", "good"),
             (f"{det.get('grounded_rate', 0):.0%}", "grounded answers", "valid citations only", "good"),
             (f"{min(m['mae'] for m in fcs.get('metrics', [{'mae': 0}])):.2f}", "forecast MAE",
              f"{fcs.get('selected_method', '-').replace('_', ' ')} · backtest", "forecast")])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Classification", "Retrieval", "RAG", "Forecasting"])
    with t1:
        cm = ev.get("classification_manual", {})
        if cm.get("status") == "ok":
            ui.kpis([(f"{cm['topic']['accuracy']:.3f}", "topic accuracy", "human-verified", "good"),
                     (f"{cm['topic']['macro_f1']:.3f}", "macro F1", "", "good"),
                     (f"{cm['subtopic']['accuracy']:.3f}", "subtopic accuracy", "", "good"),
                     (cm["evaluated_rows"], "verified questions", "", "fact")])
        else:
            ui.callout("review", "Human-verified accuracy not available yet.",
                       "Fill the 'verified' column in data/review/validation_sample.csv, then run "
                       "`python pipeline.py --stage evaluation`. No accuracy is claimed until then.")
        cr = ev.get("classification_llm_reference", {})
        if cr.get("status") == "ok":
            st.markdown("**Automatic check** — agreement with a blind LLM reference labeller " + ui.chip("silver, not ground truth", "forecast"),
                        unsafe_allow_html=True)
            d = (pd.DataFrame(cr["by_confidence_band"]).T.reindex(["high", "medium", "low"]).dropna(how="all")
                   .reset_index(names="band"))
            table(d, extra={"band": "Confidence band", "n": "Questions",
                            "topic_agreement": st.column_config.ProgressColumn("Topic agreement", min_value=0, max_value=1, format="%.2f"),
                            "subtopic_agreement": st.column_config.ProgressColumn("Subtopic agreement", min_value=0, max_value=1, format="%.2f")})
            st.caption("Agreement falls from the high to the low band, as intended: confidence tracks correctness.")
    with t2:
        for name in ("paraphrase_set", "repeat_set"):
            blk = rt.get(name, {})
            if "semantic_faiss" in blk:
                st.markdown(f"**{name.replace('_', ' ').capitalize()}** — {blk['description']}")
                rows = []
                for sys_, lab in (("semantic_faiss", "Semantic (FAISS)"), ("tfidf_baseline", "TF-IDF baseline")):
                    m = blk[sys_]
                    rows.append({"system": lab, "MRR": m["mrr"], "P@1": m["precision@1"], "Recall@5": m["recall@5"],
                                 "Hit@5": m["hit_rate@5"], "Hit@10": m["hit_rate@10"], "queries": m["n_queries"]})
                table(pd.DataFrame(rows).round(3))
    with t3:
        if rg:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Deterministic checks**")
                table(pd.DataFrame([{"check": k.replace("_", " "), "value": v} for k, v in det.items()]))
            with c2:
                st.markdown("**Automatic LLM judge** " + ui.chip("silver", "forecast"), unsafe_allow_html=True)
                j = {k: v for k, v in rg.get("llm_judge", {}).items() if k != "type"}
                table(pd.DataFrame([{"metric": k.replace("_", " "), "value": v} for k, v in j.items()]))
            table(pd.DataFrame(rg["per_query"]), ["id", "query", "insufficient", "expect_insufficient", "best_similarity",
                                                  "context_keyword_precision", "citation_coverage", "grounded"])
    with t4:
        if fcs.get("metrics"):
            m = pd.DataFrame(fcs["metrics"])
            st.plotly_chart(charts.backtest_chart(m), width="stretch", key="ev_bt")
            table(m.round(3))
    st.caption(f"Full report: {cfg['paths']['evaluation_html'].relative_to(ROOT).as_posix()}")


# ============================================================================== Navigation
PAGES = {
    "Overview": [st.Page(dashboard, title="Dashboard", icon=":material/space_dashboard:", url_path="dashboard", default=True)],
    "Explore": [st.Page(ask, title="Ask the question bank", icon=":material/forum:", url_path="ask"),
                st.Page(similar, title="Similar questions", icon=":material/join_inner:", url_path="similar"),
                st.Page(explorer, title="Question explorer", icon=":material/table_view:", url_path="explorer")],
    "Analyse": [st.Page(trends, title="Trend analysis", icon=":material/trending_up:", url_path="trends"),
                st.Page(forecast, title="Forecast", icon=":material/online_prediction:", url_path="forecast")],
    "Quality": [st.Page(review, title="Review queue", icon=":material/rule:", url_path="review"),
                st.Page(evaluation, title="Evaluation", icon=":material/fact_check:", url_path="evaluation")],
}

st.logo(str(ROOT / "app" / "static" / "logo.svg"), size="large", icon_image=str(ROOT / "app" / "static" / "icon.svg"))
nav = st.navigation(PAGES, position="sidebar")

with st.sidebar:
    st.caption(f"GTU machine-learning papers · {PERIODS[0]}–{PERIODS[-1]}")
    st.divider()
    st.markdown("<div class='eyebrow'>Colour key</div>", unsafe_allow_html=True)
    ui.sidebar_legend()
    st.markdown("<div class='eyebrow' style='margin-top:1rem'>Subjects</div>", unsafe_allow_html=True)
    st.markdown("".join(f"<div class='legend-row'>{ui.subject_chip(s)} {ui.esc(s)}</div>" for s in SUBJECTS),
                unsafe_allow_html=True)
    st.caption(f"Embeddings `{cfg['embedding']['model']}` · LLM `{cfg['llm']['provider']}`")

nav.run()
