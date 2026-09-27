# Academic Question Intelligence using RAG and Contextual Embeddings

This system analyses historical university examination questions. It labels every question with its syllabus topic, makes the question bank searchable by meaning, answers questions about it with cited evidence (RAG), measures how topics trend over time, detects repeated and rephrased questions, and forecasts which topics are likely to matter in the next exam. It ships with an interactive dashboard and a full evaluation.

It runs on the real dataset in this repository: **1,000 descriptive questions** from **40 GTU papers**. That's 4 machine-learning subjects × 10 exam sessions, from Winter 2021 to Summer 2026.

---

## Problem statement

Students and teachers preparing for university exams rely on past papers, but the papers are unstructured PDFs:
- The same concept is asked in many different wordings.
- Topics are spread over several related subjects.
- It is hard to tell which topics recur, which are rising or fading, and where the evidence is.

This project turns the papers into a structured, searchable and analysable knowledge base. It answers questions **only from historical evidence**, and it keeps observed facts, statistical trends and forecasts clearly separate.

## Objectives

1. Extract, clean and structure historical questions, keeping the raw text and all provenance.
2. Label every question with **subject → topic → subtopic**, a confidence score and human review for uncertain cases.
3. Build contextual embeddings and a filterable **FAISS** vector database.
4. Provide grounded **RAG** answers with verifiable source citations.
5. Analyse topic frequency, evolution and statistically defined trends.
6. Detect exact, near-identical, rephrased and semantically similar questions.
7. Forecast next-exam topic relevance, with uncertainty and backtesting.
8. Evaluate retrieval, classification, generation and forecasting honestly.
9. Demonstrate everything in a Streamlit application.

## Architecture

```text
Historical papers (PDF, data/raw)
      ↓  src/ingestion/extract_questions.py   layout-aware PDF parsing (marks/label columns)
Question CSV (data/processed/questions.csv, read-only)
      ↓  src/ingestion                        column detection → standardized schema → profiling
Cleaning                                      src/preprocessing  (raw + cleaned text, artifact flags, duplicates)
      ↓
Topic classification                          src/topic_modeling (syllabus taxonomy + clustering check,
      ↓                                        semantic classifier, confidence, LLM check, human review)
Contextual embeddings                         src/embeddings     (sentence-transformers, per-text cache, GPU/CPU)
      ↓
Vector database                               src/retrieval      (FAISS IndexFlatIP + filterable metadata, incremental)
      ↓
Semantic retrieval → RAG                      src/rag            (query understanding, grounded prompt, citation checks)
      ↓
Trend analysis & repeated questions           src/trend_analysis (frequency, Kendall/OLS trend rules, evolution, repeats)
      ↓
Forecasting                                   src/forecasting    (baseline, statistical, Random Forest; backtest selection)
      ↓
Evaluation & reports                          src/evaluation, src/visualization
      ↓
Interactive dashboard                         app/app.py (Streamlit)
```

```text
academic_rag/
├── pipeline.py            one command for the full pipeline
├── config.yaml            every model, path and threshold
├── .env.example           API keys (copy to .env)
├── app/app.py             Streamlit application
├── config/taxonomy_source.yaml   syllabus-based taxonomy (human-editable, versioned)
├── data/raw/              original PDFs          data/syllabus/  official GTU syllabi
├── data/processed/        questions.csv (original dataset, never modified)
├── data/incoming/         new papers for incremental updates
├── data/final/            questions_cleaned.csv, questions_labeled.csv, questions.json, topic_taxonomy.json
├── data/review/           topic_review.csv (human review), validation_sample.csv (ground-truth sample)
├── models/embedding_cache/  cached embeddings (per text, per model)
├── vector_db/             index.faiss, metadata.jsonl, manifest.json
├── reports/               profiles, trends, repeats, forecasts, evaluation (JSON/CSV/HTML), figures/
├── evaluation/            evaluation query sets (RAG queries, generated paraphrase set, reference labels)
├── src/                   ingestion · preprocessing · topic_modeling · embeddings · retrieval · rag ·
│                          trend_analysis · forecasting · evaluation · visualization · utils
└── tests/                 55 unit/integration tests
```

## Models used

| Purpose | Model | Notes |
|---|---|---|
| Contextual embeddings | `BAAI/bge-base-en-v1.5` (768-d) | Configurable (`embedding.model`), e.g. `sentence-transformers/all-MiniLM-L6-v2`. Runs on CUDA if available, otherwise the CPU. |
| Topic discovery | UMAP + HDBSCAN (coverage check); agglomerative clustering + class-TF-IDF (automatic taxonomy draft when a subject has no syllabus) | |
| Topic check, RAG generation, automatic judge | `claude-opus-5` via the Anthropic SDK | Configurable provider: `anthropic`, `openai_compatible` (OpenAI / Ollama / LM Studio / vLLM), `huggingface_local` (tested with `Qwen/Qwen2.5-0.5B-Instruct` on GPU), or `none` (extractive answers). |
| Forecasting | historical mean, moving average, exponential smoothing, linear trend, seasonal mean, pooled Random Forest | Selected automatically by rolling-origin backtest. No deep learning: the series have only 10 points. |

## How it works

**Topic classification.**
- **Taxonomy.** Topics are the official GTU syllabus units (PDFs in `data/syllabus/`). Subtopics are the syllabus items, refined into the concepts that recur in the papers. Each subtopic has a description and keywords. That gives 34 topics and 151 subtopics across 4 subjects.
- **Coverage check.** Embedding clustering (UMAP + HDBSCAN) checks the taxonomy's coverage and flags weak or missing entries (`reports/cluster_discovery.json`). A subject without a syllabus gets an automatically discovered draft taxonomy instead.
- **Classification.** Each question's embedding is compared with every subtopic prototype of its own subject (cosine similarity). A literal keyword match adds only a small bonus, and the most specific keyword wins.
  - Topic score = the best subtopic score in that topic.
  - Confidence = a softmax over the topic scores.
  - Questions far from every subtopic (likely off-syllabus) have their confidence capped.
- **Secondary topics.** Multi-part questions get secondary topics only when a sub-part is confidently about another topic *and* names it.
- **Routing:**
  - High confidence (≥ 0.75): accepted.
  - Medium: checked by Claude, which must choose from the taxonomy (JSON-schema constrained). A disagreement is adopted only if Claude is confident and its choice is in the classifier's top 3; otherwise it goes to review.
  - Low: human review, with Claude's suggestion shown as a hint.
- **Human review.** Decisions are made in `data/review/topic_review.csv`, or in the app's *Review Queue*. They are applied on the next run and to every copy of a repeated question, and corrected questions become extra classifier prototypes.

**Embeddings.**
- Every cleaned question is embedded once and cached, keyed by model and text.
- Queries use the bge retrieval instruction; question-to-question comparisons use symmetric embeddings.
- Vectors are L2-normalised, so the inner product equals cosine similarity.

**Vector database.**
- FAISS `IndexFlatIP` holds the question vectors, with a JSONL metadata store: question, subject, year, exam period, topic, subtopic, marks, paper, source file.
- Filters on subject, year or year range, session, topic and subtopic are applied **inside** the FAISS search with an ID selector.
- New questions are appended; edited or removed questions trigger a rebuild.

**RAG.**
- **Query understanding** reads explicit subject and year filters and the intent (lookup, trend, forecast, repeat), and softly detects topics.
- **Retrieval** runs a filtered FAISS search.
- **Context** is built from:
  - retrieved questions `[E#]`;
  - *computed* topic statistics `[S#]`, which are exact counts from the full dataset;
  - for forecast questions, model forecasts `[F#]`, labelled as estimates.
- **Generation** uses a grounded prompt, and the answer is split into *Historical evidence / Observed trend / Forecast*.
- **Safeguards:**
  - Retrieval below a similarity threshold → "insufficient evidence" without calling the LLM.
  - Citations are validated, and invented tags are removed.
  - Years or exam periods that aren't in the context are flagged.
  - If the LLM concludes the evidence is insufficient, that is recorded.
  - With no LLM available, an extractive answer is built from the same evidence.

**Trend analysis.**
- **Time unit:** the exam session (one paper per subject per session; 10 periods). Calendar-year views are included too, but 2021 and 2026 are half-years.
- **Tables:** per-period topic and subtopic frequency, with zeros filled in; share; marks; the per-exam dominant topic.
- **Labels** follow explicit, configurable rules (`config.yaml → trends`). A single appearance is never called a trend.
  - *sporadic*: fewer than 3 appearances.
  - *emerging*: at most 1 question before the last 3 periods, then present in at least 2 of them.
  - *dormant*: absent in the last 3 periods.
  - *increasing* / *declining*: OLS slope ≥ 0.10 questions per period *and* Kendall τ ≥ 0.30 (or the negatives).
  - *consistently recurring*: present in ≥ 80% of periods.
  - *stable*: otherwise.

**Repeated questions.** Every pair of questions from different papers is compared on cosine similarity and lexical similarity, then typed:
- **exact**
- **near-identical:** only spelling or filler differences
- **rephrased:** same ask, different wording
- **semantically similar:** related concept

A conflicting-terms guard (CNN/RNN, supervised/unsupervised, advantages/disadvantages) stops look-alike pairs being called repeats. Strong relations are grouped into *question families* (`reports/question_families.csv`).

**Forecasting.**
- **Target:** the number of questions on each subject topic in the next exam (W2026).
- **Method selection:** each method is backtested with rolling origins (fit on exams before the origin, predict the origin: 6 origins × 34 series), and the lowest-MAE method is selected.
- **Outputs:**
  - `forecast_score` = 1 − e^(−predicted count), the chance of at least one question if counts were Poisson.
  - `forecast_confidence` comes from how regularly the topic appears and its backtest error relative to its volume.
  - An evidence string for each topic.
- Every output is phrased as a model estimate from historical patterns, never a guarantee.

## Evaluation results

All numbers come from `reports/evaluation_report.json` and were produced by running the pipeline on this dataset. **Human-verified** and **automatic (silver)** results are reported separately.

**Topic classification, human-verified:** *not available yet*. It needs a person to fill the `verified` column of `data/review/validation_sample.csv` (100 stratified random questions). The pipeline then computes accuracy, macro and weighted precision/recall/F1, per-subject confusion matrices, subtopic accuracy, multi-label micro-F1, Hamming loss and Jaccard.

**Topic classification, automatic:** agreement with a blind Claude reference labeller on the same 100 questions. This is silver, **not ground truth**.

| Confidence band | n | Topic agreement | Subtopic agreement |
|---|---|---|---|
| high | 86 | 1.000 | 0.977 |
| medium | 11 | 0.818 | 0.818 |
| low | 3 | 0.667 | 0.333 |
| **all** | 100 | **0.970** | **0.940** |

**Retrieval.** The test sets are semi-automatic; there are no manual relevance judgements.

| Test set | System | MRR | P@1 | Recall@5 | Hit@5 | Hit@10 |
|---|---|---|---|---|---|---|
| Repeat set: 175 questions re-asked in other papers (lexical ground truth) | **Semantic (FAISS)** | **0.984** | 0.971 | 0.991 | 1.000 | 1.000 |
| | TF-IDF baseline | 0.978 | 0.960 | 0.992 | 1.000 | 1.000 |
| Paraphrase set: 59 LLM-reworded student queries (silver) | **Semantic (FAISS)** | **0.743** | 0.661 | 0.864 | 0.864 | 0.966 |
| | TF-IDF baseline | 0.380 | 0.271 | 0.517 | 0.525 | 0.644 |

Contextual embeddings roughly double MRR over lexical search when the wording differs.

**RAG:** 13 queries, including 2 out-of-scope controls (DBMS and networking); generator `claude-opus-5`.
- **Deterministic checks:**
  - Refusal accuracy: 1.00. Both out-of-scope queries were refused, and all in-scope queries were answered.
  - Grounded rate: 1.00. No invented citations, years or periods.
  - Context keyword precision: 0.82.
  - Citation coverage: about 0.73 of answer sentences.
  - Cited evidence ratio: 0.96.
- **Automatic LLM judge** (silver, same model family as the generator): faithfulness 5.0 / 5, answer relevance 5.0, context relevance 4.9, and 0 answers with unsupported claims.

**Forecasting:** rolling-origin backtest, 204 forecasts per method, origins S2024 to S2026.

| Method | MAE | RMSE | Appearance F1 | Mean rank correlation |
|---|---|---|---|---|
| **historical mean (selected)** | **0.963** | 1.307 | 0.980 | 0.603 |
| random forest | 1.010 | 1.318 | 0.980 | 0.574 |
| moving average (3) | 1.036 | 1.424 | 0.974 | 0.568 |
| exponential smoothing | 1.065 | 1.439 | 0.987 | 0.527 |
| seasonal mean | 1.087 | 1.479 | 0.980 | 0.530 |
| linear trend | 1.280 | 1.683 | 0.982 | 0.440 |

This is an honest finding: with 10 exams per subject, no method beats the long-run average. Topic weights in these papers are largely set by the syllabus, and the recent swings are mostly noise. Appearance F1 is high because most syllabus units appear in almost every paper.

## Installation

Requires Python 3.10+ (developed on 3.14, Windows 11, NVIDIA RTX GPU; CPU works too).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt      # CPU-only: change the torch line to torch==2.11.0
cp .env.example .env                 # then add ANTHROPIC_API_KEY (optional)
```

## Configuration

- **`.env`** holds secrets only.
  - `ANTHROPIC_API_KEY` for Claude.
  - `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` for any OpenAI-compatible server, including local Ollama or LM Studio.
  - No key at all: the system still runs. Uncertain labels go to human review, and RAG answers are extractive.
- **`config.yaml`** holds every model, path and threshold: the embedding model, confidence thresholds, retrieval top-k and the insufficient-evidence threshold, the LLM provider, the similarity thresholds for repeats, the trend rules and the forecasting settings.
- **`config/taxonomy_source.yaml`** is the syllabus taxonomy. Edit it, bump its `version`, and re-run.

## Running

```bash
python pipeline.py                     # full pipeline (unchanged stages are skipped)
streamlit run app/app.py               # web app → http://localhost:8501
python -m pytest tests                 # 55 tests
```

Other commands:

```bash
python pipeline.py --force             # recompute everything
python pipeline.py --skip-eval         # no evaluation stage (no LLM calls for evaluation)
python pipeline.py --stage evaluation  # re-run evaluation, e.g. after verifying the validation sample
python -m src.ingestion.extract_questions --only-new   # new PDFs in data/raw → data/incoming/*.csv
```

**Adding new papers (incremental).**
1. Put the PDFs in `data/raw/<University>/<Subject>/`, named like the existing ones. Or put a CSV with the same columns in `data/incoming/`.
2. Run the extractor with `--only-new`, then `python pipeline.py`.

The pipeline then:
- detects the new question IDs;
- embeds only the new texts;
- appends them to the FAISS index;
- recomputes the topic statistics, trends and forecasts.

This was verified by holding out one paper and adding it back: `vector DB incremental: 25 added`.

**Outputs:**
- **Data:**
  - `data/final/`: `questions_cleaned.csv`, `questions_labeled.csv`, `questions.json`, `topic_taxonomy.json`
  - `data/review/`: `topic_review.csv`, `validation_sample.csv`
  - `vector_db/`
- **Reports** (`reports/`):
  - Data and labelling: `data_profile.json`/`.html`, `topic_labeling_report.json`, `cluster_discovery.json`
  - Trends: `trend_analysis.csv`, `topic_frequency_by_period.csv`, `subtopic_frequency_by_period.csv`, `topic_evolution.csv`, `trend_summary.json`
  - Repeats: `repeated_questions.csv`, `question_families.csv`
  - Forecasting: `forecast_results.csv`, `forecast_backtest.csv`, `forecast_summary.json`
  - Evaluation: `evaluation_report.json`/`.html`
  - Charts: `figures/*.html`
  - Log: `pipeline.log`

## Web application

The app has eight sections:
- **Dashboard:** KPIs, most frequent topics, trend labels, most-repeated questions, the topic sunburst.
- **Ask Historical Questions:** RAG answers with citation badges, a sources table with similarity bars, the statistics and forecasts used, and the full context.
- **Trend Analysis:** subject and period-range filters, trend lines, an evolution heatmap, trend classification with its reasons, recurring/emerging/declining lists, topic-mix shift, dominant topics per exam, subtopic drill-down.
- **Forecast:** forecast scores with confidence, evidence, history vs forecast, and the backtest reliability table.
- **Similar Questions:** free-text similarity search, plus known repeats of any dataset question.
- **Question Explorer:** filters for subject, period, topic, subtopic, marks, text and review status, with CSV download.
- **Review Queue:** approve or correct uncertain labels in the app.
- **Evaluation:** human-verified vs automatic metrics.

## Limitations

- **Human-verified classification accuracy is not available yet.** The 100-question validation sample must be verified by a person. The reported classification agreement is against an LLM reference labeller (silver), and the same model family also checked medium-confidence labels.
- **Small time series.** There are 10 exam sessions per subject, so trend labels rest on few points, and forecasts do not beat the historical mean. They should be read as "what the syllabus weighting and history suggest", not predictions.
- **The RAG judge is automatic.** The judge is the same model family as the generator, so faithfulness scores are indicative only; no human evaluation of answers has been done. Retrieval test sets are semi-automatic.
- **PDF extraction:**
  - About 9 matrix questions have jumbled numbers, and about 7 contain flattened tables. Both are flagged.
  - There is no page-number field.
  - One paper (AML S2026) has no semester in its header.
- **Scope.** The data covers four related ML subjects from one university (GTU). Taxonomies are subject-specific but share topic names where concepts are shared.
- **Confidence settings are provisional.** The classifier temperature and thresholds should be re-tuned once human-verified labels exist (use `calibration_by_band` in the evaluation report).

## Future work

- **Human evaluation:** complete the verification of the validation sample and a human RAG evaluation, then calibrate the confidence thresholds on them.
- **Classifier training:** use corrected labels to fine-tune a small classifier or the embedding model, instead of only adding prototypes.
- **Retrieval:** hybrid lexical + semantic retrieval with a cross-encoder re-ranker, and query expansion from the taxonomy.
- **Forecasting:** more universities and subjects, so that hierarchical or pooled forecasting has enough data; mark-weighted forecasts; forecast intervals.
- **Extraction:** OCR and table/formula extraction for scanned or mathematical papers; page-number capture.
- **Syllabus changes:** version the taxonomy over syllabus revisions, so trends stay comparable when a syllabus changes.
