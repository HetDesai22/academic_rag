"""Loads pipeline artifacts and query engines for the web app, tests and notebooks."""
import json

import pandas as pd

from src.utils.common import get_logger, load_config, read_json


def load_artifacts(cfg=None):
    cfg = cfg or load_config()
    P = cfg["paths"]
    missing = [p for p in (P["output_csv"], P["trend_csv"], P["forecast_csv"], P["taxonomy_path"]) if not p.exists()]
    if missing:
        raise FileNotFoundError("Pipeline outputs missing: " + ", ".join(str(m) for m in missing)
                                + ". Run `python pipeline.py` first.")
    df = pd.read_csv(P["output_csv"], encoding="utf-8-sig")
    df["secondary_topics"] = df["secondary_topics"].map(lambda s: json.loads(s) if isinstance(s, str) and s else [])
    art = {
        "questions": df,
        "taxonomy": read_json(P["taxonomy_path"]),
        "trends": pd.read_csv(P["trend_csv"], encoding="utf-8-sig"),
        "topic_period": pd.read_csv(P["topic_period_csv"], encoding="utf-8-sig"),
        "subtopic_period": _csv(P["reports_dir"] / "subtopic_frequency_by_period.csv"),
        "evolution": _csv(P["reports_dir"] / "topic_evolution.csv"),
        "forecast": pd.read_csv(P["forecast_csv"], encoding="utf-8-sig"),
        "forecast_summary": read_json(P["reports_dir"] / "forecast_summary.json", {}),
        "backtest": _csv(P["forecast_backtest_csv"]),
        "pairs": _csv(P["repeated_csv"]),
        "families": _csv(P["reports_dir"] / "question_families.csv"),
        "review": _csv(P["review_csv"]),
        "profile": read_json(P["profile_report"], {}),
        "labeling_report": read_json(P["labeling_report"], {}),
        "evaluation": read_json(P["evaluation_report"], {}),
        "trend_summary": read_json(P["trend_summary_json"], {}),
    }
    return art


def _csv(path):
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def build_engines(cfg=None, art=None, log=None):
    from src.embeddings.embedder import Embedder
    from src.rag.rag_pipeline import RAGPipeline
    from src.retrieval.retriever import Retriever
    cfg = cfg or load_config()
    log = log or get_logger(cfg)
    art = art or load_artifacts(cfg)
    embedder = Embedder(cfg, log)
    retriever = Retriever(cfg, embedder)
    rag = RAGPipeline(cfg, retriever, art["taxonomy"], art["trends"], art["forecast"], log)
    return {"embedder": embedder, "retriever": retriever, "rag": rag}
