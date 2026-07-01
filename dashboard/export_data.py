"""Export compact CSV snapshots for Streamlit Cloud.

Primary path: read PostgreSQL tables populated by ``ml/load_dashboard_data.py``.
Fallback path: read local Parquet/JSON artifacts directly. The fallback is
important because Streamlit Cloud and quick local checks should not require
Docker/PostgreSQL to be running.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from signal_quality import add_signal_quality  # noqa: E402


OUT = Path(__file__).parent / "data"
FEATURES = ROOT / "data" / "processed" / "ml_features"
PREDICTIONS = ROOT / "data" / "processed" / "predictions.parquet"
BACKTEST_REPORT = ROOT / "data" / "processed" / "backtest_report.json"

OUT.mkdir(exist_ok=True)


def postgres_engine():
    return create_engine(
        "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
            u=os.getenv("POSTGRES_USER", "fda"),
            p=os.getenv("POSTGRES_PASSWORD", "fda"),
            h=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            db=os.getenv("POSTGRES_DB", "fda_signals"),
        )
    )


def export_from_postgres() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = postgres_engine()
    conn = engine.raw_connection()
    try:
        signals = pd.read_sql(
            """
            SELECT *
            FROM pharma.drug_signals
            ORDER BY passes_robust_filter DESC NULLS LAST,
                     robust_signal_score DESC NULLS LAST,
                     prr_chi_square DESC NULLS LAST,
                     case_count DESC
            LIMIT 2000
            """,
            conn,
        )
        preds = pd.read_sql(
            "SELECT * FROM pharma.signal_predictions ORDER BY recall_probability DESC LIMIT 2000",
            conn,
        )
        bt = pd.read_sql("SELECT * FROM pharma.backtest_results ORDER BY run_date DESC", conn)
    finally:
        conn.close()
    return signals, preds, bt


def export_from_local_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "drug_name",
        "reaction_term",
        "case_count",
        "serious_ratio",
        "death_ratio",
        "countries_count",
        "prr",
        "ror",
        "prr_chi_square",
    ]
    feats = pd.read_parquet(FEATURES, columns=feature_cols)
    feats = add_signal_quality(feats)
    signals = feats[feats["passes_robust_filter"]].copy()
    signals = signals.sort_values(
        ["robust_signal_score", "prr_chi_square", "case_count"],
        ascending=[False, False, False],
    ).head(2000)
    grand_total = int(feats["case_count"].sum())
    signals.insert(0, "id", range(1, len(signals) + 1))
    signals["drug_total"] = signals["case_count"].astype(int)
    signals["reaction_total"] = signals["case_count"].astype(int)
    signals["grand_total"] = grand_total
    signals["signal_status"] = "STRONG_SIGNAL"
    signals["confidence"] = signals.apply(
        lambda row: "HIGH"
        if row["case_count"] >= 10 and row["prr"] >= 4 and row["ror"] >= 4
        else ("MEDIUM" if row["case_count"] >= 5 else "LOW"),
        axis=1,
    )
    signals["first_detected_date"] = pd.Timestamp.today().date().isoformat()
    signals["last_updated"] = pd.Timestamp.utcnow().isoformat()
    signal_cols = [
        "id",
        "drug_name",
        "reaction_term",
        "case_count",
        "drug_total",
        "reaction_total",
        "grand_total",
        "prr",
        "ror",
        "prr_chi_square",
        "signal_status",
        "confidence",
        "first_detected_date",
        "last_updated",
        "serious_ratio",
        "death_ratio",
        "countries_count",
        "source_proxy_count",
        "passes_robust_filter",
        "passes_structural_filter",
        "artifact_reason",
        "robust_signal_score",
    ]
    signals = signals[signal_cols]

    preds = pd.read_parquet(PREDICTIONS)
    preds = preds.sort_values("recall_probability", ascending=False).head(2000)
    preds.insert(0, "id", range(1, len(preds) + 1))

    if BACKTEST_REPORT.exists():
        report = json.loads(BACKTEST_REPORT.read_text())
        bt = pd.DataFrame(
            [
                {
                    "id": 1,
                    "run_date": report.get("run_date"),
                    "model_version": "xgb-v1",
                    "train_cutoff_date": report.get("train_cutoff"),
                    "auc_roc": None,
                    "precision_at_100": report.get("precision_at_100"),
                    "recall_overall": report.get("rule_recall"),
                    "median_days_early": report.get("median_days_early"),
                    "warnings_caught": report.get("rule_warnings_caught"),
                    "warnings_total": report.get("future_warnings_count"),
                    "notes": report.get("headline"),
                }
            ]
        )
    else:
        bt = pd.DataFrame()
    return signals, preds, bt


try:
    signals_df, preds_df, bt_df = export_from_postgres()
    source = "PostgreSQL"
except (SQLAlchemyError, OSError, AttributeError) as exc:
    print(f"PostgreSQL unavailable; using local artifact fallback: {exc}")
    signals_df, preds_df, bt_df = export_from_local_artifacts()
    source = "local artifacts"

signals_df.to_csv(OUT / "signals.csv", index=False)
preds_df.to_csv(OUT / "predictions.csv", index=False)
bt_df.to_csv(OUT / "backtests.csv", index=False)

print(f"source          -> {source}")
print(f"signals.csv     -> {len(signals_df):,} rows")
print(f"predictions.csv -> {len(preds_df):,} rows")
print(f"backtests.csv   -> {len(bt_df):,} rows")
print(f"\nAll CSVs in {OUT}")
