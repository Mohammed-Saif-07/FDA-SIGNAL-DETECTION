"""
Load dashboard-facing PostgreSQL tables from local Parquet pipeline outputs.

This keeps the heavy FAERS processing in PySpark/Parquet, then publishes a
compact result set for Streamlit to query quickly.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import psycopg2
from psycopg2.extras import execute_values


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "ml_features"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "predictions.parquet"


def pg_params() -> dict[str, str]:
    return {
        "user": os.getenv("POSTGRES_USER", "fda"),
        "password": os.getenv("POSTGRES_PASSWORD", "fda"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "fda_signals"),
    }


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_rows(df: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [tuple(clean_value(v) for v in row) for row in df.itertuples(index=False, name=None)]


def load_signals(path: Path, top_n: int) -> pd.DataFrame:
    cols = [
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
    optional_cols = ["signal_first_detected_date"]
    available = pq.ParquetDataset(path).schema.names
    cols.extend([col for col in optional_cols if col in available])
    df = pd.read_parquet(path, columns=cols)

    signals = df[
        (df["case_count"] >= 3)
        & (df["prr"] > 2.0)
        & (df["ror"] > 2.0)
    ].copy()
    signals = signals.sort_values(
        ["prr_chi_square", "case_count", "prr"],
        ascending=[False, False, False],
    ).head(top_n)

    for col in ["prr", "ror", "prr_chi_square"]:
        signals[col] = signals[col].clip(lower=-999_999.9999, upper=999_999.9999)

    grand_total = int(df["case_count"].sum())
    signals["drug_total"] = signals["case_count"].astype(int)
    signals["reaction_total"] = signals["case_count"].astype(int)
    signals["grand_total"] = grand_total
    signals["signal_status"] = "STRONG_SIGNAL"
    signals["confidence"] = signals.apply(
        lambda r: "HIGH"
        if r["case_count"] >= 10 and r["prr"] >= 4 and r["ror"] >= 4
        else ("MEDIUM" if r["case_count"] >= 5 else "LOW"),
        axis=1,
    )
    if "signal_first_detected_date" in signals.columns:
        signals["first_detected_date"] = pd.to_datetime(
            signals["signal_first_detected_date"], errors="coerce"
        ).dt.date
        signals["first_detected_date"] = signals["first_detected_date"].fillna(date.today())
    else:
        signals["first_detected_date"] = date.today()
    signals["last_updated"] = pd.Timestamp.utcnow().to_pydatetime()

    out_cols = [
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
    ]
    return signals[out_cols]


def load_predictions(path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("recall_probability", ascending=False)
    df = df.drop_duplicates(["drug_name", "reaction_term", "model_version"]).head(top_n)

    out_cols = [
        "drug_name",
        "reaction_term",
        "recall_probability",
        "predicted_class",
        "predicted_date",
        "actual_fda_warning_date",
        "days_predicted_early",
        "model_version",
    ]
    return df[out_cols]


def insert_dataframe(cur, table: str, columns: list[str], df: pd.DataFrame) -> None:
    if df.empty:
        return
    col_list = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_list}) VALUES %s"
    execute_values(cur, sql, clean_rows(df), page_size=1000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--signals-top-n", type=int, default=10_000)
    parser.add_argument("--predictions-top-n", type=int, default=10_000)
    args = parser.parse_args()

    signals = load_signals(args.features, args.signals_top_n)
    predictions = load_predictions(args.predictions, args.predictions_top_n)

    with psycopg2.connect(**pg_params()) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE pharma.signal_predictions, pharma.drug_signals RESTART IDENTITY")
            insert_dataframe(cur, "pharma.drug_signals", list(signals.columns), signals)
            insert_dataframe(cur, "pharma.signal_predictions", list(predictions.columns), predictions)
        conn.commit()

    print(
        f"Loaded {len(signals):,} signals and {len(predictions):,} predictions "
        "into PostgreSQL."
    )


if __name__ == "__main__":
    main()
