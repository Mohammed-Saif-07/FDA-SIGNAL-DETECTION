"""
ml/predictor.py
===============
Load the trained XGBoost model and score the latest feature matrix.

* Reads:   ml/models/xgb_signal_recall.json
* Reads:   data/processed/ml_features/
* Writes:  PostgreSQL  pharma.signal_predictions
* Writes:  data/processed/predictions.parquet  (audit copy)

Usage:
    python ml/predictor.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine, text

LOG = logging.getLogger("predictor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ml" / "models"
FEATURES_DIR = ROOT / "data" / "processed" / "ml_features"
OUT_PARQUET = ROOT / "data" / "processed" / "predictions.parquet"


def pg_engine() -> "create_engine":
    return create_engine(
        "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
            u=os.getenv("POSTGRES_USER", "fda"),
            p=os.getenv("POSTGRES_PASSWORD", "fda"),
            h=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            db=os.getenv("POSTGRES_DB", "fda_signals"),
        )
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=FEATURES_DIR)
    p.add_argument("--model",    type=Path, default=MODEL_DIR / "xgb_signal_recall.json")
    p.add_argument("--cols",     type=Path, default=MODEL_DIR / "feature_columns.json")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-write-pg", dest="write_pg", action="store_false",
                   help="Skip writing to PostgreSQL (parquet copy only)")
    p.set_defaults(write_pg=True)
    args = p.parse_args()

    feature_cols = json.loads(args.cols.read_text())
    LOG.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    df = pd.read_parquet(args.features)
    LOG.info("Loaded %s feature rows", f"{len(df):,}")

    X = df[feature_cols].astype("float32").fillna(-1)

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    LOG.info("Loaded model %s", args.model)

    probs = model.predict_proba(X)[:, 1]
    df["recall_probability"] = probs.round(4)
    df["predicted_class"]    = (probs >= args.threshold).astype(int)
    df["predicted_date"]     = date.today()
    df["model_version"]      = "xgb-v1"

    out = df[[
        "drug_name", "reaction_term",
        "recall_probability", "predicted_class",
        "predicted_date",
        "warning_date", "model_version",
    ]].copy()
    out.rename(columns={"warning_date": "actual_fda_warning_date"}, inplace=True)

    # how many days early (negative = late)
    out["days_predicted_early"] = (
        pd.to_datetime(out["actual_fda_warning_date"]) - pd.to_datetime(out["predicted_date"])
    ).dt.days

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    LOG.info("Wrote %s predictions -> %s", f"{len(out):,}", OUT_PARQUET)

    if args.write_pg:
        try:
            eng = pg_engine()
            with eng.begin() as conn:
                conn.execute(text("TRUNCATE pharma.signal_predictions RESTART IDENTITY"))
                out.to_sql("signal_predictions", conn, schema="pharma",
                           if_exists="append", index=False, method="multi", chunksize=5000)
            LOG.info("Wrote %s rows -> pharma.signal_predictions", f"{len(out):,}")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not write to PostgreSQL (%s). "
                        "Probably running outside docker-compose; parquet copy is fine.", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
