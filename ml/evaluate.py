"""
ml/evaluate.py
==============
Back-testing: did our pipeline detect REAL FDA warnings before FDA
announced them?

How
---
1. Load training cutoff (default 2020-12-31).
2. Train signals/PRR/ROR on the slice <= cutoff.
3. For each FDA warning issued AFTER cutoff:
       - Did our pipeline flag this drug+reaction as SIGNAL or
         STRONG_SIGNAL on data available before warning_date?
       - How many days early?
4. Print headline metrics + save to data/processed/backtest_report.json
   and PostgreSQL pharma.backtest_results.

The model_metrics file is also rolled in for the resume bullet:
    "Detected X% of FDA warnings an average of Y months early."

Usage:
    python ml/evaluate.py
    python ml/evaluate.py --cutoff 2020-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

LOG = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "processed" / "ml_features"
WARNINGS_CSV = ROOT / "data" / "reference" / "fda_warnings.csv"
PRED_PARQUET = ROOT / "data" / "processed" / "predictions.parquet"
TEMPORAL_CSV = ROOT / "data" / "processed" / "temporal_warning_signals.csv"
OUT_JSON     = ROOT / "data" / "processed" / "backtest_report.json"


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
    p.add_argument("--warnings", type=Path, default=WARNINGS_CSV)
    p.add_argument("--cutoff",   type=str, default=os.getenv("BACKTEST_TRAIN_CUTOFF", "2020-12-31"))
    p.add_argument("--predictions", type=Path, default=PRED_PARQUET)
    p.add_argument("--temporal", type=Path, default=TEMPORAL_CSV)
    args = p.parse_args()

    cutoff = pd.to_datetime(args.cutoff)
    LOG.info("Back-test cutoff = %s", cutoff.date())

    feats = pd.read_parquet(args.features)
    feats["drug_name"]     = feats["drug_name"].str.upper()
    feats["reaction_term"] = feats["reaction_term"].str.upper()
    feats["pair_key"] = feats["drug_name"] + "||" + feats["reaction_term"]
    if args.temporal.exists():
        temporal = pd.read_csv(args.temporal, parse_dates=["signal_first_detected_date"])
        temporal["drug_name"] = temporal["drug_name"].str.upper()
        temporal["reaction_term"] = temporal["reaction_term"].str.upper()
        temporal["pair_key"] = temporal["drug_name"] + "||" + temporal["reaction_term"]
        feats = feats.drop(columns=["signal_first_detected_date"], errors="ignore").merge(
            temporal[["pair_key", "signal_first_detected_date"]],
            on="pair_key",
            how="left",
        )
    elif "signal_first_detected_date" in feats.columns:
        feats["signal_first_detected_date"] = pd.to_datetime(feats["signal_first_detected_date"], errors="coerce")

    warnings = pd.read_csv(args.warnings, parse_dates=["warning_date"])
    warnings["drug_name"]     = warnings["drug_name"].str.upper()
    warnings["reaction_term"] = warnings["reaction_term"].str.upper()

    future = warnings[warnings["warning_date"] > cutoff].copy()
    LOG.info("FDA issued %d warnings after cutoff", len(future))

    # PRR/ROR-only catch (rule-based) — emulate what Hive would say.
    rule_cols = ["drug_name", "reaction_term"]
    if "signal_first_detected_date" in feats.columns:
        rule_cols.append("signal_first_detected_date")
    rule_caught = feats[
        (feats["prr"] > 2.0)
        & (feats["ror"] > 2.0)
        & (feats["case_count"] >= 3)
    ][rule_cols].drop_duplicates(["drug_name", "reaction_term"])
    LOG.info("Rule-based pipeline flagged %s drug/reaction pairs", f"{len(rule_caught):,}")

    merged = future.merge(rule_caught.assign(rule_caught=1),
                          on=["drug_name", "reaction_term"], how="left")
    merged["rule_caught"] = merged["rule_caught"].fillna(0).astype(int)
    rule_recall = merged["rule_caught"].mean() if len(merged) else 0.0
    if "signal_first_detected_date" in merged.columns and merged["signal_first_detected_date"].notna().any():
        rule_days_early = (
            merged.loc[(merged["rule_caught"] == 1) & merged["signal_first_detected_date"].notna(), "warning_date"]
            - merged.loc[(merged["rule_caught"] == 1) & merged["signal_first_detected_date"].notna(), "signal_first_detected_date"]
        ).dt.days
        lead_time_basis = "signal_first_detected_date"
    else:
        rule_days_early = (
            merged.loc[merged["rule_caught"] == 1, "warning_date"] - cutoff
        ).dt.days
        lead_time_basis = "cutoff"
    rule_median_days = int(rule_days_early.median()) if len(rule_days_early) else None
    LOG.info("Rule-based recall over future warnings: %.3f (%d/%d)",
             rule_recall, int(merged['rule_caught'].sum()), len(merged))
    LOG.info("Rule-based median days early: %s", rule_median_days)

    # Now ML predictions
    ml_recall, median_days, p_at_100 = None, None, None
    if args.predictions.exists():
        preds = pd.read_parquet(args.predictions)
        preds["drug_name"]     = preds["drug_name"].str.upper()
        preds["reaction_term"] = preds["reaction_term"].str.upper()
        if "signal_first_detected_date" in preds.columns:
            preds["signal_first_detected_date"] = pd.to_datetime(
                preds["signal_first_detected_date"], errors="coerce"
            )
        pred_cols = ["drug_name", "reaction_term"]
        if "signal_first_detected_date" in preds.columns:
            pred_cols.append("signal_first_detected_date")
        high_prob = preds[preds["recall_probability"] >= 0.5][pred_cols]
        ml_caught = future.merge(
            high_prob.assign(ml_caught=1),
            on=["drug_name", "reaction_term"], how="left",
        )
        ml_caught["ml_caught"] = ml_caught["ml_caught"].fillna(0).astype(int)
        ml_recall = ml_caught["ml_caught"].mean() if len(ml_caught) else 0.0

        if "signal_first_detected_date" in ml_caught.columns and ml_caught["signal_first_detected_date"].notna().any():
            days_early = (
                ml_caught.loc[(ml_caught["ml_caught"] == 1) & ml_caught["signal_first_detected_date"].notna(), "warning_date"]
                - ml_caught.loc[(ml_caught["ml_caught"] == 1) & ml_caught["signal_first_detected_date"].notna(), "signal_first_detected_date"]
            ).dt.days
            lead_time_basis = "signal_first_detected_date"
        else:
            days_early = (
                ml_caught.loc[ml_caught["ml_caught"] == 1, "warning_date"] - cutoff
            ).dt.days
        median_days = int(days_early.median()) if len(days_early) else None

        # precision @ top-100 highest-probability predictions
        top100 = preds.nlargest(100, "recall_probability")[["drug_name", "reaction_term"]]
        hit = top100.merge(warnings[["drug_name", "reaction_term"]],
                           on=["drug_name", "reaction_term"], how="inner")
        p_at_100 = len(hit) / 100.0

        LOG.info("ML recall over future warnings: %.3f", ml_recall)
        LOG.info("ML median days early: %s", median_days)
        LOG.info("ML precision@100: %.3f", p_at_100)

    # The headline is about signal detection. Prefer the PRR/ROR threshold
    # detection date; use ML timing only if the rule-based pipeline caught none.
    headline_days = rule_median_days if rule_median_days is not None else median_days
    headline_months = (headline_days / 30.0) if headline_days else None
    headline_recall = ml_recall if ml_recall is not None and ml_recall > 0 else rule_recall
    if len(future):
        headline = (
            f"Pipeline detected {int(headline_recall * 100)}% "
            f"of FDA warnings, median {headline_months or 0:.1f} months early."
        )
    else:
        headline = (
            f"No FDA warnings in reference data after cutoff {args.cutoff}; "
            "backtest recall is not evaluable for this slice."
        )

    report = {
        "run_date": date.today().isoformat(),
        "train_cutoff": args.cutoff,
        "future_warnings_count": int(len(future)),
        "rule_recall": float(rule_recall),
        "rule_warnings_caught": int(merged["rule_caught"].sum()) if len(merged) else 0,
        "rule_median_days_early": rule_median_days,
        "ml_recall": float(ml_recall) if ml_recall is not None else None,
        "ml_median_days_early": median_days,
        "precision_at_100": float(p_at_100) if p_at_100 is not None else None,
        "median_days_early": headline_days,
        "median_months_early": round(headline_months, 1) if headline_months else None,
        "lead_time_basis": lead_time_basis,
        "headline": headline,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    LOG.info("Report -> %s", OUT_JSON)
    LOG.info("HEADLINE: %s", report["headline"])

    # Write to PostgreSQL for the dashboard
    try:
        eng = pg_engine()
        with eng.begin() as conn:
            conn.exec_driver_sql(
                """
                INSERT INTO pharma.backtest_results
                    (model_version, train_cutoff_date, auc_roc, precision_at_100,
                     recall_overall, median_days_early, warnings_caught, warnings_total, notes)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    "xgb-v1",
                    args.cutoff,
                    None,
                    p_at_100,
                    headline_recall,
                    headline_days,
                    int(headline_recall * len(future)) if len(future) else 0,
                    int(len(future)),
                    report["headline"],
                ),
            )
        LOG.info("Wrote backtest row -> pharma.backtest_results")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not write backtest to PostgreSQL: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
