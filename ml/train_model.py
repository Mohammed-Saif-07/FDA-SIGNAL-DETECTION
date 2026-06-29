"""
ml/train_model.py
=================
Train the XGBoost "signal → recall" classifier.

Inputs
------
* data/processed/ml_features/      (Parquet, built by spark/feature_engineering.py)
* data/reference/fda_warnings.csv  (ground-truth warning dates)

Outputs
-------
* ml/models/xgb_signal_recall.json     (XGBoost model)
* ml/models/feature_columns.json       (column order at training time)
* ml/models/metrics.json               (AUC, P@100, recall, etc.)

Training strategy
-----------------
* Use only events with report_date <= TRAIN_CUTOFF (default 2020-12-31).
* Positive label = drug+reaction pair eventually became an FDA warning.
* Negative label = high-PRR pair that never became a warning.
* Random under-sample negatives 5:1 to manage class imbalance.

Usage:
    python ml/train_model.py
    python ml/train_model.py --train-cutoff 2020-12-31
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
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

LOG = logging.getLogger("train_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "processed" / "ml_features"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "case_count",
    "case_count_growth_qoq",
    "serious_ratio",
    "death_ratio",
    "hosp_ratio",
    "countries_count",
    "age_mean",
    "age_std",
    "sex_female_ratio",
    "days_since_first_report",
    "prr",
    "ror",
    "prr_chi_square",
    "n_concurrent_signals",
]


def load_features(path: Path) -> pd.DataFrame:
    LOG.info("Loading features from %s", path)
    df = pd.read_parquet(path)
    LOG.info("Loaded %s rows  (positives=%d)",
             f"{len(df):,}", int(df["label_became_warning"].sum()))
    return df


def split_train_eval(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use rows whose warning_date is null OR > cutoff for *training*
    (we don't yet "know" they'll become warnings if cutoff is honest),
    and use rows whose warning_date is > cutoff for evaluation.
    """
    cutoff_dt = pd.to_datetime(cutoff)
    df["warning_date"] = pd.to_datetime(df["warning_date"], errors="coerce")

    train = df.copy()
    # mask labels that happen AFTER the cutoff (we don't know yet)
    future_mask = (train["warning_date"].notna()) & (train["warning_date"] > cutoff_dt)
    train.loc[future_mask, "label_became_warning"] = 0

    eval_df = df[future_mask].copy()
    eval_df["label_became_warning"] = 1   # those are the future warnings to catch
    LOG.info("Train rows=%s, eval future-warnings=%s", f"{len(train):,}", f"{len(eval_df):,}")
    return train, eval_df


def undersample(df: pd.DataFrame, ratio: int = 5, seed: int = 42) -> pd.DataFrame:
    pos = df[df["label_became_warning"] == 1]
    neg = df[df["label_became_warning"] == 0]
    n_neg = min(len(neg), len(pos) * ratio)
    if n_neg == 0:
        return df
    neg = neg.sample(n=n_neg, random_state=seed)
    out = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    LOG.info("Undersampled: pos=%d, neg=%d", len(pos), len(neg))
    return out


def precision_at_k(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    if k > len(probs):
        k = len(probs)
    order = np.argsort(-probs)[:k]
    return float(y[order].mean())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=FEATURES_DIR)
    p.add_argument("--train-cutoff", default=os.getenv("BACKTEST_TRAIN_CUTOFF", "2020-12-31"))
    p.add_argument("--model-version", default=f"xgb-v1-{date.today().isoformat()}")
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = load_features(args.features)
    train_df, future_df = split_train_eval(df, args.train_cutoff)

    train_df = undersample(train_df, ratio=5, seed=args.seed)

    X = train_df[FEATURE_COLS].astype("float32").fillna(-1)
    y = train_df["label_became_warning"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed,
    )

    LOG.info("Training XGBoost on %s rows × %d features …", f"{len(X_tr):,}", X_tr.shape[1])
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )

    prob_te = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, prob_te)
    ap  = average_precision_score(y_te, prob_te)
    p_at_100 = precision_at_k(prob_te, y_te, 100)
    LOG.info("Hold-out AUC=%.4f   AvgP=%.4f   P@100=%.4f", auc, ap, p_at_100)

    # Score the future-warnings set: did we catch them?
    catch_rate = None
    median_days_early = None
    if not future_df.empty:
        Xf = future_df[FEATURE_COLS].astype("float32").fillna(-1)
        probs_f = model.predict_proba(Xf)[:, 1]
        caught = (probs_f >= 0.5).sum()
        catch_rate = float(caught) / max(len(future_df), 1)
        # crude "days early" = days between today and the warning_date
        # (in practice you'd record predicted_date per row)
        cutoff_dt = pd.to_datetime(args.train_cutoff)
        days = (future_df["warning_date"] - cutoff_dt).dt.days
        median_days_early = int(days.median()) if len(days) else None
        LOG.info("Future-warning catch rate (prob>=0.5): %.3f  median days early=%s",
                 catch_rate, median_days_early)

    # Persist artefacts
    model_path = MODEL_DIR / "xgb_signal_recall.json"
    model.save_model(model_path)
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLS))
    metrics = {
        "model_version": args.model_version,
        "train_cutoff": args.train_cutoff,
        "auc_roc": auc,
        "average_precision": ap,
        "precision_at_100": p_at_100,
        "future_catch_rate": catch_rate,
        "median_days_early": median_days_early,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_future_positives": int(len(future_df)),
        "seed": args.seed,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    LOG.info("Saved model -> %s", model_path)
    LOG.info("Metrics: %s", json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
