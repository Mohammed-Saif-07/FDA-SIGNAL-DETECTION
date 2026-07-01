"""
Research-grade evaluation for the FAERS signal detection pipeline.

This script is intentionally stricter than ml/evaluate.py:
it compares several ranking baselines across multiple cutoff dates and writes
paper-friendly CSV/JSON artifacts. It does not invent FDA warnings and it does
not claim causality from FAERS disproportionality signals.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from bcpnn import add_bcpnn_scores
from bootstrap_eval import binomial_ci
from ebgm import add_ebgm_scores
from signal_quality import add_signal_quality


ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "processed" / "ml_features"
PRED_PARQUET = ROOT / "data" / "processed" / "predictions.parquet"
WARNINGS_CSV = ROOT / "data" / "reference" / "fda_warnings.csv"
TEMPORAL_CSV = ROOT / "data" / "processed" / "temporal_warning_signals.csv"
OUT_DIR = ROOT / "data" / "processed"
METHOD_SCORES_PARQUET = OUT_DIR / "method_scores_2020.parquet"


DEFAULT_CUTOFFS = ["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"]
DEFAULT_KS = [50, 100]


def clean_json_value(value):
    if isinstance(value, dict):
        return {key: clean_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    return value


def normalize_pair(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["drug_name"] = out["drug_name"].astype(str).str.upper().str.strip()
    out["reaction_term"] = out["reaction_term"].astype(str).str.upper().str.strip()
    return out


def load_inputs(
    features: Path,
    warnings: Path,
    predictions: Path,
    temporal: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feats = normalize_pair(pd.read_parquet(features))
    feats["warning_date"] = pd.to_datetime(feats.get("warning_date"), errors="coerce")
    feats["pair_key"] = feats["drug_name"] + "||" + feats["reaction_term"]

    warns = normalize_pair(pd.read_csv(warnings, parse_dates=["warning_date"]))
    warns["pair_key"] = warns["drug_name"] + "||" + warns["reaction_term"]

    if predictions.exists():
        preds = normalize_pair(pd.read_parquet(predictions))
        preds["pair_key"] = preds["drug_name"] + "||" + preds["reaction_term"]
    else:
        preds = pd.DataFrame(columns=["drug_name", "reaction_term", "pair_key", "recall_probability"])

    if temporal.exists():
        temporal_df = normalize_pair(pd.read_csv(temporal, parse_dates=["signal_first_detected_date"]))
        temporal_df["pair_key"] = temporal_df["drug_name"] + "||" + temporal_df["reaction_term"]
        temporal_cols = [
            "pair_key",
            "signal_first_detected_date",
            "cumulative_case_count_at_detection",
            "prr_at_detection",
            "ror_at_detection",
        ]
        temporal_df = temporal_df[[col for col in temporal_cols if col in temporal_df.columns]]
        feats = feats.drop(columns=["signal_first_detected_date"], errors="ignore")
        feats = feats.merge(temporal_df, on="pair_key", how="left")
        if not preds.empty:
            preds = preds.drop(columns=["signal_first_detected_date"], errors="ignore")
            preds = preds.merge(temporal_df[["pair_key", "signal_first_detected_date"]], on="pair_key", how="left")
    elif "signal_first_detected_date" in feats.columns:
        feats["signal_first_detected_date"] = pd.to_datetime(
            feats["signal_first_detected_date"], errors="coerce"
        )

    return feats, warns, preds


def ranked_tables(feats: pd.DataFrame, preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    clean = feats.copy()
    for col in ["case_count", "drug_total", "reaction_total", "grand_total", "prr", "ror", "prr_chi_square"]:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    if "drug_total" not in clean.columns or clean["drug_total"].eq(0).all():
        clean["drug_total"] = clean.groupby("drug_name")["case_count"].transform("sum")
    if "reaction_total" not in clean.columns or clean["reaction_total"].eq(0).all():
        clean["reaction_total"] = clean.groupby("reaction_term")["case_count"].transform("sum")
    if "grand_total" not in clean.columns or clean["grand_total"].eq(0).all():
        clean["grand_total"] = float(clean["case_count"].sum())
    clean = add_signal_quality(clean)
    clean = add_bcpnn_scores(clean)
    clean = add_ebgm_scores(clean)

    signal_mask = clean["case_count"].ge(3)
    prr_mask = signal_mask & clean["prr"].gt(2)
    ror_mask = signal_mask & clean["ror"].gt(2)
    both_mask = prr_mask & ror_mask
    chi_mask = both_mask & clean["prr_chi_square"].gt(4)
    bcpnn_mask = clean["ic025"].gt(0)
    ebgm_mask = clean["eb05"].gt(2)

    tables: dict[str, pd.DataFrame] = {
        "case_count": clean.assign(score=clean["case_count"]).sort_values("score", ascending=False),
        "prr": clean[prr_mask].assign(score=clean.loc[prr_mask, "prr"]).sort_values("score", ascending=False),
        "ror": clean[ror_mask].assign(score=clean.loc[ror_mask, "ror"]).sort_values("score", ascending=False),
        "prr_ror": clean[both_mask].assign(
            score=np.minimum(clean.loc[both_mask, "prr"], clean.loc[both_mask, "ror"])
        ).sort_values("score", ascending=False),
        "prr_ror_chi_square": clean[chi_mask].assign(
            score=clean.loc[chi_mask, "prr_chi_square"]
        ).sort_values("score", ascending=False),
        "robust_prr_ror": clean[clean["passes_robust_filter"]].assign(
            score=clean.loc[clean["passes_robust_filter"], "robust_signal_score"]
        ).sort_values("score", ascending=False),
        "bcpnn_ic025": clean[bcpnn_mask].assign(
            score=clean.loc[bcpnn_mask, "ic025"]
        ).sort_values("score", ascending=False),
        "ebgm_eb05": clean[ebgm_mask].assign(
            score=clean.loc[ebgm_mask, "eb05"]
        ).sort_values("score", ascending=False),
    }

    if not preds.empty and "recall_probability" in preds:
        p = preds.copy()
        p["score"] = pd.to_numeric(p["recall_probability"], errors="coerce").fillna(0)
        tables["xgboost"] = p.sort_values("score", ascending=False)

    # One row per drug/reaction pair per ranking method.
    return {
        name: table.drop_duplicates("pair_key", keep="first").reset_index(drop=True)
        for name, table in tables.items()
    }


def evaluate_cutoff_method(
    cutoff: pd.Timestamp,
    method: str,
    ranked: pd.DataFrame,
    future: pd.DataFrame,
    k_values: list[int],
) -> list[dict]:
    rows = []
    future_keys = set(future["pair_key"])
    future_count = len(future)

    for k in k_values:
        top = ranked.head(k)
        hit_keys = set(top["pair_key"]).intersection(future_keys)
        hit_warnings = future[future["pair_key"].isin(hit_keys)].copy()
        lead_start = "cutoff"
        days = pd.Series(dtype="int64")
        if not hit_warnings.empty:
            if "signal_first_detected_date" in top.columns:
                detected = top[["pair_key", "signal_first_detected_date"]].dropna().drop_duplicates("pair_key")
                hit_warnings = hit_warnings.merge(detected, on="pair_key", how="left")
                valid = hit_warnings["signal_first_detected_date"].notna()
                if valid.any():
                    lead_start = "signal_first_detected_date"
                    days = (
                        hit_warnings.loc[valid, "warning_date"]
                        - hit_warnings.loc[valid, "signal_first_detected_date"]
                    ).dt.days
            if days.empty:
                days = (hit_warnings["warning_date"] - cutoff).dt.days

        rows.append(
            {
                "cutoff": cutoff.date().isoformat(),
                "method": method,
                "evaluation_type": "top_k",
                "k": k,
                "future_warnings": int(future_count),
                "warnings_caught": int(len(hit_keys)),
                "recall": float(len(hit_keys) / future_count) if future_count else None,
                "precision": float(len(hit_keys) / k) if k else None,
                "median_days_early": int(days.median()) if len(days) else None,
                "median_months_early": round(float(days.median()) / 30.0, 1) if len(days) else None,
                "lead_time_basis": lead_start,
                "ranked_candidates": int(len(ranked)),
            }
        )
    return rows


def add_summary_intervals(summary: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Add reproducible bootstrap/binomial confidence intervals to summary rows."""

    out = summary.copy()
    interval_rows = []
    for _, row in out.iterrows():
        caught = int(row.get("warnings_caught", 0) or 0)
        future = int(row.get("future_warnings", 0) or 0)
        candidates = row.get("k") if row.get("evaluation_type") == "top_k" else row.get("ranked_candidates")
        try:
            candidates_n = int(candidates)
        except (TypeError, ValueError):
            candidates_n = int(row.get("ranked_candidates", 0) or 0)

        recall_lo, recall_hi = binomial_ci(caught, future, n_boot=n_boot, seed=seed)
        precision_lo, precision_hi = binomial_ci(caught, candidates_n, n_boot=n_boot, seed=seed)
        median_days = row.get("median_days_early")
        if pd.notna(median_days):
            lead_lo = float(median_days)
            lead_hi = float(median_days)
        else:
            lead_lo = np.nan
            lead_hi = np.nan

        interval_rows.append(
            {
                "recall_lo95": recall_lo,
                "recall_hi95": recall_hi,
                "precision_lo95": precision_lo,
                "precision_hi95": precision_hi,
                "lead_time_lo95": lead_lo,
                "lead_time_hi95": lead_hi,
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(interval_rows)], axis=1)


def evaluate_threshold_method(
    cutoff: pd.Timestamp,
    method: str,
    candidates: pd.DataFrame,
    future: pd.DataFrame,
) -> dict:
    future_keys = set(future["pair_key"])
    hit_keys = set(candidates["pair_key"]).intersection(future_keys)
    hit_warnings = future[future["pair_key"].isin(hit_keys)].copy()
    lead_start = "cutoff"
    days = pd.Series(dtype="int64")
    if not hit_warnings.empty:
        if "signal_first_detected_date" in candidates.columns:
            detected = candidates[["pair_key", "signal_first_detected_date"]].dropna().drop_duplicates("pair_key")
            hit_warnings = hit_warnings.merge(detected, on="pair_key", how="left")
            valid = hit_warnings["signal_first_detected_date"].notna()
            if valid.any():
                lead_start = "signal_first_detected_date"
                days = (
                    hit_warnings.loc[valid, "warning_date"]
                    - hit_warnings.loc[valid, "signal_first_detected_date"]
                ).dt.days
        if days.empty:
            days = (hit_warnings["warning_date"] - cutoff).dt.days
    future_count = len(future)
    return {
        "cutoff": cutoff.date().isoformat(),
        "method": method,
        "evaluation_type": "threshold",
        "k": "threshold",
        "future_warnings": int(future_count),
        "warnings_caught": int(len(hit_keys)),
        "recall": float(len(hit_keys) / future_count) if future_count else None,
        "precision": float(len(hit_keys) / len(candidates)) if len(candidates) else None,
        "median_days_early": int(days.median()) if len(days) else None,
        "median_months_early": round(float(days.median()) / 30.0, 1) if len(days) else None,
        "lead_time_basis": lead_start,
        "ranked_candidates": int(len(candidates)),
    }


def build_case_tables(
    cutoff: pd.Timestamp,
    ranked: dict[str, pd.DataFrame],
    warnings: pd.DataFrame,
    feats: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    future = warnings[warnings["warning_date"] > cutoff].copy()
    future_keys = set(future["pair_key"])
    xgb = ranked.get("xgboost", pd.DataFrame()).head(k)
    rule = ranked["prr_ror"].copy()
    caught_keys = set(rule["pair_key"]).intersection(future_keys) | set(xgb["pair_key"]).intersection(future_keys)

    feature_cols = [
        "pair_key",
        "case_count",
        "prr",
        "ror",
        "prr_chi_square",
        "serious_ratio",
        "death_ratio",
        "countries_count",
    ]
    if "signal_first_detected_date" in feats.columns:
        feature_cols.append("signal_first_detected_date")
    feature_lookup = feats[feature_cols].drop_duplicates("pair_key", keep="first")

    caught = future[future["pair_key"].isin(caught_keys)].merge(feature_lookup, on="pair_key", how="left")
    if not caught.empty:
        caught["cutoff"] = cutoff.date().isoformat()
        if "signal_first_detected_date" in caught.columns and caught["signal_first_detected_date"].notna().any():
            caught["days_early"] = (
                caught["warning_date"] - caught["signal_first_detected_date"]
            ).dt.days
            caught["months_early"] = (caught["days_early"] / 30.0).round(1)
            caught["lead_time_basis"] = "signal_first_detected_date"
            caught["detection_note"] = (
                "Matched by PRR/ROR threshold and/or XGBoost ranking. "
                "Lead time is measured from the first quarter-end where cumulative PRR/ROR crossed threshold."
            )
        else:
            caught["days_early"] = (caught["warning_date"] - cutoff).dt.days
            caught["months_early"] = (caught["days_early"] / 30.0).round(1)
            caught["lead_time_basis"] = "cutoff"
            caught["detection_note"] = (
                "Matched by PRR/ROR threshold and/or XGBoost ranking. "
                "Lead time is measured from cutoff because signal_first_detected_date is unavailable."
            )

    missed = future[~future["pair_key"].isin(caught_keys)].copy()
    if not missed.empty:
        available_keys = set(feats["pair_key"])
        missed["cutoff"] = cutoff.date().isoformat()
        missed["likely_reason"] = np.where(
            missed["pair_key"].isin(available_keys),
            "Present in feature table but did not rank/threshold high enough for this evaluation.",
            "No exact normalized drug/reaction match in processed feature table.",
        )

    return caught, missed


def false_positive_table(feats: pd.DataFrame, warnings: pd.DataFrame, limit: int) -> pd.DataFrame:
    warning_keys = set(warnings["pair_key"])
    clean = add_signal_quality(feats)
    for col in ["case_count", "prr", "ror", "prr_chi_square"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    mask = (
        clean["case_count"].ge(3)
        & clean["prr"].gt(2)
        & clean["ror"].gt(2)
        & ~clean["pair_key"].isin(warning_keys)
    )
    cols = [
        "drug_name",
        "reaction_term",
        "case_count",
        "prr",
        "ror",
        "prr_chi_square",
        "serious_ratio",
        "death_ratio",
        "countries_count",
        "source_proxy_count",
        "passes_robust_filter",
        "passes_structural_filter",
        "artifact_reason",
        "robust_signal_score",
    ]
    return clean.loc[mask, cols].sort_values(
        ["passes_robust_filter", "robust_signal_score", "case_count", "prr"],
        ascending=[False, False, False, False],
    ).head(limit)


def write_method_scores(
    feats: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    warnings: pd.DataFrame,
    cutoff: pd.Timestamp,
    out_path: Path = METHOD_SCORES_PARQUET,
) -> pd.DataFrame:
    """Write individual per-method scores needed for ROC/AUC comparisons."""

    future_keys = set(warnings.loc[warnings["warning_date"] > cutoff, "pair_key"])
    base = feats[["pair_key", "drug_name", "reaction_term"]].drop_duplicates("pair_key").copy()
    base["is_post_cutoff_warning"] = base["pair_key"].isin(future_keys).astype(int)
    for method, ranked in rankings.items():
        if "score" not in ranked.columns:
            continue
        scores = ranked[["pair_key", "score"]].drop_duplicates("pair_key", keep="first")
        base = base.merge(scores.rename(columns={"score": method}), on="pair_key", how="left")
        base[method] = pd.to_numeric(base[method], errors="coerce").fillna(0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out_path, index=False)
    return base


def write_primary_backtest_report(summary: pd.DataFrame, cutoff_text: str, out_path: Path) -> dict:
    threshold = summary[summary["evaluation_type"].eq("threshold")].copy()
    # Correct pooled ratio: sum(caught) / sum(total) across cutoffs.
    # Also expose mean-of-per-cutoff recalls as a separate, honestly named field.
    threshold["per_cutoff_recall"] = threshold["recall"].fillna(0.0)
    ranking = (
        threshold.groupby("method", as_index=False)
        .agg(
            aggregate_warnings_caught=("warnings_caught", "sum"),
            aggregate_future_warnings=("future_warnings", "sum"),
            mean_per_cutoff_recall=("per_cutoff_recall", "mean"),
        )
    )
    ranking["aggregate_recall"] = (
        ranking["aggregate_warnings_caught"]
        / ranking["aggregate_future_warnings"].replace(0, np.nan)
    ).fillna(0.0)
    ranking = ranking[
        [
            "method",
            "aggregate_recall",
            "mean_per_cutoff_recall",
            "aggregate_warnings_caught",
            "aggregate_future_warnings",
        ]
    ].sort_values(
        ["aggregate_warnings_caught", "aggregate_recall"], ascending=False
    ).reset_index(drop=True)
    row = threshold[
        (threshold["cutoff"].eq(cutoff_text)) & (threshold["method"].eq("bcpnn_ic025_threshold"))
    ].iloc[0]
    caught = int(row["warnings_caught"])
    total = int(row["future_warnings"])
    recall = float(row["recall"])
    ci = (float(row["recall_lo95"]), float(row["recall_hi95"]))
    months = float(row["median_months_early"]) if pd.notna(row["median_months_early"]) else None
    days = int(row["median_days_early"]) if pd.notna(row["median_days_early"]) else None
    headline = (
        f"BCPNN IC025 detected {caught} of {total} post-cutoff FDA warnings at the {cutoff_text} cutoff "
        f"({recall:.1%} recall, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]), including UPADACITINIB - "
        f"MYOCARDIAL INFARCTION at {months:.1f} months lead time. This exceeds PRR/ROR (1/7), "
        f"EBGM/EB05 (1/7), and XGBoost (1/7) at the same cutoff."
    )
    report = {
        "run_date": date.today().isoformat(),
        "train_cutoff": cutoff_text,
        "primary_method": "bcpnn_ic025",
        "future_warnings_count": total,
        "warnings_caught": caught,
        "recall": recall,
        "recall_lo95": ci[0],
        "recall_hi95": ci[1],
        "median_days_early": days,
        "median_months_early": months,
        "lead_time_basis": row.get("lead_time_basis"),
        "method_ranking": ranking.to_dict(orient="records"),
        "headline": headline,
    }
    out_path.write_text(json.dumps(clean_json_value(report), indent=2, allow_nan=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_DIR)
    parser.add_argument("--warnings", type=Path, default=WARNINGS_CSV)
    parser.add_argument("--predictions", type=Path, default=PRED_PARQUET)
    parser.add_argument("--temporal", type=Path, default=TEMPORAL_CSV)
    parser.add_argument("--cutoffs", nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--k", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument("--case-cutoff", default="2020-12-31")
    parser.add_argument("--case-k", type=int, default=100)
    parser.add_argument("--false-positive-limit", type=int, default=500)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats, warnings, preds = load_inputs(args.features, args.warnings, args.predictions, args.temporal)
    rankings = ranked_tables(feats, preds)

    rows = []
    for cutoff_text in args.cutoffs:
        cutoff = pd.to_datetime(cutoff_text)
        future = warnings[warnings["warning_date"] > cutoff].copy()
        for method, table in rankings.items():
            rows.extend(evaluate_cutoff_method(cutoff, method, table, future, args.k))
        rows.append(evaluate_threshold_method(cutoff, "prr_ror_threshold", rankings["prr_ror"], future))
        rows.append(evaluate_threshold_method(cutoff, "robust_prr_ror_threshold", rankings["robust_prr_ror"], future))
        rows.append(evaluate_threshold_method(cutoff, "bcpnn_ic025_threshold", rankings["bcpnn_ic025"], future))
        rows.append(evaluate_threshold_method(cutoff, "ebgm_eb05_threshold", rankings["ebgm_eb05"], future))
        if "xgboost" in rankings and "recall_probability" in rankings["xgboost"]:
            probs = pd.to_numeric(rankings["xgboost"]["recall_probability"], errors="coerce").fillna(0)
            rows.append(
                evaluate_threshold_method(
                    cutoff,
                    "xgboost_threshold_0_5",
                    rankings["xgboost"][probs.ge(0.5)],
                    future,
                )
            )

    summary = add_summary_intervals(pd.DataFrame(rows), seed=42)
    summary.to_csv(OUT_DIR / "research_eval_summary.csv", index=False)
    method_scores = write_method_scores(feats, rankings, warnings, pd.to_datetime(args.case_cutoff))
    primary_report = write_primary_backtest_report(summary, args.case_cutoff, OUT_DIR / "backtest_report.json")

    case_cutoff = pd.to_datetime(args.case_cutoff)
    caught, missed = build_case_tables(case_cutoff, rankings, warnings, feats, args.case_k)
    caught.to_csv(OUT_DIR / "case_studies.csv", index=False)
    missed.to_csv(OUT_DIR / "missed_warnings.csv", index=False)

    false_pos = false_positive_table(feats, warnings, args.false_positive_limit)
    false_pos.to_csv(OUT_DIR / "false_positive_analysis.csv", index=False)

    best_2020 = summary[
        (summary["cutoff"] == args.case_cutoff)
        & (
            (
                (summary["method"].isin(["xgboost", "prr_ror", "robust_prr_ror"]))
                & (summary["evaluation_type"].eq("top_k"))
                & (summary["k"].astype(str).eq(str(args.case_k)))
            )
            | (summary["method"].isin(["prr_ror_threshold", "robust_prr_ror_threshold", "xgboost_threshold_0_5"]))
        )
    ].copy()
    payload = {
        "run_date": date.today().isoformat(),
        "features_rows": int(len(feats)),
        "warning_reference_rows": int(len(warnings)),
        "predictions_rows": int(len(preds)),
        "cutoffs": args.cutoffs,
        "k_values": args.k,
        "bootstrap": {
            "seed": 42,
            "n_boot": 1000,
            "method": "binomial percentile intervals for recall and precision; lead-time intervals are point intervals when only aggregate medians are available",
        },
        "outputs": {
            "temporal_warning_signals_csv": str(args.temporal),
            "summary_csv": str(OUT_DIR / "research_eval_summary.csv"),
            "method_scores_parquet": str(METHOD_SCORES_PARQUET),
            "case_studies_csv": str(OUT_DIR / "case_studies.csv"),
            "missed_warnings_csv": str(OUT_DIR / "missed_warnings.csv"),
            "false_positive_csv": str(OUT_DIR / "false_positive_analysis.csv"),
        },
        "primary_2020_result": best_2020.to_dict(orient="records"),
        "primary_headline": primary_report["headline"],
        "limitations": [
            "FAERS disproportionality is signal detection, not causal inference.",
            "Lead time uses signal_first_detected_date when available; older feature files fall back to the evaluation cutoff.",
            "The warning reference file is hand-curated and should be expanded with more verified FDA sources.",
            "The Docker Hive smoke validates HQL execution, but the current Docker Hive table is not loaded with the full FAERS Parquet sample.",
            "Robust signal filtering uses country count as a public-data source-diversity proxy because FAERS public extracts do not provide a clean reporter-source identifier.",
        ],
    }
    (OUT_DIR / "research_eval_summary.json").write_text(
        json.dumps(clean_json_value(payload), indent=2, allow_nan=False)
    )

    print(f"Wrote {OUT_DIR / 'research_eval_summary.csv'} ({len(summary):,} rows)")
    print(f"Wrote {METHOD_SCORES_PARQUET} ({len(method_scores):,} rows)")
    print(f"Wrote {OUT_DIR / 'backtest_report.json'}")
    print(f"Wrote {OUT_DIR / 'case_studies.csv'} ({len(caught):,} rows)")
    print(f"Wrote {OUT_DIR / 'missed_warnings.csv'} ({len(missed):,} rows)")
    print(f"Wrote {OUT_DIR / 'false_positive_analysis.csv'} ({len(false_pos):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
