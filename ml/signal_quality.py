"""Signal-quality filters for FAERS disproportionality outputs.

The raw PRR/ROR threshold is intentionally sensitive, but it also surfaces many
obvious artifacts: very narrow products, low-seriousness consumer reports, or
single-country report clusters with extreme ratios. FAERS public extracts do not
include a clean reporter-source identifier, so this module uses conservative
observable proxies instead of pretending to measure true source count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "processed" / "ml_features"
OUT_CSV = ROOT / "data" / "processed" / "signal_quality_diagnosis.csv"

MIN_CASES = 5
MIN_COUNTRIES = 3
MIN_SERIOUS_RATIO = 0.01
MAX_REASONABLE_PRR = 100_000.0
MAX_REASONABLE_ROR = 100_000.0


def add_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with robust-filter columns.

    Columns added:
      - source_proxy_count: currently countries_count, because reporter-source
        IDs are unavailable in the public flattened table.
      - artifact_reason: first failing quality rule.
      - passes_robust_filter: conservative pass/fail flag for ranking.
      - robust_signal_score: ranking score for signals that pass the filter.
    """

    out = df.copy()
    for col in [
        "case_count",
        "drug_total",
        "reaction_total",
        "countries_count",
        "serious_ratio",
        "death_ratio",
        "prr",
        "ror",
        "prr_chi_square",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["source_proxy_count"] = out.get("countries_count", pd.Series(0, index=out.index)).fillna(0).astype(int)
    out["is_single_source_proxy"] = out["source_proxy_count"].le(1)
    if "drug_total" not in out.columns and {"drug_name", "case_count"}.issubset(out.columns):
        out["drug_total"] = out.groupby("drug_name")["case_count"].transform("sum")
    if "reaction_total" not in out.columns and {"reaction_term", "case_count"}.issubset(out.columns):
        out["reaction_total"] = out.groupby("reaction_term")["case_count"].transform("sum")

    signal_mask = (
        out["case_count"].ge(3)
        & out["prr"].gt(2)
        & out["ror"].gt(2)
        & out["prr_chi_square"].ge(4)
    )
    rules = [
        (~signal_mask, "not_prr_ror_signal"),
        (out["case_count"].lt(MIN_CASES), "too_few_cases"),
        (out["source_proxy_count"].lt(MIN_COUNTRIES), "low_source_diversity_proxy"),
        (out["serious_ratio"].fillna(0).lt(MIN_SERIOUS_RATIO), "low_seriousness_ratio"),
        (out["prr"].gt(MAX_REASONABLE_PRR) | out["ror"].gt(MAX_REASONABLE_ROR), "extreme_ratio_artifact"),
        (out["prr_chi_square"].lt(0), "invalid_negative_chi_square"),
    ]
    out["artifact_reason"] = "passes_robust_filter"
    for mask, reason in rules:
        out.loc[out["artifact_reason"].eq("passes_robust_filter") & mask.fillna(True), "artifact_reason"] = reason

    out["passes_robust_filter"] = out["artifact_reason"].eq("passes_robust_filter")
    drug_total = out.get("drug_total", pd.Series(np.nan, index=out.index))
    reaction_total = out.get("reaction_total", pd.Series(np.nan, index=out.index))
    out["passes_structural_filter"] = (
        out["passes_robust_filter"]
        & pd.to_numeric(drug_total, errors="coerce").gt(out["case_count"])
        & pd.to_numeric(reaction_total, errors="coerce").gt(out["case_count"])
    )
    serious_component = out["serious_ratio"].fillna(0).clip(lower=0, upper=1)
    death_component = out["death_ratio"].fillna(0).clip(lower=0, upper=1)
    out["robust_signal_score"] = (
        np.log1p(out["case_count"].fillna(0))
        * np.log1p(out["source_proxy_count"].fillna(0))
        * (1.0 + serious_component + 2.0 * death_component)
        * np.log1p(out["prr"].clip(lower=0, upper=MAX_REASONABLE_PRR).fillna(0))
        * np.log1p(out["ror"].clip(lower=0, upper=MAX_REASONABLE_ROR).fillna(0))
    )
    out.loc[~out["passes_robust_filter"], "robust_signal_score"] = 0.0
    return out


def summarize_quality(df: pd.DataFrame) -> pd.DataFrame:
    quality = add_signal_quality(df)
    total_signals = int(
        (
            quality["case_count"].ge(3)
            & quality["prr"].gt(2)
            & quality["ror"].gt(2)
        ).sum()
    )
    robust = int(quality["passes_robust_filter"].sum())
    structural = int(quality["passes_structural_filter"].sum())
    rows = [
        {"metric": "rows", "value": int(len(quality))},
        {"metric": "raw_prr_ror_signals", "value": total_signals},
        {"metric": "robust_signals", "value": robust},
        {"metric": "structural_signals", "value": structural},
        {"metric": "proxy_only_signals", "value": robust - structural},
        {
            "metric": "robust_pass_rate_among_raw_signals",
            "value": robust / total_signals if total_signals else 0.0,
        },
    ]
    reason_counts = quality["artifact_reason"].value_counts(dropna=False)
    rows.extend({"metric": f"artifact_reason:{reason}", "value": int(count)} for reason, count in reason_counts.items())
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_DIR)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    df = pd.read_parquet(
        args.features,
        columns=[
            "drug_name",
            "reaction_term",
            "case_count",
            "countries_count",
            "serious_ratio",
            "death_ratio",
            "prr",
            "ror",
            "prr_chi_square",
        ],
    )
    summary = summarize_quality(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
