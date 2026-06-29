"""Threshold sensitivity analysis for the 2020 cutoff."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bcpnn import add_bcpnn_scores
from ebgm import add_ebgm_scores
from signal_quality import add_signal_quality


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "ml_features"
WARNINGS = ROOT / "data" / "reference" / "fda_warnings.csv"
OUT = ROOT / "data" / "processed" / "sensitivity_grid.csv"
FIG = ROOT / "docs" / "figures" / "sensitivity.png"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["drug_name"] = out["drug_name"].astype(str).str.upper().str.strip()
    out["reaction_term"] = out["reaction_term"].astype(str).str.upper().str.strip()
    out["pair_key"] = out["drug_name"] + "||" + out["reaction_term"]
    return out


def ensure_margins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "drug_total" not in out.columns:
        out["drug_total"] = out.groupby("drug_name")["case_count"].transform("sum")
    if "reaction_total" not in out.columns:
        out["reaction_total"] = out.groupby("reaction_term")["case_count"].transform("sum")
    if "grand_total" not in out.columns:
        out["grand_total"] = float(out["case_count"].sum())
    return out


def row(method: str, name: str, value: float, candidates: pd.DataFrame, future_keys: set[str], future_count: int):
    hits = len(set(candidates["pair_key"]).intersection(future_keys))
    n = len(candidates)
    return {
        "method": method,
        "threshold_name": name,
        "threshold_value": value,
        "recall": hits / future_count if future_count else np.nan,
        "precision": hits / n if n else 0.0,
        "n_signals": n,
    }


def main() -> int:
    if not FEATURES.exists():
        raise FileNotFoundError(f"Missing {FEATURES}; run feature engineering first")
    cols = ["drug_name", "reaction_term", "case_count", "countries_count", "serious_ratio", "death_ratio", "prr", "ror", "prr_chi_square"]
    feats = normalize(pd.read_parquet(FEATURES, columns=cols))
    feats = ensure_margins(feats)
    feats = add_signal_quality(add_ebgm_scores(add_bcpnn_scores(feats)))
    warnings = normalize(pd.read_csv(WARNINGS, parse_dates=["warning_date"]))
    future = warnings[warnings["warning_date"] > pd.Timestamp("2020-12-31")]
    future_keys = set(future["pair_key"])

    rows = []
    for prr in [1.5, 2.0, 3.0, 5.0]:
        for min_cases in [3, 5, 10]:
            cand = feats[feats["case_count"].ge(min_cases) & feats["prr"].gt(prr) & feats["ror"].gt(2.0)]
            rows.append(row("prr_ror", f"prr>{prr};cases>={min_cases}", prr, cand, future_keys, len(future)))
    for threshold in [0.0, 0.5, 1.0]:
        rows.append(row("bcpnn_ic025", "ic025", threshold, feats[feats["ic025"].gt(threshold)], future_keys, len(future)))
    for threshold in [1.5, 2.0, 3.0]:
        rows.append(row("ebgm_eb05", "eb05", threshold, feats[feats["eb05"].gt(threshold)], future_keys, len(future)))

    grid = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT, index=False)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    for method, group in grid.groupby("method"):
        axes[0].plot(range(len(group)), group["recall"], marker="o", label=method)
        axes[1].plot(range(len(group)), group["precision"], marker="o", label=method)
    axes[0].set_title("Recall across thresholds")
    axes[1].set_title("Precision across thresholds")
    for ax in axes:
        ax.set_xlabel("threshold setting index")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("recall")
    axes[1].set_ylabel("precision")
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)
    print(f"Wrote {OUT}")
    print(f"Wrote {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

