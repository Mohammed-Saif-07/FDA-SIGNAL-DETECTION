"""ml/permutation_test.py — Ranking stability by bootstrap resampling of cutoffs.

For each of ``n_boot`` bootstrap resamples of the per-cutoff evaluation results,
we recompute each method's aggregate warnings caught and rank methods by that.
We then report, for each method, the fraction of resamples in which the method
achieved rank 1, 2, 3, ..., and its mean rank across resamples.

This is the honest small-sample rigor check: it tells us how confidently we
can say BCPNN IC025 is the best method given only 3-4 usable cutoffs.

Reference:
    Efron & Tibshirani (1993). An Introduction to the Bootstrap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ranking_stability(
    summary: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    metric: str = "warnings_caught",
) -> pd.DataFrame:
    """Bootstrap the cutoff axis to measure per-method rank stability.

    Parameters
    ----------
    summary : DataFrame from ``research_eval_summary.csv``. Must contain
        ``method``, ``cutoff``, and ``warnings_caught`` for threshold rows.
    n_boot : number of bootstrap resamples.
    seed : RNG seed.
    metric : which column to sum per method per resample; default warnings_caught.
    """
    threshold = summary[summary["evaluation_type"].eq("threshold")].copy()
    cutoffs = threshold["cutoff"].unique().tolist()
    methods = sorted(threshold["method"].unique().tolist())
    if not cutoffs or not methods:
        return pd.DataFrame()

    # Wide table: rows = cutoff, cols = method, values = warnings_caught
    wide = (
        threshold.pivot_table(
            index="cutoff", columns="method", values=metric, aggfunc="sum"
        )
        .fillna(0)
        .astype(float)
    )
    cutoffs = wide.index.tolist()

    rng = np.random.default_rng(seed)
    # rank counters: method -> {rank -> count}
    rank_counts = {m: np.zeros(len(methods), dtype=int) for m in methods}
    rank_sums = {m: 0.0 for m in methods}

    for _ in range(n_boot):
        idx = rng.integers(0, len(cutoffs), size=len(cutoffs))
        sampled = wide.iloc[idx]
        totals = sampled.sum(axis=0)
        # rank 1 = highest; ties broken by method name (deterministic)
        # We use pandas rank with ascending=False; average method for ties
        # for a smooth summary.
        ranks = totals.rank(method="average", ascending=False)
        for m in methods:
            r = ranks[m]
            rank_sums[m] += r
            # nearest integer bucket (1..len(methods))
            r_int = int(round(r)) - 1
            r_int = min(max(r_int, 0), len(methods) - 1)
            rank_counts[m][r_int] += 1

    rows = []
    for m in methods:
        row = {"method": m, "mean_rank": rank_sums[m] / n_boot}
        for r in range(len(methods)):
            row[f"p_rank{r+1}"] = rank_counts[m][r] / n_boot
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mean_rank")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/research_eval_summary.csv")
    ap.add_argument("--out", default="data/processed/method_ranking_stability.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary = pd.read_csv(args.inp)
    result = ranking_stability(summary, n_boot=args.n_boot, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"n_boot={args.n_boot} seed={args.seed}")
    print(result.to_string(index=False))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
