"""Method comparison utilities for paper artifacts.

The warning reference set is small, so these tests should be interpreted as
descriptive diagnostics rather than strong evidence of superiority.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from delong import delong_test


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data" / "processed" / "research_eval_summary.csv"
METHOD_SCORES = ROOT / "data" / "processed" / "method_scores_2020.parquet"
OUT = ROOT / "data" / "processed"


def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    return float(binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue)


def main() -> int:
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing {SUMMARY}; run make research-eval first")
    df = pd.read_csv(SUMMARY)
    threshold = df[df["evaluation_type"].eq("threshold")].copy()
    methods = sorted(threshold["method"].dropna().unique())

    # Aggregate "caught at least one warning in a cutoff" as the paired binary
    # outcome. This is coarse, but transparent with the current aggregate files.
    wide = threshold.pivot_table(
        index="cutoff",
        columns="method",
        values="warnings_caught",
        aggfunc="max",
        fill_value=0,
    )
    caught_binary = (wide > 0).astype(int)
    mcnemar = pd.DataFrame(1.0, index=methods, columns=methods)
    for a, b in combinations(methods, 2):
        b01 = int(((caught_binary[a] == 0) & (caught_binary[b] == 1)).sum())
        b10 = int(((caught_binary[a] == 1) & (caught_binary[b] == 0)).sum())
        p = mcnemar_exact(b01, b10)
        mcnemar.loc[a, b] = p
        mcnemar.loc[b, a] = p
    mcnemar.to_csv(OUT / "mcnemar_pvalues.csv")

    if not METHOD_SCORES.exists():
        raise FileNotFoundError(f"Missing {METHOD_SCORES}; run make research-eval first")
    scores = pd.read_parquet(METHOD_SCORES)
    score_methods = [
        col
        for col in scores.columns
        if col not in {"pair_key", "drug_name", "reaction_term", "is_post_cutoff_warning"}
    ]
    y_true = scores["is_post_cutoff_warning"].astype(int).to_numpy()
    delong = pd.DataFrame(1.0, index=score_methods, columns=score_methods)
    for a, b in combinations(score_methods, 2):
        try:
            _, _, _, p = delong_test(y_true, scores[a].to_numpy(), scores[b].to_numpy())
        except ValueError:
            p = np.nan
        delong.loc[a, b] = p
        delong.loc[b, a] = p
    delong.to_csv(OUT / "delong_pvalues.csv")
    readme = OUT / "delong_pvalues.README.txt"
    if readme.exists():
        readme.unlink()
    print(f"Wrote {OUT / 'mcnemar_pvalues.csv'}")
    print(f"Wrote {OUT / 'delong_pvalues.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
