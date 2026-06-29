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


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data" / "processed" / "research_eval_summary.csv"
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

    # DeLong requires individual y_true/y_score arrays per method. The current
    # research summary is aggregate, so write an explicit non-applicable matrix
    # instead of fabricating p-values.
    delong = pd.DataFrame(np.nan, index=methods, columns=methods)
    for method in methods:
        delong.loc[method, method] = 1.0
    delong.to_csv(OUT / "delong_pvalues.csv")
    (OUT / "delong_pvalues.README.txt").write_text(
        "DeLong AUC comparison is not computed from aggregate research_eval_summary.csv. "
        "Run a future individual-score export to enable this test.\n"
    )
    print(f"Wrote {OUT / 'mcnemar_pvalues.csv'}")
    print(f"Wrote {OUT / 'delong_pvalues.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

