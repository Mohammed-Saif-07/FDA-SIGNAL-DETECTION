"""Unit test for ml/permutation_test.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from permutation_test import ranking_stability  # noqa: E402


def _synthetic() -> pd.DataFrame:
    """Three methods with clearly ordered catch counts across 4 cutoffs."""
    rows = []
    for cutoff in ("2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"):
        rows.append(
            {"evaluation_type": "threshold", "cutoff": cutoff, "method": "best", "warnings_caught": 5}
        )
        rows.append(
            {"evaluation_type": "threshold", "cutoff": cutoff, "method": "middle", "warnings_caught": 2}
        )
        rows.append(
            {"evaluation_type": "threshold", "cutoff": cutoff, "method": "worst", "warnings_caught": 0}
        )
    return pd.DataFrame(rows)


def test_best_method_ranks_first_almost_always():
    summary = _synthetic()
    out = ranking_stability(summary, n_boot=500, seed=42)
    row = out[out["method"] == "best"].iloc[0]
    assert row["p_rank1"] > 0.95, f"best method p_rank1 too low: {row['p_rank1']}"
    assert row["mean_rank"] < 1.1, f"best method mean_rank too high: {row['mean_rank']}"


def test_worst_method_ranks_last_almost_always():
    summary = _synthetic()
    out = ranking_stability(summary, n_boot=500, seed=42)
    row = out[out["method"] == "worst"].iloc[0]
    assert row["p_rank3"] > 0.95, f"worst method p_rank3 too low: {row['p_rank3']}"
