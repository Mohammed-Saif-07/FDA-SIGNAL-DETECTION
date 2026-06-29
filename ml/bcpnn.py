"""BCPNN Information Component baseline for pharmacovigilance signals.

This module implements a lightweight, reproducible Information Component (IC)
and IC025 approximation. It is intended as an open-source baseline alongside
PRR/ROR, not as a replacement for proprietary pharmacovigilance systems.

Reference:
    Bate et al. (1998), "A Bayesian neural network method for adverse drug
    reaction signal generation", European Journal of Clinical Pharmacology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LOG2 = np.log(2.0)


def _series(value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    return pd.Series(value, index=index, dtype="float64")


def add_bcpnn_scores(
    df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    """Return a copy with BCPNN ``ic`` and ``ic025`` columns.

    Inputs expected in ``df``:
      - case_count: observed drug/reaction count ``a``
      - drug_total: marginal count for the drug
      - reaction_total: marginal count for the reaction
      - grand_total: total number of drug/reaction observations

    The IC estimate uses a smoothed observed/expected ratio:
        IC = log2((a + alpha) / (E + beta))
        E = drug_total * reaction_total / grand_total

    IC025 uses a delta-method normal approximation. This is transparent and
    deterministic; it should be treated as an approximation for benchmarking.
    """

    out = df.copy()
    idx = out.index
    a = _series(out.get("case_count", 0.0), idx).clip(lower=0)
    drug_total = _series(out.get("drug_total", a), idx).clip(lower=0)
    reaction_total = _series(out.get("reaction_total", a), idx).clip(lower=0)
    grand_total = _series(out.get("grand_total", max(float(a.sum()), 1.0)), idx).replace(0, np.nan)

    expected = (drug_total * reaction_total / grand_total).replace([np.inf, -np.inf], np.nan)
    observed = a + alpha
    expected_smoothed = expected + beta
    ic = np.log(observed / expected_smoothed) / LOG2

    # Delta-method variance for log2(observed / expected). The marginal
    # uncertainty terms are approximated from the observed count and expected
    # cell count so the result is stable on sparse public FAERS tables.
    var = (1.0 / LOG2**2) * ((1.0 / observed.clip(lower=1e-9)) + (1.0 / expected_smoothed.clip(lower=1e-9)))
    ic025 = ic - 1.96 * np.sqrt(var)

    out["ic"] = pd.Series(ic, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ic025"] = pd.Series(ic025, index=idx).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    return out

