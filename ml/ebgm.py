"""Simplified EBGM/EB05 baseline for FAERS disproportionality analysis.

This is a single-component Gamma-Poisson shrinkage baseline inspired by the
MGPS family of methods. It is intentionally documented as simplified because
FDA's production MGPS uses a richer empirical Bayes mixture model.

Reference:
    DuMouchel (1999), "Bayesian data mining in large frequency tables, with an
    application to the FDA spontaneous reporting system", The American
    Statistician.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma


def _series(value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    return pd.Series(value, index=index, dtype="float64")


def add_ebgm_scores(
    df: pd.DataFrame,
    alpha1: float = 0.5,
    beta1: float = 0.5,
) -> pd.DataFrame:
    """Return a copy with ``ebgm`` and ``eb05`` columns.

    Expected count:
        E_ij = drug_total_i * reaction_total_j / grand_total

    Posterior for the reporting ratio is approximated as:
        Gamma(alpha1 + observed, rate=beta1 + expected)

    ``eb05`` is the posterior 5th percentile. A threshold EB05 > 2.0 is a
    common conservative signal criterion in MGPS-style workflows.
    """

    out = df.copy()
    idx = out.index
    observed = _series(out.get("case_count", 0.0), idx).clip(lower=0)
    drug_total = _series(out.get("drug_total", observed), idx).clip(lower=0)
    reaction_total = _series(out.get("reaction_total", observed), idx).clip(lower=0)
    grand_total = _series(out.get("grand_total", max(float(observed.sum()), 1.0)), idx).replace(0, np.nan)

    expected = (drug_total * reaction_total / grand_total).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shape = alpha1 + observed
    rate = beta1 + expected
    scale = 1.0 / rate.clip(lower=1e-12)

    out["ebgm"] = (shape * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["eb05"] = pd.Series(gamma.ppf(0.05, a=shape, scale=scale), index=idx).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return out

