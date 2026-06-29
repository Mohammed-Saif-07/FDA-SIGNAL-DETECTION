"""Bootstrap utilities for reproducible evaluation intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_metric(
    y_true,
    y_score,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return point estimate and percentile 95% CI for a metric.

    The function is intentionally generic so tests and future methods can reuse
    it. If a resample is degenerate or the metric cannot be computed, that
    resample is skipped.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(metric_fn(y_true, y_score))
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        try:
            value = float(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
        if np.isfinite(value):
            vals.append(value)

    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def binomial_ci(successes: int, trials: int, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap percentile CI for a binomial proportion."""

    if trials <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.binomial(trials, successes / trials, size=n_boot) / trials
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def median_ci(values, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap percentile CI for a median."""

    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype="float64")
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = [float(np.median(arr[rng.integers(0, len(arr), size=len(arr))])) for _ in range(n_boot)]
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(lo), float(hi)

