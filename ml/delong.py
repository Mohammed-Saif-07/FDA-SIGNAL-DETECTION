"""DeLong ROC AUC comparison for paired method scores.

Implements the midrank formulation described by:
  DeLong ER, DeLong DM, Clarke-Pearson DL. Biometrics. 1988;44:837-845.
  Sun X, Xu W. IEEE Signal Processing Letters. 2014;21(11):1389-1393.

The public interface is intentionally small so the statistical comparison used
by the paper remains auditable and dependency-free.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(len(x), dtype=float)
    out[order] = midranks
    return out


def _fast_delong(predictions_sorted: np.ndarray, n_positive: int) -> tuple[np.ndarray, np.ndarray]:
    n_methods, n_examples = predictions_sorted.shape
    n_negative = n_examples - n_positive
    if n_positive <= 0 or n_negative <= 0:
        raise ValueError("DeLong test requires at least one positive and one negative example")

    positive = predictions_sorted[:, :n_positive]
    negative = predictions_sorted[:, n_positive:]

    tx = np.empty((n_methods, n_positive), dtype=float)
    ty = np.empty((n_methods, n_negative), dtype=float)
    tz = np.empty((n_methods, n_examples), dtype=float)
    for method_idx in range(n_methods):
        tx[method_idx] = _compute_midrank(positive[method_idx])
        ty[method_idx] = _compute_midrank(negative[method_idx])
        tz[method_idx] = _compute_midrank(predictions_sorted[method_idx])

    aucs = tz[:, :n_positive].sum(axis=1) / n_positive / n_negative - (n_positive + 1.0) / (2.0 * n_negative)
    v01 = (tz[:, :n_positive] - tx) / n_negative
    v10 = 1.0 - (tz[:, n_positive:] - ty) / n_positive
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / n_positive + sy / n_negative
    covariance = np.atleast_2d(covariance)
    return aucs, covariance


def delong_test(y_true, scores_a, scores_b) -> tuple[float, float, float, float]:
    """Return ``(auc_a, auc_b, z, p_value)`` for two correlated ROC curves."""

    y = np.asarray(y_true).astype(int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & np.isin(y, [0, 1])
    y = y[valid]
    a = a[valid]
    b = b[valid]
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        raise ValueError("DeLong test requires finite scores with both classes present")

    order = np.argsort(-y)
    predictions_sorted = np.vstack([a, b])[:, order]
    aucs, covariance = _fast_delong(predictions_sorted, int(y.sum()))
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T).item())
    auc_a = float(aucs[0])
    auc_b = float(aucs[1])
    if variance <= 0 or not math.isfinite(variance):
        z = 0.0 if math.isclose(auc_a, auc_b) else math.copysign(math.inf, auc_a - auc_b)
        p = 1.0 if math.isfinite(z) else 0.0
        return auc_a, auc_b, z, p
    z = (auc_a - auc_b) / math.sqrt(variance)
    p = 2.0 * norm.sf(abs(z))
    return auc_a, auc_b, float(z), float(p)


def auc(y_true, scores) -> float:
    """Small wrapper used by tests and reports."""

    return float(roc_auc_score(y_true, scores))
