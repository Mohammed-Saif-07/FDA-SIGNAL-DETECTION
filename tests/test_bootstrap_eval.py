import numpy as np

from ml.bootstrap_eval import bootstrap_metric, binomial_ci


def test_bootstrap_metric_returns_ordered_ci():
    y = np.array([0, 1, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    point, lo, hi = bootstrap_metric(y, score, lambda yt, ys: float((ys[yt == 1] > 0.5).mean()), n_boot=50)
    assert lo <= point <= hi


def test_binomial_ci_bounds_point_estimate():
    lo, hi = binomial_ci(1, 7, n_boot=200, seed=42)
    assert lo <= 1 / 7 <= hi

