import pytest

from ml.delong import delong_test


def test_delong_equal_curves_have_unit_p_value():
    y = [1, 1, 0, 0, 0, 1]
    scores = [0.9, 0.8, 0.2, 0.3, 0.1, 0.7]
    auc_a, auc_b, z, p = delong_test(y, scores, scores)
    assert auc_a == pytest.approx(auc_b)
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_delong_detects_clearly_different_rankings():
    y = [1] * 50 + [0] * 50
    good = list(range(100, 50, -1)) + list(range(50, 0, -1))
    bad = list(range(50, 0, -1)) + list(range(100, 50, -1))
    auc_good, auc_bad, _, p = delong_test(y, good, bad)
    assert auc_good > auc_bad
    assert p < 0.05
