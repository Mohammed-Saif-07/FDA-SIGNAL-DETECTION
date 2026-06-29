import numpy as np
import pandas as pd

from ml.bcpnn import add_bcpnn_scores


def test_bcpnn_ic025_increases_with_observed_enrichment():
    df = pd.DataFrame(
        {
            "case_count": [2, 20],
            "drug_total": [100, 100],
            "reaction_total": [200, 200],
            "grand_total": [10_000, 10_000],
        }
    )
    out = add_bcpnn_scores(df)
    assert np.isfinite(out.loc[0, "ic025"])
    assert out.loc[1, "ic025"] > out.loc[0, "ic025"]
    assert out.loc[1, "ic"] > 0


def test_bcpnn_non_enriched_pair_is_not_positive_signal():
    df = pd.DataFrame(
        {
            "case_count": [2],
            "drug_total": [100],
            "reaction_total": [200],
            "grand_total": [10_000],
        }
    )
    out = add_bcpnn_scores(df)
    assert out.loc[0, "ic025"] < 0

