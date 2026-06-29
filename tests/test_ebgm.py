import pytest
import pandas as pd

from ml.ebgm import add_ebgm_scores


def test_ebgm_sparse_artifact_is_shrunk_but_still_high():
    df = pd.DataFrame(
        {
            "case_count": [3],
            "drug_total": [3],
            "reaction_total": [3],
            "grand_total": [1_000_000],
        }
    )
    out = add_ebgm_scores(df)
    assert out.loc[0, "eb05"] == pytest.approx(2.17, rel=0.10)


def test_ebgm_genuine_medium_signal_crosses_two():
    df = pd.DataFrame(
        {
            "case_count": [400],
            "drug_total": [9_000],
            "reaction_total": [20_000],
            "grand_total": [1_000_000],
        }
    )
    out = add_ebgm_scores(df)
    assert out.loc[0, "eb05"] == pytest.approx(2.04, rel=0.05)

