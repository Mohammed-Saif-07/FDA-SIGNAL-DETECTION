import pandas as pd

from ml.signal_quality import add_signal_quality


def test_robust_filter_rejects_low_source_extreme_artifact():
    df = pd.DataFrame(
        {
            "case_count": [3],
            "countries_count": [1],
            "serious_ratio": [0.0],
            "death_ratio": [0.0],
            "prr": [300_000.0],
            "ror": [400_000.0],
            "prr_chi_square": [999.0],
        }
    )
    out = add_signal_quality(df)
    assert not bool(out.loc[0, "passes_robust_filter"])
    assert out.loc[0, "artifact_reason"] in {
        "too_few_cases",
        "low_source_diversity_proxy",
        "low_seriousness_ratio",
        "extreme_ratio_artifact",
    }


def test_robust_filter_accepts_multicountry_serious_signal():
    df = pd.DataFrame(
        {
            "case_count": [24],
            "drug_total": [100],
            "reaction_total": [80],
            "countries_count": [3],
            "serious_ratio": [1.0],
            "death_ratio": [0.08],
            "prr": [3.5],
            "ror": [3.6],
            "prr_chi_square": [40.0],
        }
    )
    out = add_signal_quality(df)
    assert bool(out.loc[0, "passes_robust_filter"])
    assert bool(out.loc[0, "passes_structural_filter"])
    assert out.loc[0, "robust_signal_score"] > 0


def test_structural_filter_rejects_margin_saturated_pair():
    df = pd.DataFrame(
        {
            "case_count": [24],
            "drug_total": [24],
            "reaction_total": [24],
            "countries_count": [5],
            "serious_ratio": [1.0],
            "death_ratio": [0.0],
            "prr": [20.0],
            "ror": [20.0],
            "prr_chi_square": [40.0],
        }
    )
    out = add_signal_quality(df)
    assert bool(out.loc[0, "passes_robust_filter"])
    assert not bool(out.loc[0, "passes_structural_filter"])
