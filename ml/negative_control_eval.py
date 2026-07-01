"""ml/negative_control_eval.py — false-positive rate on negative controls.

For each method, checks how many of the 20 curated negative-control
drug-reaction pairs would trigger a signal at the method's threshold on the
committed signals.csv snapshot. Compares to recall on curated positives.

This is a standard PV benchmark technique: a good method should have low
FP rate on negatives and high recall on positives; the ratio measures
specificity in a domain-meaningful way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ROOT / "dashboard" / "data" / "signals.csv"
NEG = ROOT / "data" / "reference" / "negative_controls.csv"
POS = ROOT / "data" / "reference" / "fda_warnings.csv"
OUT = ROOT / "data" / "processed" / "negative_control_results.csv"


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)


def _key(df: pd.DataFrame) -> pd.Series:
    return _norm(df["drug_name"]) + "||" + _norm(df["reaction_term"])


# Method thresholds: mirror the ones used in research_evaluate.py
# and paper Section 4.4. Only threshold methods evaluated here (top_k
# evaluation doesn't apply to a fixed 20-pair set).
METHODS = {
    "prr_ror_threshold": lambda s: (s["prr"] > 2) & (s["ror"] > 2) & (s["case_count"] >= 3) & (s["prr_chi_square"] >= 4),
    "robust_prr_ror_threshold": lambda s: s.get("passes_robust_filter", False) == True,
    # bcpnn/ebgm are on the feature parquet not signals.csv; report as N/A here
}


def main() -> int:
    signals = pd.read_csv(SIGNALS)
    neg = pd.read_csv(NEG)
    pos = pd.read_csv(POS)

    signals["pair_key"] = _key(signals)
    neg_keys = set(_key(neg).tolist())
    pos_keys = set(_key(pos).tolist())

    n_neg = len(neg)
    n_pos = len(pos)

    rows = []
    for method, rule in METHODS.items():
        try:
            triggered = signals[rule(signals)]["pair_key"].astype(str).tolist()
        except Exception as exc:  # noqa: BLE001
            print(f"  {method}: skipped ({exc})")
            continue
        triggered_set = set(triggered)
        fp = len(triggered_set & neg_keys)
        tp = len(triggered_set & pos_keys)
        fp_rate = fp / n_neg if n_neg else 0.0
        recall = tp / n_pos if n_pos else 0.0
        # Specificity estimate is limited: we cannot see all true negatives in
        # snapshot mode; we report the estimator honestly as
        # 1 - fp_rate_on_curated_negatives.
        specificity = 1 - fp_rate
        rows.append(
            {
                "method": method,
                "n_negatives_curated": n_neg,
                "n_positives_curated": n_pos,
                "fp_on_negatives": fp,
                "tp_on_positives": tp,
                "fp_rate_on_negatives": round(fp_rate, 4),
                "recall_on_positives": round(recall, 4),
                "specificity_estimate": round(specificity, 4),
                "note": "specificity restricted to curated negative controls (n=20); not a full-population estimate.",
            }
        )

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
