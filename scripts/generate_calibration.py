"""scripts/generate_calibration.py — XGBoost reliability diagram.

Loads dashboard/data/predictions.csv, treats rows with a non-null
``actual_fda_warning_date`` as positives, bins ``recall_probability`` into
deciles, and plots predicted vs empirical positive rate.

At the current positive rate (very small), the calibration plot will look
degenerate; we annotate the figure honestly rather than hide the situation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / "dashboard" / "data" / "predictions.csv"
FIG_PNG = ROOT / "docs" / "figures" / "xgboost_calibration.png"
FIG_SVG = ROOT / "docs" / "figures" / "xgboost_calibration.svg"


def main() -> int:
    df = pd.read_csv(PRED_CSV)
    p = pd.to_numeric(df["recall_probability"], errors="coerce")
    if "actual_fda_warning_date" in df.columns:
        # Real positives = rows with a non-null, non-empty warning date.
        raw = df["actual_fda_warning_date"]
        y = raw.notna() & raw.astype(str).str.strip().ne("").astype(bool) & raw.astype(str).str.lower().ne("nan")
    else:
        y = pd.Series(False, index=df.index)

    mask = p.notna()
    p = p[mask].to_numpy()
    y = y[mask].astype(int).to_numpy()

    n_pos = int(y.sum())
    n_total = int(len(y))

    # Decile bins
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    df_bin = pd.DataFrame({"p": p, "y": y, "bin": idx})
    agg = df_bin.groupby("bin").agg(
        mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size")
    ).reindex(range(10)).reset_index()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 6.0), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Reliability
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    valid = agg["mean_pred"].notna()
    ax1.plot(
        agg.loc[valid, "mean_pred"],
        agg.loc[valid, "obs_rate"],
        "o-",
        color="#1f77b4",
        markersize=6,
        label=f"XGBoost (n={n_total:,}, pos={n_pos})",
    )
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Empirical positive rate")
    ax1.set_title("XGBoost reliability diagram")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    ax1.grid(alpha=0.3)

    if n_pos < 20:
        ax1.text(
            0.5,
            0.5,
            (
                f"n_positive = {n_pos}\n"
                f"reliable calibration assessment\nnot possible at this positive rate"
            ),
            transform=ax1.transAxes,
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2cc", edgecolor="#d6b656"),
        )

    # Predicted probability histogram
    ax2.hist(p, bins=bins, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim(-0.02, 1.02)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"n_total={n_total} n_positive={n_pos}")
    print(f"Wrote {FIG_PNG}")
    print(f"Wrote {FIG_SVG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
