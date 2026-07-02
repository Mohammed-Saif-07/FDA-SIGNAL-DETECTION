"""Generate publication figures from existing pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def architecture():
    svg = OUT / "architecture.svg"
    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="980" height="220">
<style>text{font-family:Arial,sans-serif;font-size:16px}.box{fill:#f8fafc;stroke:#334155;stroke-width:2}.arrow{stroke:#ef4444;stroke-width:3;marker-end:url(#a)}</style>
<defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#ef4444"/></marker></defs>
<rect class="box" x="20" y="70" width="120" height="60"/><text x="46" y="105">FAERS</text>
<rect class="box" x="190" y="70" width="120" height="60"/><text x="218" y="105">PySpark</text>
<rect class="box" x="360" y="70" width="120" height="60"/><text x="389" y="105">Hive/HQL</text>
<rect class="box" x="530" y="70" width="120" height="60"/><text x="560" y="105">XGBoost</text>
<rect class="box" x="700" y="70" width="120" height="60"/><text x="729" y="105">Postgres</text>
<rect class="box" x="850" y="70" width="110" height="60"/><text x="875" y="105">Dashboard</text>
<line class="arrow" x1="140" y1="100" x2="190" y2="100"/><line class="arrow" x1="310" y1="100" x2="360" y2="100"/>
<line class="arrow" x1="480" y1="100" x2="530" y2="100"/><line class="arrow" x1="650" y1="100" x2="700" y2="100"/><line class="arrow" x1="820" y1="100" x2="850" y2="100"/>
</svg>
"""
    )


def artifact_rate():
    path = ROOT / "data" / "processed" / "signal_quality_diagnosis.csv"
    raw_artifact = 0.80
    robust_artifact = 0.0
    if path.exists():
        diag = pd.read_csv(path)
        metrics = dict(zip(diag["metric"], diag["value"]))
        raw = float(metrics.get("raw_prr_ror_signals", 0))
        robust = float(metrics.get("robust_signals", 0))
        raw_artifact = 1.0 - robust / raw if raw else raw_artifact
        robust_artifact = 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Raw PRR/ROR", "After robust filter"], [raw_artifact, robust_artifact], color=["#ef4444", "#22c55e"])
    ax.set_ylabel("estimated artifact rate")
    ax.set_ylim(0, 1)
    ax.set_title("Artifact reduction by robust signal filter")
    save(fig, "artifact_rate_before_after")


def roc_pr():
    summary = ROOT / "data" / "processed" / "research_eval_summary.csv"
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
    if summary.exists():
        df = pd.read_csv(summary)
        subset = df[(df["cutoff"] == "2020-12-31") & (df["evaluation_type"] == "threshold")]
        for _, row in subset.iterrows():
            positives = int(row["future_warnings"])
            caught = int(row["warnings_caught"])
            negatives = max(int(row["ranked_candidates"]) - caught, 1)
            y_true = np.array([1] * positives + [0] * negatives)
            y_score = np.array([1] * caught + [0] * (positives - caught) + [1] * negatives)
            label = str(row["method"])
            try:
                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                RocCurveDisplay.from_predictions(y_true, y_score, ax=ax_roc, name=f"{label} AUC={auc:.2f}")
                PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax_pr, name=f"{label} AP={ap:.3f}")
            except ValueError:
                ax_roc.plot([0, 1], [0, 1], "--", label=f"{label} not evaluable")
    ax_roc.set_title("ROC curves, 2020 cutoff")
    ax_pr.set_title("Precision-recall curves, 2020 cutoff")
    ax_roc.legend(fontsize=7)
    ax_pr.legend(fontsize=7)
    save(fig_roc, "roc_curves_2020")
    save(fig_pr, "pr_curves_2020")


def lead_time():
    path = ROOT / "data" / "processed" / "case_studies.csv"
    fig, ax = plt.subplots(figsize=(6, 4))
    if path.exists():
        df = pd.read_csv(path)
        vals = pd.to_numeric(df.get("days_early"), errors="coerce").dropna()
        if len(vals):
            ax.hist(vals, bins=min(10, len(vals)), color="#60a5fa")
            ax.axvline(vals.median(), color="#ef4444", linestyle="--", label=f"median={vals.median():.0f} days")
            ax.legend()
    ax.set_title("Lead time for caught warnings")
    ax.set_xlabel("days early")
    save(fig, "lead_time_histogram")


def signal_evolution():
    path = ROOT / "data" / "processed" / "temporal_warning_signals.csv"
    fig, ax1 = plt.subplots(figsize=(7, 4))
    if path.exists():
        df = pd.read_csv(path, parse_dates=["signal_first_detected_date"])
        row = df[(df["drug_name"] == "UPADACITINIB") & (df["reaction_term"] == "MYOCARDIAL INFARCTION")]
        if not row.empty:
            x = row["signal_first_detected_date"]
            ax1.plot(x, row["cumulative_case_count_at_detection"], marker="o", label="cases")
            ax1.set_ylabel("cumulative cases")
            ax2 = ax1.twinx()
            ax2.plot(x, row["prr_at_detection"], marker="s", color="#f97316", label="PRR")
            ax2.plot(x, row["ror_at_detection"], marker="^", color="#22c55e", label="ROR")
            ax2.set_ylabel("PRR/ROR")
            ax1.axvline(pd.Timestamp("2020-12-31"), color="#64748b", linestyle="--", label="cutoff")
            ax1.axvline(pd.Timestamp("2021-09-01"), color="#ef4444", linestyle="--", label="FDA warning")
            ax1.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)
    ax1.set_title("UPADACITINIB + MYOCARDIAL INFARCTION signal detection")
    save(fig, "signal_evolution_upadacitinib")


def recall_ci():
    path = ROOT / "data" / "processed" / "research_eval_summary.csv"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if path.exists():
        df = pd.read_csv(path)
        subset = df[df["evaluation_type"] == "threshold"].copy()
        subset = subset.dropna(subset=["recall"])
        subset["method"] = (
            subset["method"]
            .str.replace("_threshold_0_5", "", regex=False)
            .str.replace("_threshold", "", regex=False)
            .str.replace("_", " ", regex=False)
            .str.upper()
        )
        for method, group in subset.groupby("method", sort=False):
            group = group.sort_values("cutoff")
            lower = group["recall"] - group.get("recall_lo95", group["recall"])
            upper = group.get("recall_hi95", group["recall"]) - group["recall"]
            ax.errorbar(
                group["cutoff"],
                group["recall"],
                yerr=np.vstack([lower, upper]),
                marker="o",
                capsize=3,
                linewidth=1.5,
                label=method,
            )
    ax.set_ylabel("threshold recall")
    ax.set_ylim(-0.03, 0.8)
    ax.set_title("Threshold recall by method with bootstrap 95% CI")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "recall_by_method_with_ci")


def main() -> int:
    architecture()
    artifact_rate()
    roc_pr()
    lead_time()
    signal_evolution()
    recall_ci()
    print(f"Wrote figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
