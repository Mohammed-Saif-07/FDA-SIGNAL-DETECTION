"""Generate LaTeX and Markdown tables for the paper draft."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, name: str):
    df.to_markdown(OUT / f"{name}.md", index=False)
    (OUT / f"{name}.tex").write_text(df.to_latex(index=False, escape=True))


def methods_table():
    path = ROOT / "data" / "processed" / "research_eval_summary.csv"
    if not path.exists():
        df = pd.DataFrame(columns=["cutoff", "method", "recall", "precision", "median_days_early"])
    else:
        df = pd.read_csv(path)
        df = df[df["evaluation_type"].isin(["top_k", "threshold"])].copy()
        df = df[["cutoff", "method", "evaluation_type", "k", "recall", "recall_lo95", "recall_hi95", "precision", "precision_lo95", "precision_hi95", "median_days_early"]]
    write_table(df.head(80), "table_methods")


def caught_warnings():
    path = ROOT / "data" / "processed" / "case_studies.csv"
    if path.exists():
        df = pd.read_csv(path)
        keep = ["drug_name", "reaction_term", "warning_date", "signal_first_detected_date", "days_early", "months_early", "lead_time_basis"]
        df = df[[c for c in keep if c in df.columns]]
    else:
        df = pd.DataFrame(columns=["drug_name", "reaction_term", "warning_date", "signal_first_detected_date", "days_early"])
    write_table(df, "table_caught_warnings")


def dataset_table():
    rows = []
    features = ROOT / "data" / "processed" / "ml_features"
    warnings = ROOT / "data" / "reference" / "fda_warnings.csv"
    if features.exists():
        df = pd.read_parquet(features, columns=["drug_name", "reaction_term", "case_count"])
        rows.append({"metric": "feature_rows", "value": len(df)})
        rows.append({"metric": "distinct_drugs", "value": df["drug_name"].nunique()})
        rows.append({"metric": "distinct_reactions", "value": df["reaction_term"].nunique()})
        rows.append({"metric": "drug_reaction_rows_scanned", "value": int(df["case_count"].sum())})
    if warnings.exists():
        w = pd.read_csv(warnings)
        rows.append({"metric": "reference_warnings", "value": len(w)})
    write_table(pd.DataFrame(rows), "table_dataset")


def main() -> int:
    methods_table()
    caught_warnings()
    dataset_table()
    print(f"Wrote tables to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

