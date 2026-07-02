"""Generate LaTeX and Markdown tables for the paper draft."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, name: str):
    df.to_markdown(OUT / f"{name}.md", index=False)
    (OUT / f"{name}.tex").write_text(df.to_latex(index=False, escape=True, caption=None))


def _fmt_float(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_sci(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1e}"


def methods_table():
    method_labels = {
        "prr_ror_threshold": "PRR/ROR",
        "robust_prr_ror_threshold": "Robust PRR/ROR",
        "bcpnn_ic025_threshold": "BCPNN IC025",
        "ebgm_eb05_threshold": "EBGM/EB05",
        "xgboost_threshold_0_5": "XGBoost threshold 0.5",
    }
    path = ROOT / "data" / "processed" / "research_eval_summary.csv"
    if not path.exists():
        df = pd.DataFrame(columns=["cutoff", "method", "warnings", "recall", "recall_95ci", "precision", "lead_days"])
    else:
        df = pd.read_csv(path)
        df = df[df["evaluation_type"].eq("threshold")].copy()
        df["warnings"] = df["warnings_caught"].astype(int).astype(str) + "/" + df["future_warnings"].astype(int).astype(str)
        df["warnings_total"] = df["future_warnings"].astype(int)
        df["recall"] = df["recall"].map(lambda x: _fmt_float(x, 3))
        df["recall_95ci"] = df.apply(lambda r: f"[{_fmt_float(r['recall_lo95'], 2)}, {_fmt_float(r['recall_hi95'], 2)}]", axis=1)
        df["precision"] = df["precision"].map(_fmt_sci)
        df["lead_days"] = df["median_days_early"].map(lambda x: "" if pd.isna(x) else str(int(x)))
        df["method"] = df["method"].map(method_labels).fillna(df["method"])
        rows = []
        for cutoff, group in df.groupby("cutoff", sort=True):
            warnings_total = int(group["warnings_total"].iloc[0])
            rows.append(
                {
                    "cutoff": cutoff,
                    "method": "future warnings",
                    "warnings": str(warnings_total),
                    "recall": "",
                    "recall_95ci": "",
                    "precision": "",
                    "lead_days": "",
                }
            )
            rows.extend(group[["cutoff", "method", "warnings", "recall", "recall_95ci", "precision", "lead_days"]].to_dict("records"))
        df = pd.DataFrame(rows, columns=["cutoff", "method", "warnings", "recall", "recall_95ci", "precision", "lead_days"])
        df = df.rename(
            columns={
                "recall_95ci": "recall 95% CI",
                "lead_days": "lead days",
            }
        )
    write_table(df, "table_methods")


def caught_warnings():
    path = ROOT / "data" / "processed" / "case_studies.csv"
    if path.exists():
        df = pd.read_csv(path)
        keep = ["drug_name", "reaction_term", "warning_date", "signal_first_detected_date", "days_early", "months_early", "lead_time_basis"]
        df = df[[c for c in keep if c in df.columns]]
        if "months_early" in df.columns:
            df["months_early"] = df["months_early"].map(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")
        if "lead_time_basis" in df.columns:
            df["lead_time_basis"] = df["lead_time_basis"].map(
                lambda x: "" if pd.isna(x) else str(x).replace("_", " ")
            )
        df = df.rename(
            columns={
                "drug_name": "drug",
                "reaction_term": "reaction",
                "warning_date": "warning date",
                "signal_first_detected_date": "signal first detected",
                "days_early": "days early",
                "months_early": "months early",
                "lead_time_basis": "lead time basis",
            }
        )
    else:
        df = pd.DataFrame(columns=["drug_name", "reaction_term", "warning_date", "signal_first_detected_date", "days_early"])
    write_table(df, "table_caught_warnings")


def dataset_table():
    rows = []
    features = ROOT / "data" / "processed" / "ml_features"
    warnings = ROOT / "data" / "reference" / "fda_warnings.csv"
    if features.exists():
        df = pd.read_parquet(features, columns=["drug_name", "reaction_term", "case_count"])
        rows.append({"metric": "feature rows", "value": len(df)})
        rows.append({"metric": "distinct drugs", "value": df["drug_name"].nunique()})
        rows.append({"metric": "distinct reactions", "value": df["reaction_term"].nunique()})
        rows.append({"metric": "drug reaction rows scanned", "value": int(df["case_count"].sum())})
    if warnings.exists():
        w = pd.read_csv(warnings)
        rows.append({"metric": "reference warnings", "value": len(w)})
    write_table(pd.DataFrame(rows), "table_dataset")


def main() -> int:
    methods_table()
    caught_warnings()
    dataset_table()
    print(f"Wrote tables to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
