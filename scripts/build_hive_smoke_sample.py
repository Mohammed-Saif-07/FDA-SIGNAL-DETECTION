"""Build a tiny Hive-compatible Parquet partition for Docker smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "clean_adverse_events" / "report_year=2020" / "report_quarter=1"
OUT = ROOT / "data" / "processed" / "hive_smoke_sample" / "report_year=2020" / "report_quarter=1"

COLS = [
    "safetyreportid",
    "drug_name",
    "drug_role",
    "reaction_term",
    "reaction_outcome",
    "serious_int",
    "death_int",
    "hosp_int",
    "patient_sex_int",
    "age_years",
    "country",
    "report_date",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC).head(5000)
    df = df[[col for col in COLS if col in df.columns]].copy()
    df.to_parquet(OUT / "part-00000.parquet", index=False)
    print(f"wrote {len(df):,} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
