# Reproducibility Checklist

This project is designed for local, zero-cost reproduction on macOS Apple
Silicon using the `fda` conda environment. Docker Desktop is required only for
the optional HDFS/Hive/PostgreSQL/API/Streamlit service smoke test.

## What Is Reproducible From A Bare Clone

- Streamlit Cloud dashboard from CSV snapshots in `dashboard/data/`.
- Local syntax and dependency smoke checks with `make smoke-local`.
- Docker service smoke checks with `make smoke-docker` when Docker Desktop is
  running.
- Paper table and figure generation from committed summary artifacts.

## What Requires Local Data Regeneration

- Full `data/processed/ml_features/` Parquet is not committed because it is a
  generated data artifact.
- Raw FAERS zip/json files are not committed.
- To reproduce the full 20M-row feature matrix, run the documented FAERS
  ingestion and feature engineering commands locally.

## Determinism

- XGBoost training exposes `--seed`, default `42`.
- Research evaluation bootstrap intervals use seed `42`.
- Tests are deterministic and skip large-data checks when generated Parquet is
  unavailable.

## Compute Environment

- macOS Apple Silicon tested through conda `fda` Python 3.11.
- Docker Compose definitions and smoke targets are included for
  HDFS/Hive/PostgreSQL/API/Streamlit validation. The July 1, 2026 verification
  run could not execute Docker because the local Docker daemon was not running.
- No paid cloud services or GPUs are required.

## Data

- FAERS data source: public openFDA adverse-event files.
- Warning reference data: hand-curated CSV in `data/reference/fda_warnings.csv`.
- Rows with generic FDA landing-page URLs should be treated as not fully
  source-verified until replaced with deep FDA Drug Safety Communication links.

## Evaluation Limits

- Current validated 2020 cutoff result catches 2 of 7 post-cutoff warnings
  with BCPNN IC025. The clearest case study is UPADACITINIB + MYOCARDIAL
  INFARCTION.
- Lead time is 519 days, or 17.3 months, measured from the first quarter-end
  where cumulative PRR/ROR crossed threshold.
- This is pharmacovigilance signal detection, not clinical causality or
  clinical decision support.
