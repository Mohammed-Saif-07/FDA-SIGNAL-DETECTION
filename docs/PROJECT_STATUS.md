# Project Status

Last verified: 2026-07-01.

## Current Defensible Claim

BCPNN IC025 detected 2 of 7 post-cutoff FDA warning pairs at the 2020-12-31
cutoff (28.6% recall, bootstrap 95% CI [0.00, 0.71]). The clearest case study
is UPADACITINIB + MYOCARDIAL INFARCTION, detected 519 days (17.3 months) before
the reference FDA action date.

Do not claim 73% recall or clinical decision-support performance from the
current artifacts.

## Completed

- Real FAERS-derived PySpark feature table with 4,286,074 drug-reaction rows.
- PRR/ROR/chi-square signal calculations in PySpark and mirrored Hive/HQL.
- BCPNN IC025 and simplified EBGM/EB05 baselines.
- XGBoost ranking model and calibration diagnostic.
- Four-cutoff backtest with bootstrap confidence intervals.
- McNemar threshold comparison and DeLong AUC comparison.
- Negative-control evaluation on 20 curated pairs.
- Streamlit Cloud dashboard using compact CSV snapshots.
- Local Streamlit/PostgreSQL fallback export path.
- Overleaf-compatible paper source bundle.
- GitHub Actions pytest workflow.

## Verified Locally

- `make smoke-local`: pass.
- `make reproduce-paper`: pass.
- `make phase4`: pass.
- `make smoke-docker`: pass. Docker Desktop validated HDFS NameNode/DataNode,
  Hive Metastore/HiveServer2, PostgreSQL, FastAPI, and Streamlit services on a
  5,000-row real Parquet smoke sample; Hive created the expected tables and the
  PRR/ROR HQL returned 33 smoke-sample `STRONG_SIGNAL` rows.
- `make test`: 13 tests passed, with two expected warnings from an existing
  bootstrap empty-slice edge-case test.
- `dashboard/data/signals.csv`: 2,000 exported rows; all pass both
  `passes_robust_filter` and `passes_structural_filter`.
- `dashboard/data/backtests.csv`: BCPNN IC025, recall 2/7, median 519 days early.

## Not Verified In The Latest Run

- Local LaTeX compilation: not run because `latexmk`/`pdflatex` are not
  installed locally. Use Overleaf with the Desktop bundle.

## Remaining Work Before Formal Paper Submission

- Replace generic FDA landing-page URLs in `data/reference/fda_warnings.csv`
  with deep FDA Drug Safety Communication or label-change links. Currently 54 of
  56 rows still use a generic FDA drugs URL.
- Expand the curated warning reference set beyond 56 rows.
- Add stronger terminology normalization or MedDRA grouping for warning labels.
- Compile the paper in Overleaf and inspect table/figure layout manually.
- Spot-check BibTeX entries against PubMed, Crossref, or Google Scholar.
- Create a GitHub release and Zenodo DOI before using the DOI badge.
