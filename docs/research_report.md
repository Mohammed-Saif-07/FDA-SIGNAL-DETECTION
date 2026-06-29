# A Reproducible Local Pipeline for FAERS Drug Safety Signal Detection

## Abstract

This project implements a zero-cost, locally reproducible pharmacovigilance
pipeline for detecting drug safety signals in FDA FAERS adverse-event data. The
pipeline parses nested FAERS JSON, creates flattened drug/reaction Parquet
tables, computes PRR/ROR disproportionality signals, trains an XGBoost ranking
model, applies a transparent robust signal-quality filter, and backtests
detected signals against a curated FDA warning reference set. On a
`2020-12-31` cutoff, the threshold-based PRR/ROR and XGBoost
probability workflows each detected 1 of 7 post-cutoff FDA warnings. The
PRR/ROR-detected warning first crossed the signal threshold at the `2020-03-31`
quarter end, 519 days, or 17.3 months, before official FDA action. Results show that the system
is reproducible and useful for signal exploration, while also demonstrating the
low-recall and high-false-positive limitations of public FAERS
disproportionality analysis.

## Research Question

Can a fully local, zero-cost FAERS pipeline reproduce standard pharmacovigilance
signal detection methods and identify any FDA safety warnings before official
regulatory action?

## Background

FAERS is a spontaneous reporting system. Reports can reveal emerging safety
concerns, but they are noisy: duplicate reports, reporting bias, missing exposure
denominators, media-driven reporting, and confounding are common. PRR and ROR
are disproportionality metrics used to rank drug/reaction pairs that appear more
often than expected. These metrics identify statistical signals, not causality.

## Methods

The project implements:

- PySpark ingestion and cleaning of public FAERS adverse-event files.
- Feature engineering over flattened drug/reaction pairs.
- PRR, ROR, and chi-square signal metrics.
- Overflow-safe chi-square computation in Spark and Hive.
- Robust PRR/ROR filtering using case count, seriousness, ratio bounds, and
  country-count source-diversity proxy.
- Hive/HQL implementations mirroring the PRR/ROR calculations.
- XGBoost ranking over engineered signal features.
- PostgreSQL, FastAPI, and Streamlit for serving and visualizing outputs.

Signal formulas:

```text
PRR = (drug_reaction_count / drug_total) / (reaction_total / grand_total)

ROR = (drug_reaction_count * (grand_total - reaction_total))
      / ((drug_total - drug_reaction_count) * reaction_total)
```

Primary threshold:

```text
PRR > 2.0 AND ROR > 2.0 AND case_count >= 3
```

## Architecture

```mermaid
flowchart LR
    A["openFDA FAERS JSON"] --> B["PySpark Parser"]
    B --> C["Cleaned Parquet"]
    C --> D["HDFS"]
    D --> E["Hive External Tables"]
    E --> F["HiveQL PRR/ROR Signal Detection"]
    C --> G["PySpark Feature Engineering"]
    G --> H["XGBoost Classifier"]
    F --> I["PostgreSQL Results"]
    H --> I
    I --> J["FastAPI"]
    I --> K["Streamlit Dashboard"]
    L["FDA Warning Reference Set"] --> M["Backtest Evaluation"]
    F --> M
    H --> M
```

## Data

Current processed evaluation data:

- Feature rows: 4,286,074 drug/reaction pairs.
- Warning reference rows: 56 curated FDA warning rows.
- 2020-cutoff future warning rows: 7.
- Raw PRR/ROR threshold candidates: 505,220.
- Robust-pass PRR/ROR candidates: 100,090.
- Dashboard snapshot rows: 2,000 signals and 2,000 predictions for Streamlit
  Cloud.

The public Streamlit Cloud deployment uses static exported CSV snapshots because
Streamlit Cloud cannot run the local Docker network, HDFS, HiveServer2,
PostgreSQL, or Airflow services. The full stack runs locally on macOS with
Docker Compose.

## Evaluation Design

The research evaluation compares:

- `case_count`
- `prr`
- `ror`
- `prr_ror`
- `prr_ror_chi_square`
- `robust_prr_ror`
- `xgboost`
- `prr_ror_threshold`
- `robust_prr_ror_threshold`
- `xgboost_threshold_0_5`

Cutoffs:

- `2018-12-31`
- `2019-12-31`
- `2020-12-31`
- `2021-12-31`

Artifacts:

- `data/processed/research_eval_summary.csv`
- `data/processed/research_eval_summary.json`
- `data/processed/temporal_warning_signals.csv`
- `data/processed/case_studies.csv`
- `data/processed/missed_warnings.csv`
- `data/processed/false_positive_analysis.csv`

## Results

For the main `2020-12-31` cutoff:

| Method | Evaluation | Future Warnings | Caught | Recall | Precision | Median Lead Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PRR/ROR threshold | threshold | 7 | 1 | 14.3% | ~0.0002% | 519 days |
| Robust PRR/ROR threshold | threshold | 7 | 1 | 14.3% | ~0.0010% | 519 days |
| XGBoost >= 0.5 | threshold | 7 | 1 | 14.3% | ~0.0009% | 519 days |
| XGBoost top 100 | ranking | 7 | 0 | 0.0% | 0.0% | n/a |
| PRR/ROR top 100 | ranking | 7 | 0 | 0.0% | 0.0% | n/a |

The headline result is therefore:

> Using a `2020-12-31` cutoff, the pipeline detected 1 of 7 post-cutoff FDA
> warnings. The PRR/ROR signal for the detected warning first crossed threshold
> at quarter end `2020-03-31`, 519 days, or 17.3 months, before official FDA
> action.

This is a modest result. It supports the reproducibility and engineering value
of the system, but not a claim of high predictive accuracy.

The robust filter reduced raw PRR/ROR candidates from 505,220 to 100,090
without losing the detected post-cutoff warning. It also removed the earlier
single-country/low-diversity consumer-product artifacts from the dashboard
export. Because the public flattened FAERS table does not include a clean
reporter-source identifier, `countries_count` is explicitly labelled as a
source-diversity proxy rather than true independent source count.

## Case Study

The detected warning in the 2020-cutoff evaluation was:

| Drug | Reaction | First Signal Date | FDA Warning Date | Days Early | PRR | ROR | Case Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UPADACITINIB | MYOCARDIAL INFARCTION | 2020-03-31 | 2021-09-01 | 519 | 3.5179 | 3.5313 | 24 |

Missed warnings are written to `data/processed/missed_warnings.csv` with a
reason label. The most common reasons are exact terminology mismatch and signals
not ranking or thresholding high enough.

## Limitations

- FAERS signals are not causal proof.
- The current warning reference set is small; only 7 rows are after the
  2020-cutoff.
- Current first-detected dates are computed for FDA warning reference pairs;
  extending this to every candidate signal would support richer precision and
  false-positive timing analysis.
- Top-k ranking performance is weak in the current run.
- False-positive volume is high: hundreds of thousands of PRR/ROR threshold
  candidates do not match the curated FDA warning reference set.
- Robust filtering reduces obvious artifact-like rankings but remains a
  heuristic; it should be compared against BCPNN/IC and EBGM/MGPS baselines for
  a stronger paper.
- Docker Hive smoke validates schema and HQL execution, but the current Docker
  Hive table is not loaded with the full FAERS Parquet sample.

## Reproducibility

Local smoke:

```bash
make smoke-local
```

Docker/Hive/HDFS/API/Streamlit smoke:

```bash
make smoke-docker
```

Research evaluation:

```bash
make research-eval
```

Primary backtest:

```bash
conda run -n fda python ml/evaluate.py --cutoff 2020-12-31
```

## Conclusion

The project is research-ready as a reproducible applied systems paper or
capstone report, provided the claim stays honest: it demonstrates a local,
zero-cost FAERS signal detection workflow and catches one post-cutoff FDA
warning 17.3 months early, while exposing the practical limitations of PRR/ROR
and ML ranking on noisy public pharmacovigilance data.
