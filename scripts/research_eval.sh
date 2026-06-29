#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_RUN="${CONDA_RUN:-conda run -n fda}"
SPARK_HOME="${SPARK_HOME:-/opt/anaconda3/envs/fda/lib/python3.11/site-packages/pyspark}"
SPARK_SUBMIT="${SPARK_SUBMIT:-$CONDA_RUN env SPARK_HOME=$SPARK_HOME spark-submit --driver-memory 4g --conf spark.driver.maxResultSize=1g}"

echo "[research-eval] Computing warning-pair first-detected signal dates"
$SPARK_SUBMIT ml/temporal_signal_detection.py

echo "[research-eval] Running baseline/case-study evaluation"
$CONDA_RUN python ml/research_evaluate.py

echo "[research-eval] Outputs"
ls -lh \
  data/processed/temporal_warning_signals.csv \
  data/processed/research_eval_summary.csv \
  data/processed/research_eval_summary.json \
  data/processed/case_studies.csv \
  data/processed/missed_warnings.csv \
  data/processed/false_positive_analysis.csv

echo "[research-eval] PASS"
