#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_RUN="${CONDA_RUN:-conda run -n fda}"

echo "[local-smoke] Python/dependency check"
$CONDA_RUN python - <<'PY'
import pandas, pyarrow, pyspark, xgboost, sklearn
print("deps ok")
PY

echo "[local-smoke] Backtest check"
REPORT="data/processed/backtest_report.json"
TMP_REPORT="$(mktemp)"
HAD_REPORT=0
if [[ -f "$REPORT" ]]; then
  cp "$REPORT" "$TMP_REPORT"
  HAD_REPORT=1
fi
$CONDA_RUN python ml/evaluate.py --cutoff "${BACKTEST_CUTOFF:-2020-12-31}"
if [[ "$HAD_REPORT" -eq 1 ]]; then
  cp "$TMP_REPORT" "$REPORT"
else
  rm -f "$REPORT"
fi
rm -f "$TMP_REPORT"

echo "[local-smoke] PASS"
