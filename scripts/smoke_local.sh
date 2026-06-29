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
$CONDA_RUN python ml/evaluate.py --cutoff "${BACKTEST_CUTOFF:-2020-12-31}"

echo "[local-smoke] PASS"
