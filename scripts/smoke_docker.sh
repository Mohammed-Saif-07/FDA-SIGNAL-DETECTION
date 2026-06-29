#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONDA_RUN="${CONDA_RUN:-conda run -n fda}"
SAMPLE_DIR="$ROOT/data/processed/hive_smoke_sample"
HDFS_RAW="/user/hive/warehouse/fda_pharma.db/raw_adverse_events"

echo "[docker-smoke] Starting Docker services"
docker compose up -d postgres api streamlit namenode datanode hive-metastore hive-server

echo "[docker-smoke] Waiting for HiveServer2"
for i in $(seq 1 30); do
  if docker exec hive-server /opt/hive/bin/beeline --verbose=false -u jdbc:hive2://127.0.0.1:10000 -e 'show databases;' >/dev/null 2>&1; then
    echo "[docker-smoke] HiveServer2 ready"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "[docker-smoke] HiveServer2 did not become ready" >&2
    exit 1
  fi
  sleep 5
done

echo "[docker-smoke] Hive schema"
docker exec hive-server /opt/hive/bin/beeline --verbose=false -u jdbc:hive2://127.0.0.1:10000 -f /sql/create_tables.hql

echo "[docker-smoke] Loading a real Parquet sample into HDFS/Hive"
rm -rf "$SAMPLE_DIR"
$CONDA_RUN python scripts/build_hive_smoke_sample.py

docker exec hive-server hdfs dfs -rm -r -f "$HDFS_RAW/report_year=2020/report_quarter=1" >/dev/null 2>&1 || true
docker exec hive-server hdfs dfs -mkdir -p "$HDFS_RAW/report_year=2020/report_quarter=1"
docker cp "$SAMPLE_DIR/report_year=2020/report_quarter=1/part-00000.parquet" hive-server:/tmp/fda-hive-smoke.parquet
docker exec hive-server hdfs dfs -put -f /tmp/fda-hive-smoke.parquet "$HDFS_RAW/report_year=2020/report_quarter=1/part-00000.parquet"
docker exec hive-server /opt/hive/bin/beeline --verbose=false -u jdbc:hive2://127.0.0.1:10000 -e \
  "USE fda_pharma; MSCK REPAIR TABLE raw_adverse_events; SELECT COUNT(*) AS loaded_rows FROM raw_adverse_events;"

echo "[docker-smoke] Hive PRR/ROR HQL"
docker exec hive-server /opt/hive/bin/beeline --verbose=false -u jdbc:hive2://127.0.0.1:10000 -f /sql/signal_detection.hql

echo "[docker-smoke] API/Streamlit/HDFS checks"
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/stats
curl -fsSI http://localhost:8501 >/dev/null
curl -fsSI http://localhost:9870 >/dev/null

echo "[docker-smoke] PASS"
