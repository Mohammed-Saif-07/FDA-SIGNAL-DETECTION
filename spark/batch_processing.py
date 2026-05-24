"""
spark/batch_processing.py
=========================
Incremental processing for the QUARTERLY pipeline.

Called by the Airflow DAG when FDA releases a new quarter of FAERS data.
It only re-parses + cleans the new quarter rather than re-running every
script from scratch.

Steps
-----
1. Locate the new quarter under data/raw/<YYYY>q<n>/
2. PySpark-parse it (delegates to ingestion.parse_faers logic)
3. Clean+normalise the new partition only
4. Append into data/processed/clean_adverse_events/
5. Print the count so the DAG can log it

Usage:
    spark-submit spark/batch_processing.py --quarter 2025q1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from ingestion.parse_faers import build_spark as build_parse_spark, parse_quarter   # noqa: E402
from spark.data_cleaning import _normalise_drug, DRUG_SYNONYMS                       # noqa: E402

LOG = logging.getLogger("batch_processing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "processed" / "clean_adverse_events"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quarter", required=True, help="e.g. 2025q1")
    args = p.parse_args()

    spark = build_parse_spark()
    try:
        LOG.info("=== Parse quarter %s ===", args.quarter)
        parse_quarter(spark, RAW / args.quarter)

        LOG.info("=== Clean quarter %s ===", args.quarter)
        proc_dir = ROOT / "data" / "processed" / "raw_adverse_events"
        year, q = int(args.quarter[:4]), int(args.quarter[-1])
        partition = proc_dir / f"report_year={year}" / f"report_quarter={q}"

        df = spark.read.parquet(str(partition))
        norm_udf = F.udf(_normalise_drug)
        df = (df.withColumn("drug_name", norm_udf("drug_name"))
                .withColumn("reaction_term", F.upper(F.trim("reaction_term")))
                .filter(F.col("drug_name").isNotNull()))
        df = df.withColumn("report_year", F.lit(year)) \
               .withColumn("report_quarter", F.lit(q))

        # Append into the clean dataset
        (df.write
           .mode("append")
           .partitionBy("report_year", "report_quarter")
           .parquet(str(CLEAN)))

        LOG.info("Appended %s rows to clean dataset", f"{df.count():,}")
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
