"""
ingestion/parse_faers.py
=========================
Convert FAERS quarterly JSON(.zip) bundles into flat Parquet partitions
using PySpark.  One row per (safetyreportid, drug, reaction) — the shape
required for PRR/ROR disproportionality analysis.

Input:   data/raw/<year>q<n>/drug-event-*.json.zip
Output:  data/processed/raw_adverse_events/
            report_year=YYYY/report_quarter=Q/part-*.parquet

Usage:
    spark-submit --master local[*] ingestion/parse_faers.py
    spark-submit ingestion/parse_faers.py --quarter 2024q1
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

LOG = logging.getLogger("parse_faers")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "raw_adverse_events"

# --------------------------------------------------------------------------- #
# Schema — only the fields we actually use downstream.  Declaring a tight
# schema is dramatically faster than letting Spark infer 10M JSON rows.
# --------------------------------------------------------------------------- #
DRUG_SCHEMA = T.StructType([
    T.StructField("medicinalproduct", T.StringType()),
    T.StructField("drugcharacterization", T.StringType()),
    T.StructField("drugindication", T.StringType()),
])

REACTION_SCHEMA = T.StructType([
    T.StructField("reactionmeddrapt", T.StringType()),
    T.StructField("reactionoutcome", T.StringType()),
])

PATIENT_SCHEMA = T.StructType([
    T.StructField("patientsex", T.StringType()),
    T.StructField("patientonsetage", T.StringType()),
    T.StructField("patientonsetageunit", T.StringType()),
    T.StructField("drug", T.ArrayType(DRUG_SCHEMA)),
    T.StructField("reaction", T.ArrayType(REACTION_SCHEMA)),
])

RECORD_SCHEMA = T.StructType([
    T.StructField("safetyreportid", T.StringType()),
    T.StructField("serious", T.StringType()),
    T.StructField("seriousnessdeath", T.StringType()),
    T.StructField("seriousnesshospitalization", T.StringType()),
    T.StructField("receivedate", T.StringType()),
    T.StructField("receiptdate", T.StringType()),
    T.StructField("occurcountry", T.StringType()),
    T.StructField("primarysourcecountry", T.StringType()),
    T.StructField("patient", PATIENT_SCHEMA),
])

WRAPPER_SCHEMA = T.StructType([
    T.StructField("results", T.ArrayType(RECORD_SCHEMA)),
])


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("fda_faers_parse")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


def _transform_faers_json(spark: SparkSession, json_path: Path, year: int, q: int):
    df = (
        spark.read.option("multiLine", True)
                  .schema(WRAPPER_SCHEMA)
                  .json(str(json_path))
    )

    # explode top-level results array
    df = df.select(F.explode("results").alias("r")).select("r.*")

    # explode patient.drug and patient.reaction independently then cross-join
    # within the same report → one row per drug × reaction pair.
    df = (
        df.select(
            F.col("safetyreportid"),
            F.col("serious"),
            F.col("seriousnessdeath"),
            F.col("seriousnesshospitalization"),
            F.col("receivedate"),
            F.col("occurcountry"),
            F.col("primarysourcecountry"),
            F.col("patient.patientsex").alias("patientsex"),
            F.col("patient.patientonsetage").alias("patient_age"),
            F.explode_outer("patient.drug").alias("d"),
            F.col("patient.reaction").alias("reactions"),
        )
        .filter(F.col("d.medicinalproduct").isNotNull())
        .select(
            "safetyreportid", "serious", "seriousnessdeath",
            "seriousnesshospitalization", "receivedate",
            "occurcountry", "primarysourcecountry",
            "patientsex", "patient_age",
            F.col("d.medicinalproduct").alias("drug_name_raw"),
            F.col("d.drugcharacterization").alias("drug_role"),
            F.explode_outer("reactions").alias("rx"),
        )
        .filter(F.col("rx.reactionmeddrapt").isNotNull())
        .select(
            "safetyreportid", "serious", "seriousnessdeath",
            "seriousnesshospitalization", "receivedate",
            "occurcountry", "primarysourcecountry",
            "patientsex", "patient_age",
            "drug_name_raw", "drug_role",
            F.col("rx.reactionmeddrapt").alias("reaction_term_raw"),
            F.col("rx.reactionoutcome").alias("reaction_outcome"),
        )
    )

    # ------------------------------------------------------------------ #
    # normalisation & typing                                              #
    # ------------------------------------------------------------------ #
    df = (
        df.withColumn("drug_name", F.upper(F.trim(F.col("drug_name_raw"))))
          .withColumn("reaction_term", F.upper(F.trim(F.col("reaction_term_raw"))))
          .withColumn("serious_int", F.when(F.col("serious") == "1", 1).otherwise(0))
          .withColumn("death_int",
                      F.when(F.col("seriousnessdeath") == "1", 1).otherwise(0))
          .withColumn("hosp_int",
                      F.when(F.col("seriousnesshospitalization") == "1", 1).otherwise(0))
          .withColumn("patient_sex_int",
                      F.when(F.col("patientsex") == "1", 1)
                       .when(F.col("patientsex") == "2", 2)
                       .otherwise(0))
          .withColumn("age_years",
                      F.col("patient_age").cast("double"))
          .withColumn("country",
                      F.coalesce("occurcountry", "primarysourcecountry"))
          .withColumn("report_date",
                      F.to_date(F.col("receivedate"), "yyyyMMdd"))
          .withColumn("report_year", F.lit(year))
          .withColumn("report_quarter", F.lit(q))
          .drop("drug_name_raw", "reaction_term_raw",
                "serious", "seriousnessdeath", "seriousnesshospitalization",
                "patientsex", "patient_age",
                "occurcountry", "primarysourcecountry", "receivedate")
    )

    # cheap dedupe — same report, drug, reaction can be repeated
    df = df.dropDuplicates(["safetyreportid", "drug_name", "reaction_term"])

    return df


def _extract_zip_member(zip_path: Path, tmp_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        json_members = [name for name in zf.namelist() if name.endswith(".json")]
        if len(json_members) != 1:
            raise RuntimeError(f"Expected one JSON member in {zip_path}, found {json_members}")
        member = json_members[0]
        target = tmp_dir / Path(member).name
        with zf.open(member) as src, target.open("wb") as dest:
            shutil.copyfileobj(src, dest, length=1024 * 1024)
        return target


def parse_quarter(spark: SparkSession, quarter_dir: Path) -> None:
    quarter = quarter_dir.name              # e.g. "2024q1"
    m = re.match(r"(\d{4})q([1-4])", quarter)
    if not m:
        LOG.warning("Skipping non-quarter directory %s", quarter_dir)
        return
    year, q = int(m.group(1)), int(m.group(2))

    zip_files = sorted(quarter_dir.glob("drug-event-*.json.zip"))
    json_files = sorted(quarter_dir.glob("drug-event-*.json"))
    inputs = zip_files or json_files
    if not inputs:
        LOG.warning("No FAERS JSON inputs found under %s", quarter_dir)
        return

    partition_dir = OUT_DIR / f"report_year={year}" / f"report_quarter={q}"
    if partition_dir.exists():
        LOG.info("Removing existing partition %s", partition_dir)
        shutil.rmtree(partition_dir)

    total = 0
    out_path = str(OUT_DIR)
    LOG.info("Writing parquet -> %s (partition %d Q%d)", out_path, year, q)
    for i, input_path in enumerate(inputs, start=1):
        LOG.info("[%d/%d] Reading %s", i, len(inputs), input_path)
        with tempfile.TemporaryDirectory(prefix=f"faers_{quarter}_") as tmp:
            json_path = (
                _extract_zip_member(input_path, Path(tmp))
                if input_path.suffix == ".zip"
                else input_path
            )
            df = _transform_faers_json(spark, json_path, year, q)
            cnt = df.count()
            if cnt:
                (df.write
                   .mode("append")
                   .partitionBy("report_year", "report_quarter")
                   .parquet(out_path))
            total += cnt
            LOG.info("Wrote %s rows from %s", f"{cnt:,}", input_path.name)

    LOG.info("Wrote %s rows for %s", f"{total:,}", quarter)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--quarter", type=str, default=None,
                        help="Process a single quarter (e.g. 2024q1)")
    args = parser.parse_args()

    spark = build_spark()
    try:
        if args.quarter:
            parse_quarter(spark, args.raw_dir / args.quarter)
        else:
            for qd in sorted(p for p in args.raw_dir.iterdir() if p.is_dir()):
                parse_quarter(spark, qd)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
