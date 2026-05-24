"""
spark/data_cleaning.py
======================
Large-scale cleaning of FAERS data with PySpark.

Reads:     data/processed/raw_adverse_events/   (Parquet, partitioned)
Writes:    data/processed/clean_adverse_events/  (Parquet, partitioned)

What it does
------------
1. Standardise drug names  (ASPIRIN = aspirin = Aspirin = ASA)
   - upper-case, strip whitespace
   - strip dose/route suffixes ("ASPIRIN 81 MG" -> "ASPIRIN")
   - apply a brand→generic synonym map (extendable)
2. Normalise reaction terms (upper-case, trim)
3. De-duplicate by safetyreportid + drug + reaction
4. Impute missing demographics
5. Repartition for downstream Hive consumption

Usage:
    spark-submit --master local[*] spark/data_cleaning.py
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

LOG = logging.getLogger("data_cleaning")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
IN_DIR  = ROOT / "data" / "processed" / "raw_adverse_events"
OUT_DIR = ROOT / "data" / "processed" / "clean_adverse_events"

# A small but useful brand→generic mapping.  Extend as needed.
DRUG_SYNONYMS = {
    "ASA": "ASPIRIN",
    "ACETYLSALICYLIC ACID": "ASPIRIN",
    "TYLENOL": "ACETAMINOPHEN",
    "PARACETAMOL": "ACETAMINOPHEN",
    "ADVIL": "IBUPROFEN",
    "MOTRIN": "IBUPROFEN",
    "ALEVE": "NAPROXEN",
    "VIOXX": "ROFECOXIB",
    "BEXTRA": "VALDECOXIB",
    "LIPITOR": "ATORVASTATIN",
    "ZOCOR": "SIMVASTATIN",
    "PLAVIX": "CLOPIDOGREL",
    "WARFARIN SODIUM": "WARFARIN",
    "COUMADIN": "WARFARIN",
    "XARELTO": "RIVAROXABAN",
    "ELIQUIS": "APIXABAN",
    "PRADAXA": "DABIGATRAN",
    "XELJANZ": "TOFACITINIB",
    "RINVOQ": "UPADACITINIB",
    "OLUMIANT": "BARICITINIB",
    "OZEMPIC": "SEMAGLUTIDE",
    "WEGOVY": "SEMAGLUTIDE",
    "MOUNJARO": "TIRZEPATIDE",
    "TRULICITY": "DULAGLUTIDE",
    "PROZAC": "FLUOXETINE",
    "ZOLOFT": "SERTRALINE",
    "PAXIL": "PAROXETINE",
}

DOSE_RE = re.compile(
    r"\s*\d+(\.\d+)?\s*(MG|MCG|MG/ML|MG/KG|ML|G|IU|UNIT|%|MG/L)\b.*$",
    flags=re.IGNORECASE,
)
ROUTE_RE = re.compile(
    r"\s+(TABLET|CAPSULE|INJECTION|ORAL|IV|TOPICAL|CREAM|GEL|SYRUP|PATCH|PILL)\b.*$",
    flags=re.IGNORECASE,
)


def _normalise_drug(s: str | None) -> str | None:
    if s is None:
        return None
    name = s.strip().upper()
    name = DOSE_RE.sub("", name)
    name = ROUTE_RE.sub("", name)
    name = re.sub(r"[^A-Z0-9 \-/]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return DRUG_SYNONYMS.get(name, name) or None


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("fda_data_cleaning")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir",  type=Path, default=IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    spark = build_spark()
    try:
        normalise_udf = F.udf(_normalise_drug, T.StringType())
        # Broadcast the synonym map so the UDF doesn't ship it per task.
        broadcast_syn = spark.sparkContext.broadcast(DRUG_SYNONYMS)

        LOG.info("Reading %s", args.in_dir)
        df = spark.read.parquet(str(args.in_dir))
        before = df.count()
        LOG.info("Loaded %s rows", f"{before:,}")

        # Normalise drug & reaction
        df = (
            df.withColumn("drug_name", normalise_udf(F.col("drug_name")))
              .withColumn("reaction_term", F.upper(F.trim(F.col("reaction_term"))))
              .filter(F.col("drug_name").isNotNull() & (F.length("drug_name") > 1))
              .filter(F.col("reaction_term").isNotNull())
        )

        # Impute country / sex
        df = (
            df.withColumn("country", F.coalesce("country", F.lit("UNK")))
              .withColumn("patient_sex_int", F.coalesce("patient_sex_int", F.lit(0)))
        )

        # Age sanity: drop biologically impossible values
        df = df.withColumn(
            "age_years",
            F.when(F.col("age_years").between(0, 120), F.col("age_years"))
        )

        # De-dup at fact-row level
        df = df.dropDuplicates(["safetyreportid", "drug_name", "reaction_term"])

        after = df.count()
        LOG.info("After cleaning: %s rows  (removed %s)",
                 f"{after:,}", f"{before - after:,}")

        (df.write
           .mode("overwrite")
           .partitionBy("report_year", "report_quarter")
           .parquet(str(args.out_dir)))
        LOG.info("Wrote clean dataset -> %s", args.out_dir)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
