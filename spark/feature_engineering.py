"""
spark/feature_engineering.py
============================
Build the ML feature matrix for the XGBoost "signal → recall" model
using PySpark — so we can join 10M+ adverse events without melting a
laptop.

Reads:
    data/processed/clean_adverse_events/        (Parquet, partitioned)
    data/reference/fda_warnings.csv             (ground truth labels)

Writes:
    data/processed/ml_features/                 (Parquet)

Features (one row per drug+reaction pair):
    drug_name, reaction_term
    case_count               total reports
    case_count_growth_qoq    quarter-over-quarter growth rate
    serious_ratio            serious / total
    death_ratio              deaths / total
    hosp_ratio               hospitalisations / total
    countries_count          how many countries reporting
    age_mean / age_std       demographic
    sex_female_ratio
    days_since_first_report
    prr, ror, prr_chi_square (computed inline so this script is self-contained)
    n_concurrent_signals     other reactions reaching SIGNAL for same drug
    label_became_warning     0/1   (joined from fda_warnings.csv)

Usage:
    spark-submit spark/feature_engineering.py
    spark-submit spark/feature_engineering.py --train-cutoff 2020-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window, functions as F

LOG = logging.getLogger("feature_engineering")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "processed" / "clean_adverse_events"
WARNINGS_CSV = ROOT / "data" / "reference" / "fda_warnings.csv"
OUT_DIR = ROOT / "data" / "processed" / "ml_features"


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("fda_feature_engineering")
        .config("spark.sql.shuffle.partitions", "256")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir",  type=Path, default=CLEAN_DIR)
    p.add_argument("--warnings",   type=Path, default=WARNINGS_CSV)
    p.add_argument("--out-dir",    type=Path, default=OUT_DIR)
    p.add_argument("--train-cutoff", type=str, default=None,
                   help="If set, only use reports with report_date <= cutoff "
                        "when computing features (for backtesting).")
    args = p.parse_args()

    spark = build_spark()
    try:
        events = spark.read.parquet(str(args.clean_dir))

        if args.train_cutoff:
            events = events.filter(F.col("report_date") <= F.lit(args.train_cutoff))
            LOG.info("Using reports up to %s", args.train_cutoff)

        events = events.persist(StorageLevel.DISK_ONLY)
        total_rows = events.count()
        LOG.info("Loaded %s rows", f"{total_rows:,}")

        # ------------------------------------------------------------- #
        # Aggregate pair-level facts                                     #
        # ------------------------------------------------------------- #
        pair = (
            events.groupBy("drug_name", "reaction_term")
            .agg(
                F.countDistinct("safetyreportid").alias("case_count"),
                F.sum(F.col("serious_int").cast("int")).alias("serious_cases"),
                F.sum(F.col("death_int").cast("int")).alias("death_cases"),
                F.sum(F.col("hosp_int").cast("int")).alias("hosp_cases"),
                F.countDistinct("country").alias("countries_count"),
                F.avg("age_years").alias("age_mean"),
                F.stddev("age_years").alias("age_std"),
                F.avg(F.when(F.col("patient_sex_int") == 2, 1.0).otherwise(0.0))
                  .alias("sex_female_ratio"),
                F.min("report_date").alias("first_seen"),
                F.max("report_date").alias("last_seen"),
            )
        )

        pair = (
            pair.withColumn("serious_ratio", F.col("serious_cases") / F.col("case_count"))
                .withColumn("death_ratio",   F.col("death_cases")   / F.col("case_count"))
                .withColumn("hosp_ratio",    F.col("hosp_cases")    / F.col("case_count"))
                .withColumn("days_since_first_report",
                            F.datediff(F.current_date(), F.col("first_seen")))
        )

        # ------------------------------------------------------------- #
        # PRR / ROR / chi-square (same logic as signal_detection.hql)    #
        # ------------------------------------------------------------- #
        drug_totals = pair.groupBy("drug_name").agg(F.sum("case_count").alias("drug_total"))
        rx_totals   = pair.groupBy("reaction_term").agg(F.sum("case_count").alias("rx_total"))
        grand_total = pair.agg(F.sum("case_count").alias("grand_total")).first()["grand_total"]
        LOG.info("Grand total pair-rows = %s", f"{grand_total:,}")

        feats = (
            pair.join(drug_totals, "drug_name")
                .join(rx_totals, "reaction_term")
                .withColumn("a", F.col("case_count"))
                .withColumn("b", F.col("drug_total") - F.col("case_count"))
                .withColumn("c", F.col("rx_total")   - F.col("case_count"))
                .withColumn("d", F.lit(grand_total) - F.col("drug_total")
                                  - F.col("rx_total") + F.col("case_count"))
        )

        feats = (
            feats.withColumn("prr",
                F.when((F.col("a") + F.col("b")) * F.col("c") == 0, None)
                 .otherwise(
                    (F.col("a") / (F.col("a") + F.col("b"))) /
                    (F.col("c") / (F.col("c") + F.col("d")))
                 ))
            .withColumn("ror",
                F.when(F.col("b") * F.col("c") == 0, None)
                 .otherwise((F.col("a") * F.col("d")) / (F.col("b") * F.col("c"))))
            .withColumn("prr_chi_square",
                F.when((F.col("a") + F.col("b")) == 0, None)
                 .otherwise(
                    F.pow(F.abs(F.col("a") * F.col("d") -
                                F.col("b") * F.col("c")) - F.lit(grand_total) / 2.0, 2)
                    * F.lit(grand_total)
                    /
                    ((F.col("a") + F.col("b"))
                     * (F.col("c") + F.col("d"))
                     * (F.col("a") + F.col("c"))
                     * (F.col("b") + F.col("d")))
                 ))
        )

        # ------------------------------------------------------------- #
        # Quarter-over-quarter growth rate (computed per pair)           #
        # ------------------------------------------------------------- #
        quarterly = (
            events.groupBy("drug_name", "reaction_term", "report_year", "report_quarter")
                  .agg(F.countDistinct("safetyreportid").alias("q_count"))
        )
        w = (Window.partitionBy("drug_name", "reaction_term")
                   .orderBy("report_year", "report_quarter"))
        quarterly = quarterly.withColumn("prev_q", F.lag("q_count").over(w))
        quarterly = quarterly.withColumn(
            "qoq_growth",
            F.when(F.col("prev_q").isNotNull() & (F.col("prev_q") > 0),
                   (F.col("q_count") - F.col("prev_q")) / F.col("prev_q"))
        )
        growth = quarterly.groupBy("drug_name", "reaction_term").agg(
            F.avg("qoq_growth").alias("case_count_growth_qoq")
        )
        feats = feats.join(growth, ["drug_name", "reaction_term"], how="left")

        # ------------------------------------------------------------- #
        # # concurrent signals = other reactions on same drug w/ PRR>2   #
        # ------------------------------------------------------------- #
        signal_flag = feats.withColumn(
            "is_signal",
            (F.col("prr") > 2.0) & (F.col("a") >= 3)
        )
        concurrent = (
            signal_flag.filter("is_signal")
                       .groupBy("drug_name")
                       .agg(F.count("*").alias("n_concurrent_signals"))
        )
        feats = feats.join(concurrent, "drug_name", how="left")
        feats = feats.withColumn("n_concurrent_signals",
                                 F.coalesce("n_concurrent_signals", F.lit(0)))

        # ------------------------------------------------------------- #
        # Join ground-truth labels                                       #
        # ------------------------------------------------------------- #
        if args.warnings.exists():
            warnings = (
                spark.read.option("header", True).csv(str(args.warnings))
                     .withColumn("drug_name",     F.upper(F.trim(F.col("drug_name"))))
                     .withColumn("reaction_term", F.upper(F.trim(F.col("reaction_term"))))
                     .withColumn("warning_date",  F.to_date("warning_date"))
            )
            feats = feats.join(warnings.select("drug_name", "reaction_term",
                                               "warning_date"),
                               ["drug_name", "reaction_term"], how="left")
            feats = feats.withColumn(
                "label_became_warning",
                F.when(F.col("warning_date").isNotNull(), F.lit(1)).otherwise(F.lit(0))
            )
        else:
            LOG.warning("fda_warnings.csv not found — label will be 0 for everyone")
            feats = feats.withColumn("warning_date", F.lit(None).cast("date"))
            feats = feats.withColumn("label_became_warning", F.lit(0))

        # ------------------------------------------------------------- #
        # Select final feature columns                                   #
        # ------------------------------------------------------------- #
        out = feats.select(
            "drug_name", "reaction_term",
            "case_count", "case_count_growth_qoq",
            "serious_ratio", "death_ratio", "hosp_ratio",
            "countries_count",
            "age_mean", "age_std",
            "sex_female_ratio",
            "days_since_first_report",
            F.round("prr", 4).alias("prr"),
            F.round("ror", 4).alias("ror"),
            F.round("prr_chi_square", 4).alias("prr_chi_square"),
            "n_concurrent_signals",
            "warning_date",
            "label_became_warning",
        )

        (out.write.mode("overwrite").parquet(str(args.out_dir)))
        n_total = out.count()
        n_pos = out.filter("label_became_warning = 1").count()
        LOG.info("Wrote %s feature rows  (positives=%s, pos_rate=%.4f)",
                 f"{n_total:,}", f"{n_pos:,}", n_pos / max(n_total, 1))

    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
