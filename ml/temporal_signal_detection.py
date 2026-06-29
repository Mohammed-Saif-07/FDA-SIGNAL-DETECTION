"""
Compute first PRR/ROR threshold crossing dates for FDA warning reference pairs.

This is intentionally narrower than full feature engineering. For research
lead-time validation we only need temporal signal dates for curated FDA warning
pairs, not for every one of millions of FAERS drug/reaction combinations.

Output:
    data/processed/temporal_warning_signals.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "processed" / "clean_adverse_events"
WARNINGS_CSV = ROOT / "data" / "reference" / "fda_warnings.csv"
OUT_CSV = ROOT / "data" / "processed" / "temporal_warning_signals.csv"


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("fda_temporal_warning_signals")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def cumulative_by_quarter(df, partition_cols, value_col, out_col):
    w = (
        Window.partitionBy(*partition_cols)
        .orderBy("quarter_index")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    return df.withColumn(out_col, F.sum(value_col).over(w))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--warnings", type=Path, default=WARNINGS_CSV)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        warnings = (
            spark.read.option("header", True).csv(str(args.warnings))
            .select(
                F.upper(F.trim(F.col("drug_name"))).alias("drug_name"),
                F.upper(F.trim(F.col("reaction_term"))).alias("reaction_term"),
            )
            .dropna()
            .dropDuplicates()
        )
        warning_drugs = warnings.select("drug_name").dropDuplicates()
        warning_reactions = warnings.select("reaction_term").dropDuplicates()

        events = (
            spark.read.parquet(str(args.clean_dir))
            .select("safetyreportid", "drug_name", "reaction_term", "report_year", "report_quarter")
            .withColumn("drug_name", F.upper(F.trim(F.col("drug_name"))))
            .withColumn("reaction_term", F.upper(F.trim(F.col("reaction_term"))))
            .withColumn("quarter_index", F.col("report_year") * F.lit(4) + F.col("report_quarter"))
        )

        pair_q = (
            events.join(F.broadcast(warnings), ["drug_name", "reaction_term"], "inner")
            .groupBy("drug_name", "reaction_term", "report_year", "report_quarter", "quarter_index")
            .agg(F.countDistinct("safetyreportid").alias("pair_q_count"))
        )
        drug_q = (
            events.join(F.broadcast(warning_drugs), "drug_name", "inner")
            .groupBy("drug_name", "report_year", "report_quarter", "quarter_index")
            .agg(F.countDistinct("safetyreportid").alias("drug_q_count"))
        )
        rx_q = (
            events.join(F.broadcast(warning_reactions), "reaction_term", "inner")
            .groupBy("reaction_term", "report_year", "report_quarter", "quarter_index")
            .agg(F.countDistinct("safetyreportid").alias("rx_q_count"))
        )
        grand_q = (
            events.groupBy("report_year", "report_quarter", "quarter_index")
            .agg(F.countDistinct("safetyreportid").alias("grand_q_count"))
        )

        pair_cum = cumulative_by_quarter(pair_q, ["drug_name", "reaction_term"], "pair_q_count", "a")
        drug_cum = cumulative_by_quarter(drug_q, ["drug_name"], "drug_q_count", "drug_total")
        rx_cum = cumulative_by_quarter(rx_q, ["reaction_term"], "rx_q_count", "reaction_total")
        grand_cum = cumulative_by_quarter(grand_q, [], "grand_q_count", "grand_total")

        timeline = (
            pair_cum.join(
                drug_cum.select("drug_name", "report_year", "report_quarter", "quarter_index", "drug_total"),
                ["drug_name", "report_year", "report_quarter", "quarter_index"],
            )
            .join(
                rx_cum.select("reaction_term", "report_year", "report_quarter", "quarter_index", "reaction_total"),
                ["reaction_term", "report_year", "report_quarter", "quarter_index"],
            )
            .join(
                grand_cum.select("report_year", "report_quarter", "quarter_index", "grand_total"),
                ["report_year", "report_quarter", "quarter_index"],
            )
            .withColumn("b", F.col("drug_total") - F.col("a"))
            .withColumn("c", F.col("reaction_total") - F.col("a"))
            .withColumn("d", F.col("grand_total") - F.col("drug_total") - F.col("reaction_total") + F.col("a"))
            .withColumn(
                "prr",
                F.when((F.col("a") + F.col("b")) * F.col("c") == 0, None)
                .otherwise(
                    (F.col("a") / (F.col("a") + F.col("b")))
                    / (F.col("c") / (F.col("c") + F.col("d")))
                ),
            )
            .withColumn(
                "ror",
                F.when(F.col("b") * F.col("c") == 0, None)
                .otherwise((F.col("a") * F.col("d")) / (F.col("b") * F.col("c"))),
            )
            .withColumn(
                "signal_first_detected_date",
                F.last_day(
                    F.to_date(
                        F.concat_ws(
                            "-",
                            F.col("report_year").cast("string"),
                            F.lpad((F.col("report_quarter") * F.lit(3)).cast("string"), 2, "0"),
                            F.lit("01"),
                        )
                    )
                ),
            )
            .filter((F.col("a") >= 3) & (F.col("prr") > 2.0) & (F.col("ror") > 2.0))
        )

        first = (
            timeline.withColumn(
                "rn",
                F.row_number().over(
                    Window.partitionBy("drug_name", "reaction_term").orderBy("quarter_index")
                ),
            )
            .filter(F.col("rn") == 1)
            .select(
                "drug_name",
                "reaction_term",
                "signal_first_detected_date",
                F.col("a").alias("cumulative_case_count_at_detection"),
                F.round("prr", 4).alias("prr_at_detection"),
                F.round("ror", 4).alias("ror_at_detection"),
                "report_year",
                "report_quarter",
            )
            .orderBy("drug_name", "reaction_term")
        )

        pdf = first.toPandas()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pdf.to_csv(args.out, index=False)
        print(f"Wrote {args.out} ({len(pdf):,} rows)")
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
