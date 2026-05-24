-- =====================================================================
-- create_tables.hql
-- Hive schema for FDA Drug Safety Signal Detection
-- Run with:  beeline -u jdbc:hive2://localhost:10000 -f hive/create_tables.hql
-- =====================================================================

CREATE DATABASE IF NOT EXISTS fda_pharma
  COMMENT 'FDA FAERS pharmacovigilance signal detection'
  LOCATION '/user/hive/warehouse/fda_pharma.db';

USE fda_pharma;

-- =====================================================================
-- Table 1: raw_adverse_events
-- One row per (safetyreportid, drug, reaction) — the fact table.
-- External + partitioned + Parquet = scan only the partitions you need.
-- =====================================================================
DROP TABLE IF EXISTS raw_adverse_events;
CREATE EXTERNAL TABLE raw_adverse_events (
    safetyreportid    STRING,
    drug_name         STRING,
    drug_role         STRING,         -- 1=suspect, 2=concomitant, 3=interacting
    reaction_term     STRING,
    reaction_outcome  STRING,
    serious_int       TINYINT,
    death_int         TINYINT,
    hosp_int          TINYINT,
    patient_sex_int   TINYINT,
    age_years         DOUBLE,
    country           STRING,
    report_date       DATE
)
PARTITIONED BY (report_year INT, report_quarter INT)
STORED AS PARQUET
LOCATION '/user/hive/warehouse/fda_pharma.db/raw_adverse_events'
TBLPROPERTIES ('parquet.compression'='SNAPPY');

-- After uploading parquet files into the partitioned location:
MSCK REPAIR TABLE raw_adverse_events;

-- =====================================================================
-- Table 2: drug_signals (PRR + ROR results)
-- Populated by signal_detection.hql
-- =====================================================================
DROP TABLE IF EXISTS drug_signals;
CREATE TABLE drug_signals (
    drug_name              STRING,
    reaction_term          STRING,
    case_count             INT,
    drug_total             INT,
    reaction_total         INT,
    grand_total            BIGINT,
    prr                    DOUBLE,
    ror                    DOUBLE,
    prr_chi_square         DOUBLE,
    signal_status          STRING,
    confidence             STRING,
    first_detected_date    DATE,
    last_updated_ts        TIMESTAMP
)
STORED AS PARQUET;

-- =====================================================================
-- Table 3: signal_predictions (XGBoost output, written from PySpark/pandas)
-- =====================================================================
DROP TABLE IF EXISTS signal_predictions;
CREATE TABLE signal_predictions (
    drug_name                STRING,
    reaction_term            STRING,
    recall_probability       DOUBLE,
    predicted_class          TINYINT,
    predicted_date           DATE,
    actual_fda_warning_date  DATE,
    days_predicted_early     INT,
    model_version            STRING
)
STORED AS PARQUET;

-- =====================================================================
-- Table 4: fda_official_warnings (ground truth, loaded from CSV)
-- =====================================================================
DROP TABLE IF EXISTS fda_official_warnings;
CREATE TABLE fda_official_warnings (
    drug_name        STRING,
    reaction_term    STRING,
    warning_date     DATE,
    warning_type     STRING,   -- BOXED_WARNING / SAFETY_COMMUNICATION / RECALL
    recall_date      DATE,
    source_url       STRING,
    notes            STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
TBLPROPERTIES ("skip.header.line.count"="1");

-- =====================================================================
-- Table 5: signal_trends (per-quarter PRR/ROR series)
-- =====================================================================
DROP TABLE IF EXISTS signal_trends;
CREATE TABLE signal_trends (
    drug_name        STRING,
    reaction_term    STRING,
    report_year      INT,
    report_quarter   INT,
    case_count       INT,
    prr              DOUBLE,
    ror              DOUBLE
)
STORED AS PARQUET;

-- =====================================================================
-- Quick smoke-test: list tables + row count of raw fact table
-- =====================================================================
SHOW TABLES;
SELECT COUNT(*) AS total_rows FROM raw_adverse_events;
