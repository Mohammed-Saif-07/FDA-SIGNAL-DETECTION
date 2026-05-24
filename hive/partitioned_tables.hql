-- =====================================================================
-- partitioned_tables.hql
-- Manage partitions on raw_adverse_events.
--
-- Use this after dropping in a new quarter of Parquet into HDFS.
-- =====================================================================

USE fda_pharma;

-- ---- 1) Auto-discover all partitions present on disk ----------------
MSCK REPAIR TABLE raw_adverse_events;

-- ---- 2) Manual add of a single quarter (idempotent) -----------------
-- Replace the year/quarter literals before running.
ALTER TABLE raw_adverse_events
  ADD IF NOT EXISTS PARTITION (report_year=2024, report_quarter=1)
  LOCATION '/user/hive/warehouse/fda_pharma.db/raw_adverse_events/report_year=2024/report_quarter=1';

-- ---- 3) Drop an old/corrupt partition --------------------------------
-- ALTER TABLE raw_adverse_events
--   DROP IF EXISTS PARTITION (report_year=2004, report_quarter=1);

-- ---- 4) Show all current partitions ---------------------------------
SHOW PARTITIONS raw_adverse_events;

-- ---- 5) Compute table statistics (helps the optimiser) --------------
ANALYZE TABLE raw_adverse_events PARTITION (report_year, report_quarter)
  COMPUTE STATISTICS;
ANALYZE TABLE raw_adverse_events PARTITION (report_year, report_quarter)
  COMPUTE STATISTICS FOR COLUMNS drug_name, reaction_term;
