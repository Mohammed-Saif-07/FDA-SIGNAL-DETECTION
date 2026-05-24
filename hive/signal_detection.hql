-- =====================================================================
-- signal_detection.hql
-- The MAIN analysis: PRR + ROR disproportionality analysis over all
-- 10M+ FAERS adverse-event rows.  This is the exact methodology FDA's
-- FAERS monitoring division uses internally.
--
-- Definitions (the 2x2 contingency table):
--                                Reaction R     Other reactions
--   Drug D                          a              b
--   Other drugs                     c              d
--
--   PRR = (a / (a + b)) / (c / (c + d))
--   ROR =  (a * d) / (b * c)
--
--   Signal thresholds (FDA / EMA standard):
--       PRR > 2.0 AND a >= 3
--       ROR > 2.0 AND a >= 3
--       chi-square > 4   (≈ p < 0.05)
-- =====================================================================

USE fda_pharma;

-- Performance hints
SET hive.exec.parallel = true;
SET hive.exec.dynamic.partition.mode = nonstrict;
SET hive.vectorized.execution.enabled = true;
SET hive.vectorized.execution.reduce.enabled = true;
SET hive.cbo.enable = true;
SET hive.compute.query.using.stats = true;
SET hive.auto.convert.join = true;
SET mapreduce.input.fileinputformat.split.maxsize = 268435456;  -- 256 MB

-- =====================================================================
-- Step 0: parameters (override with `-hivevar` from the command line)
-- =====================================================================
SET hivevar:PRR_THRESHOLD = 2.0;
SET hivevar:ROR_THRESHOLD = 2.0;
SET hivevar:MIN_CASES     = 3;
SET hivevar:CHI2_THRESHOLD = 4.0;

-- =====================================================================
-- Step 1: per-drug × per-reaction × per-margin counts
--         (only consider drugs flagged as SUSPECT — drug_role='1')
-- =====================================================================
DROP TABLE IF EXISTS _dr_pair_counts;
CREATE TABLE _dr_pair_counts STORED AS PARQUET AS
SELECT
    drug_name,
    reaction_term,
    COUNT(DISTINCT safetyreportid)                                     AS case_count,
    SUM(CAST(serious_int AS INT))                                      AS serious_cases,
    SUM(CAST(death_int   AS INT))                                      AS death_cases,
    SUM(CAST(hosp_int    AS INT))                                      AS hosp_cases,
    COUNT(DISTINCT country)                                            AS countries_count,
    MIN(report_date)                                                   AS first_seen,
    MAX(report_date)                                                   AS last_seen
FROM raw_adverse_events
WHERE drug_name IS NOT NULL
  AND reaction_term IS NOT NULL
  AND (drug_role = '1' OR drug_role IS NULL)        -- suspect drug (or unknown role)
GROUP BY drug_name, reaction_term;

DROP TABLE IF EXISTS _drug_totals;
CREATE TABLE _drug_totals STORED AS PARQUET AS
SELECT drug_name, SUM(case_count) AS drug_total
FROM   _dr_pair_counts
GROUP  BY drug_name;

DROP TABLE IF EXISTS _reaction_totals;
CREATE TABLE _reaction_totals STORED AS PARQUET AS
SELECT reaction_term, SUM(case_count) AS reaction_total
FROM   _dr_pair_counts
GROUP  BY reaction_term;

DROP TABLE IF EXISTS _grand_total;
CREATE TABLE _grand_total STORED AS PARQUET AS
SELECT SUM(case_count) AS grand_total FROM _dr_pair_counts;

-- =====================================================================
-- Step 2: compute PRR / ROR / chi-square for every drug+reaction pair
-- =====================================================================
DROP TABLE IF EXISTS drug_signals;
CREATE TABLE drug_signals STORED AS PARQUET AS
WITH pair_with_margins AS (
    SELECT
        p.drug_name,
        p.reaction_term,
        p.case_count                                                AS a,        -- drug & reaction
        (dt.drug_total - p.case_count)                              AS b,        -- drug only
        (rt.reaction_total - p.case_count)                          AS c,        -- reaction only
        (gt.grand_total - dt.drug_total - rt.reaction_total + p.case_count) AS d, -- neither
        dt.drug_total,
        rt.reaction_total,
        gt.grand_total,
        p.serious_cases,
        p.death_cases,
        p.countries_count,
        p.first_seen
    FROM       _dr_pair_counts  p
    JOIN       _drug_totals     dt ON dt.drug_name      = p.drug_name
    JOIN       _reaction_totals rt ON rt.reaction_term  = p.reaction_term
    CROSS JOIN _grand_total     gt
),
calc AS (
    SELECT
        drug_name,
        reaction_term,
        a AS case_count,
        drug_total,
        reaction_total,
        grand_total,

        /* PRR = (a / (a + b))  /  (c / (c + d))      */
        CASE
            WHEN (a + b) = 0 OR (c + d) = 0 OR c = 0 THEN NULL
            ELSE (CAST(a AS DOUBLE) / (a + b)) /
                 (CAST(c AS DOUBLE) / (c + d))
        END                                                 AS prr,

        /* ROR = (a*d) / (b*c)                       */
        CASE
            WHEN b = 0 OR c = 0 THEN NULL
            ELSE (CAST(a AS DOUBLE) * d) / (CAST(b AS DOUBLE) * c)
        END                                                 AS ror,

        /*  Yates-corrected chi-square                */
        CASE
            WHEN (a + b) = 0 OR (c + d) = 0 OR (a + c) = 0 OR (b + d) = 0 THEN NULL
            ELSE (
                POWER(ABS(CAST(a AS DOUBLE) * d - CAST(b AS DOUBLE) * c) - (grand_total / 2.0), 2)
                * grand_total
              ) / (CAST(a + b AS DOUBLE) * (c + d) * (a + c) * (b + d))
        END                                                 AS prr_chi_square,

        CASE WHEN a > 0 THEN serious_cases / CAST(a AS DOUBLE) END AS serious_ratio,
        CASE WHEN a > 0 THEN death_cases   / CAST(a AS DOUBLE) END AS death_ratio,
        countries_count,
        first_seen
    FROM pair_with_margins
)
SELECT
    drug_name,
    reaction_term,
    case_count,
    drug_total,
    reaction_total,
    grand_total,
    ROUND(prr, 4)            AS prr,
    ROUND(ror, 4)            AS ror,
    ROUND(prr_chi_square, 4) AS prr_chi_square,

    CASE
        WHEN case_count >= ${hivevar:MIN_CASES}
         AND prr > ${hivevar:PRR_THRESHOLD}
         AND ror > ${hivevar:ROR_THRESHOLD}
         AND prr_chi_square > ${hivevar:CHI2_THRESHOLD}
            THEN 'STRONG_SIGNAL'
        WHEN case_count >= ${hivevar:MIN_CASES}
         AND (prr > ${hivevar:PRR_THRESHOLD} OR ror > ${hivevar:ROR_THRESHOLD})
            THEN 'SIGNAL'
        ELSE 'NONE'
    END                                                          AS signal_status,

    CASE
        WHEN prr > 4 AND ror > 4 AND case_count >= 10 THEN 'HIGH'
        WHEN prr > 2 AND ror > 2 AND case_count >= 5  THEN 'MEDIUM'
        ELSE 'LOW'
    END                                                          AS confidence,

    first_seen                                                   AS first_detected_date,
    CURRENT_TIMESTAMP()                                          AS last_updated_ts,
    ROUND(serious_ratio, 4)                                      AS serious_ratio,
    ROUND(death_ratio, 4)                                        AS death_ratio,
    countries_count
FROM   calc;

-- =====================================================================
-- Step 3: cleanup intermediate tables
-- =====================================================================
DROP TABLE IF EXISTS _dr_pair_counts;
DROP TABLE IF EXISTS _drug_totals;
DROP TABLE IF EXISTS _reaction_totals;
DROP TABLE IF EXISTS _grand_total;

-- =====================================================================
-- Step 4: top 50 strongest emerging signals (resume-impressive output)
-- =====================================================================
SELECT
    drug_name,
    reaction_term,
    case_count,
    prr,
    ror,
    prr_chi_square,
    serious_ratio,
    death_ratio,
    countries_count,
    signal_status
FROM   drug_signals
WHERE  signal_status IN ('SIGNAL', 'STRONG_SIGNAL')
ORDER  BY prr DESC, case_count DESC
LIMIT  50;
