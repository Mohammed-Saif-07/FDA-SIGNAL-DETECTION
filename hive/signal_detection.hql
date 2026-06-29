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
SET hive.mapred.mode = nonstrict;
SET hive.strict.checks.cartesian.product = false;
SET hive.vectorized.execution.enabled = false;
SET hive.vectorized.execution.reduce.enabled = false;
SET hive.cbo.enable = true;
SET hive.compute.query.using.stats = true;
SET hive.auto.convert.join = true;
SET mapreduce.framework.name = local;
SET hive.exec.mode.local.auto = true;
SET mapreduce.task.io.sort.mb = 16;
SET mapreduce.map.java.opts = -Xmx512m;
SET mapreduce.reduce.java.opts = -Xmx512m;
SET mapred.child.java.opts = -Xmx512m;
SET mapreduce.input.fileinputformat.split.maxsize = 268435456;  -- 256 MB

-- =====================================================================
-- Step 0: parameters (override with `-hivevar` from the command line)
-- =====================================================================
SET hivevar:PRR_THRESHOLD = 2.0;
SET hivevar:ROR_THRESHOLD = 2.0;
SET hivevar:MIN_CASES     = 3;
SET hivevar:CHI2_THRESHOLD = 4.0;
SET hivevar:CHI2_CAP = 1000000000.0;

-- =====================================================================
-- Step 1: per-drug × per-reaction × per-margin counts
--         (only consider drugs flagged as SUSPECT — drug_role='1')
-- =====================================================================
DROP TABLE IF EXISTS tmp_dr_pair_counts;
CREATE TABLE tmp_dr_pair_counts STORED AS PARQUET AS
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

DROP TABLE IF EXISTS tmp_drug_totals;
CREATE TABLE tmp_drug_totals STORED AS PARQUET AS
SELECT drug_name, SUM(case_count) AS drug_total
FROM   tmp_dr_pair_counts
GROUP  BY drug_name;

DROP TABLE IF EXISTS tmp_reaction_totals;
CREATE TABLE tmp_reaction_totals STORED AS PARQUET AS
SELECT reaction_term, SUM(case_count) AS reaction_total
FROM   tmp_dr_pair_counts
GROUP  BY reaction_term;

DROP TABLE IF EXISTS tmp_grand_total;
CREATE TABLE tmp_grand_total STORED AS PARQUET AS
SELECT SUM(case_count) AS grand_total FROM tmp_dr_pair_counts;

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
    FROM       tmp_dr_pair_counts  p
    JOIN       tmp_drug_totals     dt ON dt.drug_name      = p.drug_name
    JOIN       tmp_reaction_totals rt ON rt.reaction_term  = p.reaction_term
    CROSS JOIN tmp_grand_total     gt
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
            ELSE (CAST(a AS DOUBLE) / (CAST(a AS DOUBLE) + CAST(b AS DOUBLE))) /
                 (CAST(c AS DOUBLE) / (CAST(c AS DOUBLE) + CAST(d AS DOUBLE)))
        END                                                 AS prr,

        /* ROR = (a*d) / (b*c)                       */
        CASE
            WHEN b = 0 OR c = 0 THEN NULL
            ELSE (CAST(a AS DOUBLE) * d) / (CAST(b AS DOUBLE) * c)
        END                                                 AS ror,

        /*  Yates-corrected chi-square.
            Cast all contingency-table cells before multiplication to avoid
            integer overflow on large FAERS tables. */
        CASE
            WHEN (a + b) = 0 OR (c + d) = 0 OR (a + c) = 0 OR (b + d) = 0 THEN NULL
            ELSE (
                POWER(
                    ABS(CAST(a AS DOUBLE) * CAST(d AS DOUBLE) - CAST(b AS DOUBLE) * CAST(c AS DOUBLE))
                    - (CAST(grand_total AS DOUBLE) / 2.0),
                    2
                )
                * CAST(grand_total AS DOUBLE)
              ) / (
                    (CAST(a AS DOUBLE) + CAST(b AS DOUBLE))
                  * (CAST(c AS DOUBLE) + CAST(d AS DOUBLE))
                  * (CAST(a AS DOUBLE) + CAST(c AS DOUBLE))
                  * (CAST(b AS DOUBLE) + CAST(d AS DOUBLE))
              )
        END                                                 AS prr_chi_square_raw,

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
    ROUND(LEAST(prr_chi_square_raw, ${hivevar:CHI2_CAP}), 4) AS prr_chi_square,
    CASE WHEN prr_chi_square_raw > ${hivevar:CHI2_CAP} THEN true ELSE false END AS prr_chi_square_capped,

    CASE
        WHEN case_count >= ${hivevar:MIN_CASES}
         AND prr > ${hivevar:PRR_THRESHOLD}
         AND ror > ${hivevar:ROR_THRESHOLD}
         AND LEAST(prr_chi_square_raw, ${hivevar:CHI2_CAP}) > ${hivevar:CHI2_THRESHOLD}
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
    countries_count,
    countries_count                                              AS source_proxy_count,
    CASE
        WHEN case_count >= 5
         AND countries_count >= 3
         AND serious_ratio >= 0.01
         AND prr BETWEEN 2 AND 100000
         AND ror BETWEEN 2 AND 100000
         AND LEAST(prr_chi_square_raw, ${hivevar:CHI2_CAP}) >= 4
            THEN true
        ELSE false
    END                                                          AS passes_robust_filter,
    CASE
        WHEN case_count < 3 OR prr <= 2 OR ror <= 2 OR LEAST(prr_chi_square_raw, ${hivevar:CHI2_CAP}) < 4 THEN 'not_prr_ror_signal'
        WHEN case_count < 5 THEN 'too_few_cases'
        WHEN countries_count < 3 THEN 'low_source_diversity_proxy'
        WHEN serious_ratio < 0.01 THEN 'low_seriousness_ratio'
        WHEN prr > 100000 OR ror > 100000 THEN 'extreme_ratio_artifact'
        WHEN prr_chi_square_raw < 0 THEN 'invalid_negative_chi_square'
        ELSE 'passes_robust_filter'
    END                                                          AS artifact_reason,
    CASE
        WHEN case_count >= 5
         AND countries_count >= 3
         AND serious_ratio >= 0.01
         AND prr BETWEEN 2 AND 100000
         AND ror BETWEEN 2 AND 100000
         AND LEAST(prr_chi_square_raw, ${hivevar:CHI2_CAP}) >= 4
            THEN LN(1 + case_count)
               * LN(1 + countries_count)
               * (1 + LEAST(GREATEST(serious_ratio, 0), 1) + 2 * LEAST(GREATEST(death_ratio, 0), 1))
               * LN(1 + prr)
               * LN(1 + ror)
        ELSE 0.0
    END                                                          AS robust_signal_score
FROM   calc;

-- =====================================================================
-- Step 3: cleanup intermediate tables
-- =====================================================================
DROP TABLE IF EXISTS tmp_dr_pair_counts;
DROP TABLE IF EXISTS tmp_drug_totals;
DROP TABLE IF EXISTS tmp_reaction_totals;
DROP TABLE IF EXISTS tmp_grand_total;

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
