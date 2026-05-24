-- =====================================================================
-- signal_trends.hql
-- Time-series, geographic, and demographic decomposition of signals.
--
-- Run AFTER signal_detection.hql.
-- =====================================================================

USE fda_pharma;

SET hive.exec.parallel = true;
SET hive.vectorized.execution.enabled = true;

-- =====================================================================
-- 1) Per-quarter PRR/ROR trend for every confirmed signal pair
--    Lets the dashboard plot "signal strength over time".
-- =====================================================================
DROP TABLE IF EXISTS signal_trends;
CREATE TABLE signal_trends STORED AS PARQUET AS
WITH per_quarter AS (
    SELECT
        drug_name,
        reaction_term,
        report_year,
        report_quarter,
        COUNT(DISTINCT safetyreportid) AS case_count
    FROM   raw_adverse_events
    WHERE  drug_name IS NOT NULL AND reaction_term IS NOT NULL
    GROUP  BY drug_name, reaction_term, report_year, report_quarter
),
drug_q AS (
    SELECT drug_name, report_year, report_quarter,
           SUM(case_count) AS drug_total_q
    FROM   per_quarter
    GROUP  BY drug_name, report_year, report_quarter
),
rx_q AS (
    SELECT reaction_term, report_year, report_quarter,
           SUM(case_count) AS rx_total_q
    FROM   per_quarter
    GROUP  BY reaction_term, report_year, report_quarter
),
total_q AS (
    SELECT report_year, report_quarter, SUM(case_count) AS grand_total_q
    FROM   per_quarter
    GROUP  BY report_year, report_quarter
)
SELECT
    p.drug_name,
    p.reaction_term,
    p.report_year,
    p.report_quarter,
    p.case_count,
    CASE
        WHEN (dq.drug_total_q - p.case_count) = 0
          OR (rq.rx_total_q   - p.case_count) = 0 THEN NULL
        ELSE (CAST(p.case_count AS DOUBLE) / dq.drug_total_q) /
             ((rq.rx_total_q - p.case_count) /
              CAST(tq.grand_total_q - dq.drug_total_q AS DOUBLE))
    END AS prr,
    CASE
        WHEN (dq.drug_total_q - p.case_count) = 0
          OR (rq.rx_total_q   - p.case_count) = 0 THEN NULL
        ELSE (CAST(p.case_count AS DOUBLE)
              * (tq.grand_total_q - dq.drug_total_q - rq.rx_total_q + p.case_count))
             /
             (CAST(dq.drug_total_q - p.case_count AS DOUBLE)
              * (rq.rx_total_q - p.case_count))
    END AS ror
FROM   per_quarter p
JOIN   drug_q   dq ON dq.drug_name=p.drug_name
                  AND dq.report_year=p.report_year
                  AND dq.report_quarter=p.report_quarter
JOIN   rx_q     rq ON rq.reaction_term=p.reaction_term
                  AND rq.report_year=p.report_year
                  AND rq.report_quarter=p.report_quarter
JOIN   total_q  tq ON tq.report_year=p.report_year
                  AND tq.report_quarter=p.report_quarter;

-- =====================================================================
-- 2) Geographic breakdown — which countries drive the top signals?
-- =====================================================================
DROP TABLE IF EXISTS signal_geo_breakdown;
CREATE TABLE signal_geo_breakdown STORED AS PARQUET AS
SELECT
    e.drug_name,
    e.reaction_term,
    COALESCE(e.country, 'UNK') AS country,
    COUNT(DISTINCT e.safetyreportid)                              AS reports,
    SUM(CAST(e.death_int AS INT))                                 AS deaths
FROM   raw_adverse_events e
JOIN   drug_signals s
       ON s.drug_name = e.drug_name AND s.reaction_term = e.reaction_term
WHERE  s.signal_status IN ('SIGNAL', 'STRONG_SIGNAL')
GROUP  BY e.drug_name, e.reaction_term, COALESCE(e.country, 'UNK');

-- =====================================================================
-- 3) Demographic decomposition — age band & sex
-- =====================================================================
DROP TABLE IF EXISTS signal_demographics;
CREATE TABLE signal_demographics STORED AS PARQUET AS
SELECT
    e.drug_name,
    e.reaction_term,
    CASE
        WHEN e.age_years IS NULL          THEN 'unknown'
        WHEN e.age_years < 18             THEN '<18'
        WHEN e.age_years BETWEEN 18 AND 39 THEN '18-39'
        WHEN e.age_years BETWEEN 40 AND 64 THEN '40-64'
        WHEN e.age_years BETWEEN 65 AND 84 THEN '65-84'
        ELSE '85+'
    END                                                              AS age_band,
    CASE e.patient_sex_int
        WHEN 1 THEN 'male'
        WHEN 2 THEN 'female'
        ELSE 'unknown'
    END                                                              AS sex,
    COUNT(DISTINCT e.safetyreportid)                                 AS reports
FROM   raw_adverse_events e
JOIN   drug_signals s
       ON s.drug_name = e.drug_name AND s.reaction_term = e.reaction_term
WHERE  s.signal_status IN ('SIGNAL', 'STRONG_SIGNAL')
GROUP  BY e.drug_name, e.reaction_term,
         CASE
            WHEN e.age_years IS NULL          THEN 'unknown'
            WHEN e.age_years < 18             THEN '<18'
            WHEN e.age_years BETWEEN 18 AND 39 THEN '18-39'
            WHEN e.age_years BETWEEN 40 AND 64 THEN '40-64'
            WHEN e.age_years BETWEEN 65 AND 84 THEN '65-84'
            ELSE '85+'
         END,
         CASE e.patient_sex_int WHEN 1 THEN 'male' WHEN 2 THEN 'female' ELSE 'unknown' END;

-- =====================================================================
-- 4) Compare detected signals vs official FDA warnings (validation)
--    A signal that matches an FDA warning is a "true positive".
-- =====================================================================
SELECT
    s.drug_name,
    s.reaction_term,
    s.prr,
    s.ror,
    s.case_count,
    s.first_detected_date,
    w.warning_date            AS official_warning_date,
    w.warning_type,
    DATEDIFF(w.warning_date, s.first_detected_date) AS days_we_were_early
FROM   drug_signals s
JOIN   fda_official_warnings w
       ON UPPER(w.drug_name)     = s.drug_name
      AND UPPER(w.reaction_term) = s.reaction_term
WHERE  s.signal_status IN ('SIGNAL', 'STRONG_SIGNAL')
ORDER  BY days_we_were_early DESC;

-- =====================================================================
-- 5) Top 50 emerging signals NOT yet officially flagged by FDA
-- =====================================================================
SELECT
    s.drug_name,
    s.reaction_term,
    s.case_count,
    s.prr,
    s.ror,
    s.serious_ratio,
    s.death_ratio,
    s.confidence
FROM   drug_signals s
LEFT JOIN fda_official_warnings w
       ON UPPER(w.drug_name)     = s.drug_name
      AND UPPER(w.reaction_term) = s.reaction_term
WHERE  s.signal_status = 'STRONG_SIGNAL'
  AND  w.drug_name IS NULL                          -- not yet flagged
ORDER  BY s.prr DESC, s.case_count DESC
LIMIT  50;
