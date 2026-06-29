-- =====================================================================
-- Robust signal ranking for FAERS PRR/ROR outputs.
--
-- Purpose:
--   The naive PRR/ROR threshold is sensitive but artifact-prone. This query
--   ranks only signals with enough case count, country/source-diversity proxy,
--   seriousness, finite ratio bounds, and non-negative chi-square.
--
-- Important limitation:
--   Public FAERS extracts do not provide a clean reporter-source identifier in
--   this flattened project table, so countries_count is used as a conservative
--   public-data source-diversity proxy. This is not claimed to be exact source
--   count.
-- =====================================================================

USE fda_pharma;

SET hivevar:MIN_CASES = 5;
SET hivevar:MIN_SOURCE_PROXY = 3;
SET hivevar:MIN_SERIOUS_RATIO = 0.01;
SET hivevar:MAX_RATIO = 100000.0;

DROP TABLE IF EXISTS robust_drug_signals;

CREATE TABLE robust_drug_signals STORED AS PARQUET AS
SELECT
    drug_name,
    reaction_term,
    case_count,
    drug_total,
    reaction_total,
    grand_total,
    prr,
    ror,
    prr_chi_square,
    serious_ratio,
    death_ratio,
    countries_count,
    countries_count AS source_proxy_count,
    first_detected_date,
    last_updated_ts,
    signal_status,
    confidence,
    true AS passes_robust_filter,
    'passes_robust_filter' AS artifact_reason,
    (
        LN(1 + case_count)
        * LN(1 + countries_count)
        * (1 + LEAST(GREATEST(COALESCE(serious_ratio, 0.0), 0.0), 1.0)
             + 2 * LEAST(GREATEST(COALESCE(death_ratio, 0.0), 0.0), 1.0))
        * LN(1 + prr)
        * LN(1 + ror)
    ) AS robust_signal_score
FROM drug_signals
WHERE signal_status IN ('SIGNAL', 'STRONG_SIGNAL')
  AND case_count >= ${hivevar:MIN_CASES}
  AND countries_count >= ${hivevar:MIN_SOURCE_PROXY}
  AND COALESCE(serious_ratio, 0.0) >= ${hivevar:MIN_SERIOUS_RATIO}
  AND prr BETWEEN 2.0 AND ${hivevar:MAX_RATIO}
  AND ror BETWEEN 2.0 AND ${hivevar:MAX_RATIO}
  AND prr_chi_square >= 4.0;

SELECT
    drug_name,
    reaction_term,
    case_count,
    countries_count AS source_proxy_count,
    prr,
    ror,
    prr_chi_square,
    serious_ratio,
    death_ratio,
    robust_signal_score
FROM robust_drug_signals
ORDER BY robust_signal_score DESC, prr_chi_square DESC, case_count DESC
LIMIT 50;
