-- =====================================================================
-- PostgreSQL schema for FDA Drug Safety Signal Detection results
-- Auto-loaded by the postgres container on first start.
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS pharma;

-- ----------------------- DETECTED SIGNALS -----------------------------
CREATE TABLE IF NOT EXISTS pharma.drug_signals (
    id                  BIGSERIAL PRIMARY KEY,
    drug_name           TEXT        NOT NULL,
    reaction_term       TEXT        NOT NULL,
    case_count          INTEGER     NOT NULL,
    drug_total          INTEGER     NOT NULL,
    reaction_total      INTEGER     NOT NULL,
    grand_total         BIGINT      NOT NULL,
    prr                 NUMERIC(10,4),
    ror                 NUMERIC(10,4),
    prr_chi_square      NUMERIC(10,4),
    signal_status       TEXT,            -- 'SIGNAL', 'STRONG_SIGNAL', 'NONE'
    confidence          TEXT,            -- 'HIGH','MEDIUM','LOW'
    first_detected_date DATE,
    last_updated        TIMESTAMP DEFAULT NOW(),
    serious_ratio       NUMERIC(6,4),
    death_ratio         NUMERIC(6,4),
    countries_count     INTEGER,
    source_proxy_count  INTEGER,
    passes_robust_filter BOOLEAN DEFAULT FALSE,
    artifact_reason     TEXT,
    robust_signal_score NUMERIC(14,4),
    UNIQUE (drug_name, reaction_term)
);

CREATE INDEX IF NOT EXISTS idx_signals_drug      ON pharma.drug_signals(drug_name);
CREATE INDEX IF NOT EXISTS idx_signals_status    ON pharma.drug_signals(signal_status);
CREATE INDEX IF NOT EXISTS idx_signals_prr       ON pharma.drug_signals(prr DESC);
CREATE INDEX IF NOT EXISTS idx_signals_robust    ON pharma.drug_signals(passes_robust_filter, robust_signal_score DESC);

-- ----------------------- ML PREDICTIONS -------------------------------
CREATE TABLE IF NOT EXISTS pharma.signal_predictions (
    id                       BIGSERIAL PRIMARY KEY,
    drug_name                TEXT NOT NULL,
    reaction_term            TEXT NOT NULL,
    recall_probability       NUMERIC(6,4),
    predicted_class          INTEGER,    -- 0/1
    predicted_date           DATE        DEFAULT CURRENT_DATE,
    actual_fda_warning_date  DATE,
    days_predicted_early     INTEGER,
    model_version            TEXT,
    UNIQUE (drug_name, reaction_term, model_version)
);
CREATE INDEX IF NOT EXISTS idx_pred_prob ON pharma.signal_predictions(recall_probability DESC);

-- ----------------------- OFFICIAL FDA WARNINGS ------------------------
CREATE TABLE IF NOT EXISTS pharma.fda_official_warnings (
    id              BIGSERIAL PRIMARY KEY,
    drug_name       TEXT NOT NULL,
    reaction_term   TEXT NOT NULL,
    warning_date    DATE,
    warning_type    TEXT,    -- 'BOXED_WARNING','SAFETY_COMMUNICATION','RECALL'
    recall_date     DATE,
    source_url      TEXT,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_warn_drug ON pharma.fda_official_warnings(drug_name);

-- ----------------------- TREND HISTORY --------------------------------
CREATE TABLE IF NOT EXISTS pharma.signal_trends (
    id              BIGSERIAL PRIMARY KEY,
    drug_name       TEXT NOT NULL,
    reaction_term   TEXT NOT NULL,
    report_year     INTEGER,
    report_quarter  INTEGER,
    case_count      INTEGER,
    prr             NUMERIC(10,4),
    ror             NUMERIC(10,4)
);
CREATE INDEX IF NOT EXISTS idx_trend_dr ON pharma.signal_trends(drug_name, reaction_term);

-- ----------------------- BACKTEST RESULTS -----------------------------
CREATE TABLE IF NOT EXISTS pharma.backtest_results (
    id                  BIGSERIAL PRIMARY KEY,
    run_date            TIMESTAMP DEFAULT NOW(),
    model_version       TEXT,
    train_cutoff_date   DATE,
    auc_roc             NUMERIC(6,4),
    precision_at_100    NUMERIC(6,4),
    recall_overall      NUMERIC(6,4),
    median_days_early   INTEGER,
    warnings_caught     INTEGER,
    warnings_total      INTEGER,
    notes               TEXT
);
