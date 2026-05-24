"""
dashboard/app.py
================
Streamlit dashboard for the FDA Drug Safety Signal Detection pipeline.

Five pages:
    1. Overview          (headline stats)
    2. Signal Explorer   (filterable table of PRR/ROR signals)
    3. Prediction Leaderboard (top XGBoost picks)
    4. Backtesting Results
    5. Big Data Stats    (rows processed, Hive timings)

Run locally:
    streamlit run dashboard/app.py
or via docker-compose: http://localhost:8501
"""

from __future__ import annotations

import os
import re
from datetime import date

import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="FDA Signal Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_connection_params():
    return dict(
        user=os.getenv("POSTGRES_USER", "fda"),
        password=os.getenv("POSTGRES_PASSWORD", "fda"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "fda_signals"),
    )


@st.cache_data(ttl=300)
def q(sql: str, **params) -> pd.DataFrame:
    try:
        # pandas 2.2 can mis-detect SQLAlchemy 1.4 engines in this local env,
        # so use psycopg2 directly and translate SQLAlchemy-style :name params.
        sql = re.sub(r":([A-Za-z_][A-Za-z0-9_]*)", r"%(\1)s", sql)
        with psycopg2.connect(**get_connection_params()) as c:
            return pd.read_sql(sql, c, params=params)
    except Exception as exc:
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


# ----------------------------------------------------------------------------
# Sidebar — page navigation
# ----------------------------------------------------------------------------
st.sidebar.title("FDA Pharma Pipeline")
st.sidebar.caption("Built by Saif Mohammed · Seattle University · MSCSDS")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Signal Explorer",
        "Prediction Leaderboard",
        "Backtesting Results",
        "Big Data Stats",
    ],
)


# ============================================================================
# Page 1 — Overview
# ============================================================================
def page_overview():
    st.title("FDA Drug Safety Signal Detection — Overview")
    st.markdown(
        "Continuously processing **10M+ FAERS adverse-event reports** through "
        "Hive/HQL + PySpark + XGBoost to surface drug safety signals before "
        "they become official FDA warnings."
    )

    stats = q("""
        SELECT
          (SELECT COUNT(*) FROM pharma.drug_signals) AS signals_total,
          (SELECT COUNT(*) FROM pharma.drug_signals WHERE signal_status='STRONG_SIGNAL') AS strong,
          (SELECT COUNT(*) FROM pharma.drug_signals WHERE confidence='HIGH') AS high_conf,
          (SELECT COUNT(*) FROM pharma.signal_predictions) AS preds,
          (SELECT COUNT(*) FROM pharma.fda_official_warnings) AS known_warnings
    """)
    if stats.empty:
        st.info("No data yet — run the pipeline once.")
        return

    s = stats.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total signals", f"{int(s['signals_total']):,}")
    c2.metric("STRONG signals", f"{int(s['strong']):,}")
    c3.metric("HIGH confidence", f"{int(s['high_conf']):,}")
    c4.metric("ML predictions", f"{int(s['preds']):,}")

    st.divider()
    st.subheader("Pipeline at a glance")
    st.markdown(
        "- **Storage:** HDFS (Hadoop) on Docker  \n"
        "- **SQL:** Apache Hive — partitioned Parquet, vectorised execution  \n"
        "- **Compute:** PySpark 3.5  \n"
        "- **ML:** XGBoost classifier predicting *signal → FDA warning*  \n"
        "- **Orchestration:** Airflow (`@quarterly`)  \n"
        "- **Results store:** PostgreSQL  \n"
    )


# ============================================================================
# Page 2 — Signal Explorer
# ============================================================================
def page_signal_explorer():
    st.title("Signal Explorer")
    col1, col2, col3, col4 = st.columns(4)
    drug = col1.text_input("Drug contains", "")
    rx   = col2.text_input("Reaction contains", "")
    status = col3.selectbox("Status", ["ANY", "STRONG_SIGNAL", "SIGNAL", "NONE"])
    conf   = col4.selectbox("Confidence", ["ANY", "HIGH", "MEDIUM", "LOW"])

    sql = "SELECT * FROM pharma.drug_signals WHERE 1=1"
    params = {}
    if drug:
        sql += " AND UPPER(drug_name) LIKE :drug"
        params["drug"] = f"%{drug.upper()}%"
    if rx:
        sql += " AND UPPER(reaction_term) LIKE :rx"
        params["rx"] = f"%{rx.upper()}%"
    if status != "ANY":
        sql += " AND signal_status = :status"
        params["status"] = status
    if conf != "ANY":
        sql += " AND confidence = :conf"
        params["conf"] = conf
    sql += " ORDER BY prr DESC NULLS LAST LIMIT 500"

    df = q(sql, **params)
    st.caption(f"{len(df)} rows")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if not df.empty:
        st.subheader("PRR distribution")
        st.plotly_chart(
            px.histogram(df, x="prr", nbins=40, title="PRR across results"),
            use_container_width=True,
        )


# ============================================================================
# Page 3 — Prediction Leaderboard
# ============================================================================
def page_predictions():
    st.title("Prediction Leaderboard")
    st.caption("Top XGBoost picks — most likely to become FDA warnings.")

    n = st.slider("Show top N", 10, 200, 50, step=10)
    df = q(
        """
        SELECT p.drug_name, p.reaction_term, p.recall_probability,
               p.predicted_date, p.actual_fda_warning_date,
               p.days_predicted_early,
               s.case_count, s.prr, s.ror
        FROM   pharma.signal_predictions p
        LEFT JOIN pharma.drug_signals s USING (drug_name, reaction_term)
        ORDER  BY p.recall_probability DESC
        LIMIT  :n
        """,
        n=n,
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    if not df.empty:
        chart_df = df.head(25).copy()
        chart_df["signal"] = chart_df["drug_name"] + " — " + chart_df["reaction_term"]
        chart_df = chart_df.sort_values("recall_probability", ascending=True)
        st.plotly_chart(
            px.bar(
                chart_df,
                x="recall_probability",
                y="signal",
                color="recall_probability",
                color_continuous_scale="Reds",
                orientation="h",
                title="Top 25 predicted future FDA warnings",
                range_x=[0, 1],
            ),
            use_container_width=True,
        )


# ============================================================================
# Page 4 — Backtesting
# ============================================================================
def page_backtesting():
    st.title("Backtesting — did we catch real FDA warnings early?")

    bt = q("SELECT * FROM pharma.backtest_results ORDER BY run_date DESC LIMIT 10")
    if bt.empty:
        st.info("Run `python ml/evaluate.py` first.")
        return

    latest = bt.iloc[0]
    headline = latest["notes"] or "—"
    st.success(f"**Headline:** {headline}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Median days early",  latest["median_days_early"] or "—")
    c2.metric("Warnings caught",
              f"{int(latest['warnings_caught'] or 0)} / "
              f"{int(latest['warnings_total'] or 0)}")
    c3.metric("Precision @ 100",
              f"{(latest['precision_at_100'] or 0):.2%}")

    st.subheader("Recent back-test runs")
    st.dataframe(bt[[
        "run_date", "model_version", "train_cutoff_date",
        "recall_overall", "precision_at_100",
        "median_days_early", "warnings_caught", "warnings_total",
    ]], use_container_width=True, hide_index=True)


# ============================================================================
# Page 5 — Big Data Stats
# ============================================================================
def page_bigdata():
    st.title("Big Data Stats")
    st.markdown(
        "These stats summarize the local PySpark/Parquet run. The HDFS/Hive "
        "files in the repo mirror the same schema and PRR/ROR calculations for "
        "the distributed version."
    )
    df = q("""
        SELECT
          MAX(grand_total) AS grand_total,
          COUNT(*) AS exported_signals,
          COUNT(*) FILTER (WHERE confidence='HIGH') AS high_confidence
        FROM pharma.drug_signals
        LIMIT 1
    """)
    if df.empty:
        st.info("No signal rows loaded yet.")
        return

    row = df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Drug/reaction rows scanned", f"{int(row['grand_total'] or 0):,}")
    c2.metric("Signals exported to dashboard", f"{int(row['exported_signals'] or 0):,}")
    c3.metric("High-confidence signals", f"{int(row['high_confidence'] or 0):,}")

    st.markdown(
        "- HDFS UI: [localhost:9870](http://localhost:9870)  \n"
        "- Spark UI: [localhost:8080](http://localhost:8080)  \n"
        "- Airflow: [localhost:8081](http://localhost:8081)  \n"
        "- API docs: [localhost:8000/docs](http://localhost:8000/docs)"
    )
    st.caption("All services run locally via docker-compose.")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
if   page == "Overview":               page_overview()
elif page == "Signal Explorer":        page_signal_explorer()
elif page == "Prediction Leaderboard": page_predictions()
elif page == "Backtesting Results":    page_backtesting()
elif page == "Big Data Stats":         page_bigdata()
