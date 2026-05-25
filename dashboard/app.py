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
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import psycopg2
except ImportError:  # Streamlit Cloud demo can run from CSV snapshots only.
    psycopg2 = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

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


def use_postgres() -> bool:
    return bool(os.getenv("POSTGRES_HOST")) and psycopg2 is not None


@st.cache_data(ttl=300)
def q(sql: str, **params) -> pd.DataFrame:
    if not use_postgres():
        return pd.DataFrame()

    try:
        # pandas 2.2 can mis-detect SQLAlchemy 1.4 engines in this local env,
        # so use psycopg2 directly and translate SQLAlchemy-style :name params.
        sql = re.sub(r":([A-Za-z_][A-Za-z0-9_]*)", r"%(\1)s", sql)
        with psycopg2.connect(**get_connection_params()) as c:
            return pd.read_sql(sql, c, params=params)
    except Exception as exc:
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def csv_table(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_signals() -> pd.DataFrame:
    if use_postgres():
        df = q("SELECT * FROM pharma.drug_signals")
        if not df.empty:
            return df
    return csv_table("signals.csv")


def load_predictions() -> pd.DataFrame:
    if use_postgres():
        df = q("""
            SELECT p.drug_name, p.reaction_term, p.recall_probability,
                   p.predicted_date, p.actual_fda_warning_date,
                   p.days_predicted_early, s.case_count, s.prr, s.ror
            FROM pharma.signal_predictions p
            LEFT JOIN pharma.drug_signals s USING (drug_name, reaction_term)
            ORDER BY p.recall_probability DESC
        """)
        if not df.empty:
            return df

    preds = csv_table("predictions.csv")
    signals = csv_table("signals.csv")
    if preds.empty:
        return preds
    if signals.empty:
        return preds
    keep = ["drug_name", "reaction_term", "case_count", "prr", "ror"]
    return preds.merge(signals[keep], on=["drug_name", "reaction_term"], how="left")


def load_backtests() -> pd.DataFrame:
    if use_postgres():
        df = q("SELECT * FROM pharma.backtest_results ORDER BY run_date DESC LIMIT 10")
        if not df.empty:
            return df
    df = csv_table("backtests.csv")
    if not df.empty and "run_date" in df.columns:
        df = df.sort_values("run_date", ascending=False).head(10)
    return df


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

    signals = load_signals()
    preds = load_predictions()
    if signals.empty and preds.empty:
        st.info("No data yet — run the pipeline once.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total signals", f"{len(signals):,}")
    c2.metric(
        "STRONG signals",
        f"{int((signals.get('signal_status') == 'STRONG_SIGNAL').sum()):,}"
        if not signals.empty else "0",
    )
    c3.metric(
        "HIGH confidence",
        f"{int((signals.get('confidence') == 'HIGH').sum()):,}"
        if not signals.empty else "0",
    )
    c4.metric("ML predictions", f"{len(preds):,}")

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

    df = load_signals()
    if drug:
        df = df[df["drug_name"].str.upper().str.contains(drug.upper(), na=False)]
    if rx:
        df = df[df["reaction_term"].str.upper().str.contains(rx.upper(), na=False)]
    if status != "ANY":
        df = df[df["signal_status"] == status]
    if conf != "ANY":
        df = df[df["confidence"] == conf]

    if not df.empty:
        df = df.sort_values("prr", ascending=False, na_position="last").head(500)
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
    df = load_predictions()
    if not df.empty:
        df = df.sort_values("recall_probability", ascending=False).head(n)
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

    bt = load_backtests()
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
    df = load_signals()
    if df.empty:
        st.info("No signal rows loaded yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Drug/reaction rows scanned", f"{int(df['grand_total'].max() or 0):,}")
    c2.metric("Signals exported to dashboard", f"{len(df):,}")
    c3.metric("High-confidence signals", f"{int((df['confidence'] == 'HIGH').sum()):,}")

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
