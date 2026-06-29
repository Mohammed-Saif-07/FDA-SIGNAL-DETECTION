"""
streamlit_app.py
================
Cloud-friendly version of the FDA Signal Detection dashboard.
Reads pre-exported CSVs (dashboard/data/*.csv) — no DB required.
Deployed at share.streamlit.io.
"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="FDA Signal Detection",
    page_icon="💊",
    layout="wide",
)

DATA = Path(__file__).parent / "dashboard" / "data"


@st.cache_data
def load(name: str) -> pd.DataFrame:
    p = DATA / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


signals = load("signals.csv")
preds   = load("predictions.csv")
bt      = load("backtests.csv")

st.sidebar.title("FDA Pharma Pipeline")
st.sidebar.caption("Built by Saif Mohammed · Seattle University · MSCSDS")
st.sidebar.markdown(
    "[GitHub repo](https://github.com/Mohammed-Saif-07/FDA-SIGNAL-DETECTION)"
)
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Signal Explorer", "Prediction Leaderboard",
     "Backtesting Results", "About"],
)

# ================================================================== #
if page == "Overview":
    st.title("FDA Drug Safety Signal Detection — Overview")
    st.markdown(
        "Processing **20M+ FAERS adverse-event reports** through Hive/HQL + "
        "PySpark + XGBoost to surface drug safety signals before they "
        "become official FDA warnings."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drug-reaction signals", f"{len(signals):,}")
    c2.metric("STRONG signals",
              int((signals["signal_status"] == "STRONG_SIGNAL").sum())
              if "signal_status" in signals else 0)
    c3.metric("HIGH confidence",
              int((signals["confidence"] == "HIGH").sum())
              if "confidence" in signals else 0)
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

# ================================================================== #
elif page == "Signal Explorer":
    st.title("Signal Explorer")
    if signals.empty:
        st.info("No signals data yet — run dashboard/export_data.py first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        drug = c1.text_input("Drug contains", "")
        rx = c2.text_input("Reaction contains", "")
        status_opts = ["ANY"] + (
            sorted(signals["signal_status"].dropna().unique().tolist())
            if "signal_status" in signals else []
        )
        conf_opts = ["ANY"] + (
            sorted(signals["confidence"].dropna().unique().tolist())
            if "confidence" in signals else []
        )
        status = c3.selectbox("Status", status_opts)
        conf = c4.selectbox("Confidence", conf_opts)

        df = signals.copy()
        if drug:
            df = df[df["drug_name"].str.contains(drug, case=False, na=False)]
        if rx:
            df = df[df["reaction_term"].str.contains(rx, case=False, na=False)]
        if status != "ANY":
            df = df[df["signal_status"] == status]
        if conf != "ANY":
            df = df[df["confidence"] == conf]

        st.caption(f"{len(df):,} rows")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if not df.empty and "prr" in df:
            st.subheader("PRR distribution")
            st.plotly_chart(
                px.histogram(df.query("prr < 1000"), x="prr", nbins=40,
                             title="PRR distribution"),
                use_container_width=True,
            )

# ================================================================== #
elif page == "Prediction Leaderboard":
    st.title("Prediction Leaderboard")
    st.caption("Top XGBoost picks — most likely to become FDA warnings")

    if preds.empty:
        st.info("No predictions data yet — run dashboard/export_data.py first.")
    else:
        n = st.slider("Show top N", 10, min(200, len(preds)), 50, step=10)
        top = preds.nlargest(n, "recall_probability")
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.subheader("Top 25 predicted future FDA warnings")
        chart = top.head(25).copy()
        chart["signal"] = chart["drug_name"] + " — " + chart["reaction_term"]
        st.plotly_chart(
            px.bar(chart, x="recall_probability", y="signal",
                   orientation="h", color="recall_probability",
                   color_continuous_scale="Reds"),
            use_container_width=True,
        )

# ================================================================== #
elif page == "Backtesting Results":
    st.title("Backtesting — did we catch real FDA warnings early?")

    if bt.empty:
        st.info("No backtest data yet.")
    else:
        latest = bt.iloc[0]
        st.success(f"**Headline:** {latest.get('notes', 'n/a')}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Median days early",
                  int(latest["median_days_early"]) if pd.notna(latest.get("median_days_early")) else "—")
        c2.metric("Warnings caught",
                  f"{int(latest['warnings_caught'])} / {int(latest['warnings_total'])}"
                  if pd.notna(latest.get("warnings_caught")) else "—")
        c3.metric("Precision @ 100",
                  f"{latest['precision_at_100']:.2%}"
                  if pd.notna(latest.get("precision_at_100")) else "—")

        st.subheader("Recent backtest runs")
        st.dataframe(bt, use_container_width=True, hide_index=True)

# ================================================================== #
else:  # About
    st.title("About this project")
    st.markdown(
        """
A distributed pharmacovigilance pipeline that processes **20M+ FDA FAERS
adverse-event reports** with Apache Hive/HQL + PySpark + XGBoost to
detect emerging drug safety signals — replicating the methodology FDA's
own FAERS monitoring division uses (PRR + ROR disproportionality
analysis), and predicting which signals will become official FDA
warnings months before FDA announces them.

**Stack:** Hadoop HDFS · Apache Hive · PySpark 3.5 · XGBoost 2.0 ·
PostgreSQL · Apache Airflow · FastAPI · Streamlit · Docker Compose.


**Source:** [github.com/Mohammed-Saif-07/FDA-SIGNAL-DETECTION](https://github.com/Mohammed-Saif-07/FDA-SIGNAL-DETECTION)

**Author:** Saif Mohammed — MSCSDS, Seattle University · smohammed8@seattleu.edu
        """
    )