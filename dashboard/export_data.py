"""
dashboard/export_data.py
========================
Dump PostgreSQL tables to CSV so Streamlit Cloud can read without DB.
"""
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

engine = create_engine(
    "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
        u=os.getenv("POSTGRES_USER", "fda"),
        p=os.getenv("POSTGRES_PASSWORD", "fda"),
        h=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        db=os.getenv("POSTGRES_DB", "fda_signals"),
    )
)

signals = pd.read_sql(
    "SELECT * FROM pharma.drug_signals ORDER BY prr DESC NULLS LAST LIMIT 2000",
    engine,
)
signals.to_csv(OUT / "signals.csv", index=False)
print(f"signals.csv     -> {len(signals):,} rows")

preds = pd.read_sql(
    "SELECT * FROM pharma.signal_predictions ORDER BY recall_probability DESC LIMIT 2000",
    engine,
)
preds.to_csv(OUT / "predictions.csv", index=False)
print(f"predictions.csv -> {len(preds):,} rows")

bt = pd.read_sql("SELECT * FROM pharma.backtest_results ORDER BY run_date DESC", engine)
bt.to_csv(OUT / "backtests.csv", index=False)
print(f"backtests.csv   -> {len(bt):,} rows")

print(f"\nAll CSVs in {OUT}")