"""
api/main.py
===========
FastAPI service exposing FDA Signal Detection results.

Run locally:
    uvicorn api.main:app --reload --port 8000

Or via docker-compose:
    http://localhost:8000/docs
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text


def get_engine():
    return create_engine(
        "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
            u=os.getenv("POSTGRES_USER", "fda"),
            p=os.getenv("POSTGRES_PASSWORD", "fda"),
            h=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            db=os.getenv("POSTGRES_DB", "fda_signals"),
        ),
        pool_pre_ping=True,
    )


app = FastAPI(
    title="FDA Drug Safety Signal Detection API",
    description=(
        "Read access to detected pharmacovigilance signals (PRR/ROR) and "
        "XGBoost recall-probability predictions over 10M+ FDA FAERS reports."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------- #
# Schemas                                                                 #
# ---------------------------------------------------------------------- #
class Signal(BaseModel):
    drug_name: str
    reaction_term: str
    case_count: int
    prr: Optional[float] = None
    ror: Optional[float] = None
    prr_chi_square: Optional[float] = None
    signal_status: Optional[str] = None
    confidence: Optional[str] = None
    serious_ratio: Optional[float] = None
    death_ratio: Optional[float] = None
    countries_count: Optional[int] = None
    first_detected_date: Optional[date] = None


class Prediction(BaseModel):
    drug_name: str
    reaction_term: str
    recall_probability: float
    predicted_class: int
    predicted_date: Optional[date] = None
    actual_fda_warning_date: Optional[date] = None
    days_predicted_early: Optional[int] = None


class BacktestSummary(BaseModel):
    headline: str
    train_cutoff_date: Optional[date] = None
    recall_overall: Optional[float] = None
    precision_at_100: Optional[float] = None
    median_days_early: Optional[int] = None
    warnings_caught: Optional[int] = None
    warnings_total: Optional[int] = None


# ---------------------------------------------------------------------- #
# Endpoints                                                               #
# ---------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    try:
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"db unavailable: {exc}")


@app.get("/signals", response_model=list[Signal])
def list_signals(
    drug: Optional[str] = None,
    reaction: Optional[str] = None,
    status: Optional[str] = Query(None, description="SIGNAL or STRONG_SIGNAL"),
    confidence: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List detected signals with optional filters, ordered by PRR desc."""
    q = "SELECT * FROM pharma.drug_signals WHERE 1=1"
    p: dict = {}
    if drug:
        q += " AND UPPER(drug_name) LIKE :drug"
        p["drug"] = f"%{drug.upper()}%"
    if reaction:
        q += " AND UPPER(reaction_term) LIKE :reaction"
        p["reaction"] = f"%{reaction.upper()}%"
    if status:
        q += " AND signal_status = :status"
        p["status"] = status
    if confidence:
        q += " AND confidence = :confidence"
        p["confidence"] = confidence
    q += " ORDER BY prr DESC NULLS LAST LIMIT :lim OFFSET :off"
    p["lim"] = limit
    p["off"] = offset

    with get_engine().connect() as c:
        rows = c.execute(text(q), p).mappings().all()
    return [Signal(**dict(r)) for r in rows]


@app.get("/signals/{drug}/{reaction}", response_model=Signal)
def get_signal(drug: str, reaction: str):
    with get_engine().connect() as c:
        row = c.execute(
            text(
                "SELECT * FROM pharma.drug_signals "
                "WHERE UPPER(drug_name)=:d AND UPPER(reaction_term)=:r"
            ),
            {"d": drug.upper(), "r": reaction.upper()},
        ).mappings().first()
    if not row:
        raise HTTPException(404, "signal not found")
    return Signal(**dict(row))


@app.get("/predictions", response_model=list[Prediction])
def list_predictions(top: int = 50, min_prob: float = 0.0):
    """Top-N XGBoost predictions of which signals will become FDA warnings."""
    q = (
        "SELECT * FROM pharma.signal_predictions "
        "WHERE recall_probability >= :p "
        "ORDER BY recall_probability DESC LIMIT :n"
    )
    with get_engine().connect() as c:
        rows = c.execute(text(q), {"p": min_prob, "n": top}).mappings().all()
    return [Prediction(**dict(r)) for r in rows]


@app.get("/backtest", response_model=BacktestSummary)
def backtest_summary():
    """Latest back-test result — headline metric for resume/recruiter."""
    with get_engine().connect() as c:
        row = c.execute(
            text(
                "SELECT * FROM pharma.backtest_results "
                "ORDER BY run_date DESC LIMIT 1"
            )
        ).mappings().first()
    if not row:
        return BacktestSummary(headline="No back-test run yet")
    return BacktestSummary(
        headline=row["notes"] or "",
        train_cutoff_date=row["train_cutoff_date"],
        recall_overall=float(row["recall_overall"]) if row["recall_overall"] else None,
        precision_at_100=float(row["precision_at_100"]) if row["precision_at_100"] else None,
        median_days_early=row["median_days_early"],
        warnings_caught=row["warnings_caught"],
        warnings_total=row["warnings_total"],
    )


@app.get("/stats")
def overall_stats():
    """Headline numbers for the dashboard."""
    sql = {
        "signals_total":   "SELECT COUNT(*) FROM pharma.drug_signals",
        "strong_signals":  "SELECT COUNT(*) FROM pharma.drug_signals WHERE signal_status='STRONG_SIGNAL'",
        "high_confidence": "SELECT COUNT(*) FROM pharma.drug_signals WHERE confidence='HIGH'",
        "predictions":     "SELECT COUNT(*) FROM pharma.signal_predictions",
    }
    out = {}
    with get_engine().connect() as c:
        for k, q in sql.items():
            out[k] = c.execute(text(q)).scalar()
    return out
