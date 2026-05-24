"""
api/report.py
=============
Generate a quarterly PDF summary of detected signals.

Used by Airflow's `generate_report` task.

Run standalone:
    python -c "from api.report import build_pdf_report; build_pdf_report()"
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib import colors
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"


def _engine():
    return create_engine(
        "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
            u=os.getenv("POSTGRES_USER", "fda"),
            p=os.getenv("POSTGRES_PASSWORD", "fda"),
            h=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            db=os.getenv("POSTGRES_DB", "fda_signals"),
        )
    )


def build_pdf_report(out_path: Path | None = None) -> Path:
    out_path = out_path or (OUT_DIR / f"signal_report_{date.today().isoformat()}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    story = []

    story.append(Paragraph("FDA FAERS Drug Safety Signal Detection Report",
                           styles["Title"]))
    story.append(Paragraph(f"Generated: {date.today().isoformat()}", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    eng = _engine()
    with eng.connect() as c:
        stats = {
            "signals_total":   c.execute(text("SELECT COUNT(*) FROM pharma.drug_signals")).scalar() or 0,
            "strong_signals":  c.execute(text(
                "SELECT COUNT(*) FROM pharma.drug_signals WHERE signal_status='STRONG_SIGNAL'"
            )).scalar() or 0,
            "predictions":     c.execute(text("SELECT COUNT(*) FROM pharma.signal_predictions")).scalar() or 0,
        }
        top_signals = c.execute(text("""
            SELECT drug_name, reaction_term, case_count, prr, ror, confidence
            FROM   pharma.drug_signals
            WHERE  signal_status='STRONG_SIGNAL'
            ORDER  BY prr DESC NULLS LAST
            LIMIT  25
        """)).mappings().all()

        backtest = c.execute(text(
            "SELECT * FROM pharma.backtest_results ORDER BY run_date DESC LIMIT 1"
        )).mappings().first()

    story.append(Paragraph(
        f"<b>Signals detected:</b> {stats['signals_total']:,} "
        f"(<b>STRONG:</b> {stats['strong_signals']:,})", styles["Normal"]))
    story.append(Paragraph(f"<b>ML predictions:</b> {stats['predictions']:,}",
                           styles["Normal"]))

    if backtest:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("<b>Back-test headline</b>", styles["Heading2"]))
        story.append(Paragraph(backtest.get("notes") or "n/a", styles["Normal"]))

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Top 25 STRONG signals (by PRR)", styles["Heading2"]))

    header = ["Drug", "Reaction", "Cases", "PRR", "ROR", "Confidence"]
    table_data = [header] + [
        [r["drug_name"][:25], r["reaction_term"][:35],
         r["case_count"], f"{r['prr']:.2f}" if r["prr"] else "-",
         f"{r['ror']:.2f}" if r["ror"] else "-", r["confidence"] or "-"]
        for r in top_signals
    ]
    table = Table(table_data, colWidths=[1.4*inch, 2.1*inch, 0.6*inch,
                                         0.6*inch, 0.6*inch, 0.9*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E4FB7")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.HexColor("#F0F4FB")]),
    ]))
    story.append(table)

    doc.build(story)
    print(f"Wrote PDF report -> {out_path}")
    return out_path


if __name__ == "__main__":
    build_pdf_report()
