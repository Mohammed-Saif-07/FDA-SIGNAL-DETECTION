"""Streamlit Cloud entrypoint.

The full dashboard lives in dashboard/app.py. Streamlit Cloud is configured to
run this root file, so execute the real app from here.
"""

from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "dashboard" / "app.py"
exec(compile(APP_PATH.read_text(), str(APP_PATH), "exec"))
