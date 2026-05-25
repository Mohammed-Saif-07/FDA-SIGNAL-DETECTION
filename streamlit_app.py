"""Streamlit Cloud entrypoint.

The full dashboard lives in dashboard/app.py. Streamlit Cloud is configured to
run this root file, so execute the real app from here.
"""

from pathlib import Path
import runpy

APP_PATH = Path(__file__).resolve().parent / "dashboard" / "app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
